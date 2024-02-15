from Models.ModelBasics import *
from math import sqrt


class EmbedAttention(nn.Module):
    def __init__(self, in_c: int, embed_c: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.H = num_heads

        self.qkv_proj = nn.Sequential(
            nn.GroupNorm(32, in_c + embed_c),
            Swish(),
            nn.Conv1d(in_c + embed_c, in_c * 3 * self.H, 3, 1, 1),
        )

        self.scale = 1 / sqrt(in_c)

        self.out_proj = nn.Sequential(
            nn.GroupNorm(32, in_c * self.H),
            nn.Dropout(dropout),
            nn.Conv1d(in_c * self.H, in_c, 3, 1, 1)
        )

        nn.init.zeros_(self.out_proj[-1].weight)

    def forward(self, x: torch.Tensor, mix_embed: torch.Tensor) -> torch.Tensor:
        # x: (B, C_x, L_x)
        # mix_embed: (B, C_m, 1)
        L_x = x.shape[2]

        q, k, v = self.qkv_proj(torch.cat([x, mix_embed.repeat(1, 1, L_x)], dim=1)).chunk(3, dim=1)
        q = rearrange(q, 'b (h c) l -> (b h) c l', h=self.H)  # (B*H, C, L)
        k = rearrange(k, 'b (h c) l -> (b h) l c', h=self.H)  # (B*H, L, C)
        v = rearrange(v, 'b (h c) l -> (b h) c l', h=self.H)  # (B*H, C, L)

        attn = torch.softmax(torch.bmm(k, q) * self.scale, dim=1)  # (B*H, L, L)

        return x + self.out_proj(rearrange(torch.bmm(v, attn), '(b h) c l -> b (h c) l', h=self.H))


class ChannelAttention(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 embed_c: int,
                 expand: int,
                 dropout: float = 0.1):
        super().__init__()

        self.mid_c = in_c * expand

        self.s1 = nn.Sequential(
            nn.GroupNorm(32, in_c + embed_c),
            Swish(),
            nn.Conv1d(in_c + embed_c, self.mid_c, 3, 1, 1),
        )

        self.channel_attn_proj = nn.Sequential(
            nn.Conv1d(embed_c, self.mid_c, 1, 1, 0),
            Swish(),
            nn.Conv1d(self.mid_c, self.mid_c, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.s2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.GroupNorm(32, self.mid_c),
            Swish(),
            nn.Conv1d(self.mid_c, out_c, 3, 1, 1),
        )

        self.shortcut = nn.Conv1d(in_c, out_c, 1, 1, 0) if in_c != out_c else nn.Identity()

        nn.init.zeros_(self.s2[-1].weight)

    def forward(self, x: torch.Tensor, mix_embed: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        # mix_embed: (B, C_m, 1)
        attn = self.channel_attn_proj(mix_embed)  # (B, mid_c, 1)
        residual = self.s2(self.s1(torch.cat([x, mix_embed.repeat(1, 1, x.shape[2])], dim=1)) * attn)
        return self.shortcut(x) + residual


class TWResBlock(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 embed_c: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()

        self.block1 = ChannelAttention(in_c, out_c, embed_c, num_heads, dropout)
        self.block2 = ChannelAttention(out_c, out_c, embed_c, num_heads, dropout)

    def forward(self, x: torch.Tensor, mix_embed: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, mix_embed)
        return self.block2(x, mix_embed)


class TWTransBlock(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 embed_c: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()

        self.token_attn_res_fuser = EmbedAttention(in_c, embed_c, num_heads, dropout)

        self.channel_attn_res_fuser = ChannelAttention(in_c, out_c, embed_c, 2, dropout)

    def forward(self, x: torch.Tensor, mix_embed: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        # traj_context: (B, C_c, L_c)
        # mix_embed: (B, C_m, 1)
        x = self.token_attn_res_fuser(x, mix_embed)  # merge the embedding information
        return self.channel_attn_res_fuser(x, mix_embed)





class TrajWeaver(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 down_schedule: List[int],
                 diffusion_steps: int,
                 channel_schedule: List[int],
                 n_res_blocks: List[int],
                 embed_c: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 max_length: int = 256,
                 self_attn: bool = True
                 ) -> None:
        """
        :param in_c:                * number of channels of the input trajectory representation
        :param down_schedule:       * downsample schedule of the UNet, example: [1, 2, 2] means keep length at first and
                                    then downsample by 2 for 2 times
        :param diffusion_steps:     * total diffusion steps
        :param channel_schedule:    * channel schedule of the UNet, example: [32, 64, 128, 256]
        :param n_res_blocks:        * number of residual blocks in each stage, example: [2, 2, 1]
        :param embedings_c:         * number of channels of the embedding vector
        :param embed_c:        * number of channels of the time embedding vector
        :param traj_context_c:      * number of channels of the broken trajectory context
        :param num_heads:           * number of heads in the multi-head attention
        :param dropout:             * dropout rate
        :param max_length:          * max length of the trajectory
        """
        super().__init__()

        self.down_schedule = down_schedule
        self.channel_schedule = channel_schedule
        self.stages = len(channel_schedule) - 1
        if self_attn:
            self.res_module = TWTransBlock
        else:
            self.res_module = TWResBlock

        self.res_params = {'embed_c': embed_c,
                           'num_heads': num_heads,
                           'dropout': dropout}

        # This block adds positional encoding and trajectory length encoding to the input trajectory
        self.pre_embed = PosLengthEncoder(in_c=in_c, out_c=channel_schedule[0], max_length=max_length,
                                          embed_dim=128, k=3, s=down_schedule[0], p=1)

        # This obtains the time embedding of the input trajectory (but not yet added)
        self.time_embedder = TimestepEncoder(max_time=diffusion_steps, hidden_dim=256, embed_dim=embed_c)
        self.length_embedder = nn.Sequential(
            nn.Embedding(max_length, 256),
            nn.Linear(256, embed_c),
        )

        # Create Encoder (Down sampling) Blocks for UNet
        in_channels = channel_schedule[:-1]
        out_channels = channel_schedule[1:]
        self.down_blocks = nn.ModuleList()
        for i in range(self.stages):
            self.down_blocks.append(self.__makeEncoderStage(in_channels[i], out_channels[i], self.down_schedule[i + 1],
                                                            n_res_blocks[i]))

        # Create Middle Attention Block for UNet
        self.mid_attn_block = TWTransBlock(in_c=out_channels[-1], out_c=out_channels[-1], **self.res_params)

        # Create Decoder (Up sampling) Blocks for UNet
        self.up_blocks = nn.ModuleList()
        # reverse the channel schedule
        in_channels = channel_schedule[-1:0:-1]
        out_channels = channel_schedule[-2::-1]
        for i in range(self.stages):
            self.up_blocks.append(self.__makeDecoderStage(in_channels[i] * 2, out_channels[i], n_res_blocks[-i]))

        self.head = nn.Conv1d(channel_schedule[0], out_c, 3, 1, 1)

    def __makeEncoderStage(self, in_c: int, out_c: int, downscale: int, n_res: int) -> nn.ModuleList:
        # Make one encoder stage
        layers = [self.res_module(in_c=in_c, out_c=out_c, **self.res_params),
                  nn.Conv1d(out_c, out_c, 3, downscale, 1)]

        for i in range(n_res - 1):
            layers.insert(1, self.res_module(in_c=out_c, out_c=out_c, **self.res_params))
        return nn.ModuleList(layers)

    def __makeDecoderStage(self, in_c: int, out_c: int, n_res: int) -> nn.ModuleList:
        # Make one decoder stage
        layers = [self.res_module(in_c=in_c, out_c=out_c, **self.res_params),
                  nn.Conv1d(out_c, out_c, 3, 1, 1)]

        for i in range(n_res - 1):
            layers.insert(1, self.res_module(in_c=out_c, out_c=out_c, **self.res_params))
        return nn.ModuleList(layers)

    def __encoderForward(self, x: torch.Tensor,
                         mix_embed: torch.Tensor) -> List[torch.Tensor]:
        """
        :param x: (B, stem_channels, L)
        :param t_embed: (B, 128)
        :param context: (B, 256, N)
        :return: List of (B, C', L//2**i)
        """

        outputs = []
        for down_stage in self.down_blocks:
            for layer in down_stage[:-1]:
                x = layer(x, mix_embed)  # (B, C, L) -> (B, C, L)
            x = down_stage[-1](x)  # (B, C, L) -> (B, C', L//2)
            outputs.append(x)
        return outputs

    def __decoderForward(self, x: torch.Tensor,
                         mix_embed: torch.Tensor,
                         down_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        :param x: (B, C', L//2**i)
        :param embedding: (B, 256, 1)
        :param down_outputs: List of (B, C', L//2**i)
        :return: (B, C, L)
        """
        for i, up_stage in enumerate(self.up_blocks):
            # fuse with skip connection
            x = torch.cat([x, down_outputs[-i - 1]], dim=1)  # (B, C*2, L//2**i)
            for layer in up_stage[:-1]:
                x = layer(x, mix_embed)
            # upsample
            if x.shape[-1] != down_outputs[-i - 2].shape[-1]:
                x = F.interpolate(x, size=down_outputs[-i - 2].shape[-1], mode='nearest')
            x = up_stage[-1](x)
        return x

    def forward(self,
                x: torch.Tensor,
                diffusion_t: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, 4, L) the input trajectory and binary_mask
        :param diffusion_t: (B) the diffusion step
        :param traj_erase: (B, 3, l) the erased trajectory
        :param insertion_mask: (B, l-1) the insertion mask
        :param extra_embed: (B, E) the vector of embeddings
        """
        B, C, L = x.shape

        lengths = torch.tensor([L], device=x.device).repeat(B)
        mix_embed = (self.time_embedder(diffusion_t) + self.length_embedder(lengths)).unsqueeze(-1)

        x = self.pre_embed(x)
        down_outputs = self.__encoderForward(x, mix_embed)  # List of (B, C', L//2**i)
        down_outputs.insert(0, x)
        x = self.mid_attn_block(down_outputs[-1], mix_embed)  # (B, C', L//2**i)
        x = self.__decoderForward(x, mix_embed, down_outputs)
        if self.down_schedule[0] != 1:
            x = F.interpolate(x, size=L, mode='linear', align_corners=True)
        return self.head(x)  # (B, C, L)


if __name__ == "__main__":
    model = TrajWeaver(
        in_c=12,  # input trajectory encoding channels
        out_c=2,
        down_schedule=[1, 2, 2, 2],  # downsample schedule, first element is step stride
        diffusion_steps=300,  # maximum diffusion steps
        channel_schedule=[64, 128, 128, 256],  # channel schedule of stages, first element is stem output channels
        n_res_blocks=[2, 2, 2],  # number of resblocks in each stage
        embed_c=64,  # channels of time embeddings
        num_heads=4,  # number of heads for attention
        dropout=0.0,  # dropout
        max_length=4096,  # maximum input encoding length
        self_attn=True
    ).cuda()
    # length: 16 ~ 64
    torch.save(model.state_dict(), 'Models/TrajWeaver.pth')