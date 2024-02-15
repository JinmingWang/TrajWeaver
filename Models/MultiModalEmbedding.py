from Models.ModelBasics import *
from math import sqrt


class MapEmbedding(nn.Module):
    def __init__(self, in_c: int, embed_c: int, density_map, direction_map):
        super().__init__()

        self.density_map = density_map.view(1, 1, 1024, 1024)
        self.direction_map = direction_map.view(1, 1, 1024, 1024)

        row_nums = torch.linspace(-1, 1, 1024).view(1, 1, 1024).repeat(1, 1024, 1)
        col_nums = torch.linspace(-1, 1, 1024).view(1, 1024, 1).repeat(1, 1, 1024)
        self.pos_enc = torch.cat([row_nums, col_nums], dim=0).unsqueeze(0).cuda()  # (1, 2, 1024, 1024)

        # (B, 4, 1024, 1024)
        self.map_proj = nn.Sequential(
            nn.Conv2d(4, 16, 3, 2, 1), Swish(),
            nn.Conv2d(16, 32, 3, 2, 1), Swish(),
            nn.Conv2d(32, 64, 3, 2, 1), Swish(),
            nn.Conv2d(64, 128, 3, 2, 1), Swish(),
            nn.Conv2d(128, 256, 3, 2, 1), Swish(),
            # (B, 256, 32, 32)
            nn.Flatten(2),  # (B, 256, 1024)
            nn.Conv1d(256, 512, 1, 1, 0)
        )

        self.traj_proj = nn.Sequential(
            nn.Conv1d(in_c, 64, 3, 1, 1), Swish(),
            nn.Conv1d(64, 256, 3, 1, 1)
        )

        self.out_proj = nn.Conv1d(256, embed_c, 1, 1, 0)

        self.scale = 1 / sqrt(256)

    def forward(self, traj):
        B = traj.shape[0]
        map_feature = torch.cat([self.density_map, self.direction_map, self.pos_enc], dim=1).repeat(B, 1, 1, 1)
        map_feature = self.map_proj(map_feature)  # (B, 512, 1024)

        k, v = map_feature.chunk(2, dim=1)
        k = rearrange(k, 'b c l -> b l c')  # (B, 1024, 256)
        q = self.traj_proj(traj)  # (B, 256, L)

        attn = torch.softmax(torch.bmm(k, q) * self.scale, dim=1)  # (B, 1024, 1024)
        return self.out_proj(torch.bmm(v, attn))  # (B, embed_c, 1024)


class UserEmbedding(nn.Module):
    def __init__(self, in_c: int, embed_c: int, num_users: int):
        super().__init__()

        self.uid_proj = nn.Sequential(
            nn.Embedding(num_users, 512),  # (B, 512)
            nn.Linear(512, 1024),  # (B, 1024)
            Swish(),
            nn.Unflatten(1, (8, 128)),  # (B, 8, 128)
            nn.Conv1d(8, 512, 1, 1, 0)  # (B, 512, 128)
        )

        self.scale = 1 / sqrt(256)

        self.traj_proj = nn.Sequential(
            nn.Conv1d(in_c, 64, 3, 1, 1), Swish(),
            nn.Conv1d(64, 256, 3, 1, 1)
        )

        self.out_proj = nn.Conv1d(256, embed_c, 1, 1, 0)

    def forward(self, traj, uid):
        user_feature = self.uid_proj(uid)  # (B, 512, 128)
        k, v = user_feature.chunk(2, dim=1)     # 2 * (B, 256, 128)
        k = rearrange(k, 'b c l -> b l c')  # (B, 128, 256)
        q = self.traj_proj(traj)  # (B, 256, L)

        attn = torch.softmax(torch.bmm(k, q) * self.scale, dim=1)  # (B, 128, L)

        return self.out_proj(torch.bmm(v, attn))  # (B, embed_c, L)


class MultimodalEmbedding(nn.Module):
    def __init__(self, in_c: int, map_embed_c: int, user_embed_c: int, num_users: int, density_map, direction_map):
        super().__init__()

        self.map_embedder = MapEmbedding(in_c, map_embed_c, density_map, direction_map)

        self.user_embedder = UserEmbedding(in_c, user_embed_c, num_users)


    def forward(self, traj, uid):
        return torch.cat([self.map_embedder(traj), self.user_embedder(traj, uid)], dim=1)  # (B, embed_c, L)