from Models.TrajWeaver import TrajWeaver
from Utils import loadModel
from DiffusionManager import DiffusionManager
from eval import eval

if __name__ == '__main__':
    diff_manager = DiffusionManager(
        min_beta=0.0001,
        max_beta=0.05,
        max_diffusion_step=500,
        scale_mode="quadratic"
    )

    model = TrajWeaver(
        in_c=6,  # input trajectory encoding channels
        out_c=2,
        down_schedule=[1, 2, 2, 2],  # downsample schedule, first element is step stride
        diffusion_steps=500,  # maximum diffusion steps
        channel_schedule=[64, 128, 128, 256],  # channel schedule of stages, first element is stem output channels
        n_res_blocks=[2, 2, 2],  # number of resblocks in each stage
        embed_c=64,  # channels of time embeddings
        num_heads=4,  # number of heads for attention
        dropout=0.0,  # dropout
        max_length=4096,  # maximum input encoding length
        self_attn=True
    ).cuda()

    # loadModel(model, "")
    # model = model.eval()
    # eval(diff_manager, model, None)