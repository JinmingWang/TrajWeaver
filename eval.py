from EvalUtils import *
from Models.TrajWeaver import TrajWeaver
from Models.MultiModalEmbedding import MultimodalEmbedding
from functools import partial


def getInferFunction(model: nn.Module,
                     traj_0: torch.Tensor,
                     mask: torch.Tensor,
                     interp_mask: torch.Tensor,
                     global_embed: torch.Tensor,
                     traj_guess: torch.Tensor
                     ):
    """
    This function returns a function that can be used to infer the noise of a trajectory
    :param model: The diffusion model
    :param traj_0: The original trajectory (B, 3, L)
    :param mask: The mask of the original trajectory (B, 3, L)
    :param interp_mask: The mask of the interpolated trajectory (B, 1, L)
    :return:
    """

    B = traj_0.shape[0]

    lnglat_guess = traj_guess[:, :2, :]

    def infer(interp_lnglat_t: torch.Tensor, t: torch.Tensor):
        # interp_lnglat_t: (B, 2, n_interp)
        # t: (B,)
        traj_t = traj_0.clone()  # (B, 3, L)
        # print(traj_t[mask].shape, interp_lnglat_t.shape)
        traj_t[mask] = interp_lnglat_t.reshape(-1)  # fill in the generated / interpolated points
        if global_embed is not None:
            traj_input = torch.cat([traj_t, lnglat_guess, interp_mask, global_embed], dim=1)  # construct the input
        else:
            traj_input = torch.cat([traj_t, lnglat_guess, interp_mask], dim=1)
        output = model(traj_input, t)
        epsilon_pred = output[mask[:, :2]].view(B, 2, -1)  # extract the predicted part / interpolated part
        return epsilon_pred

    return infer


def getDiffusionManager(n_steps=400, min_beta=0.0001, max_beta=0.04, scale_mode="quadratic"):
    """
    Get a DiffusionManager
    :param is_IPD: whether to use interpolation prior diffusion
    :return:
    """
    diffusion_args = {
        "min_beta": min_beta,
        "max_beta": max_beta,
        "max_diffusion_step": n_steps,
        "scale_mode": scale_mode,
    }

    return DiffusionManager(**diffusion_args)


def getModel(ckpt_model: str, ckpt_embedder: str, dataset: "Dataset" = None):
    model = TrajWeaver(
        in_c=12,  # input trajectory encoding channels
        out_c=2,
        down_schedule=[1, 2, 2, 2, 2],  # downsample schedule, first element is step stride
        diffusion_steps=400,  # maximum diffusion steps
        channel_schedule=[64, 64, 128, 256, 256],  # channel schedule of stages, first element is stem output channels
        n_res_blocks=[2, 2, 2, 2],  # number of resblocks in each stage
        embed_c=64,  # channels of time embeddings
        num_heads=4,  # number of heads for attention
        dropout=0,  # dropout
        max_length=4096,  # maximum input encoding length
        self_attn=True
    ).cuda()
    loadModel(model, ckpt_model)
    model = model.eval()

    if dataset is None:
        density_map = torch.zeros(1024, 1024).cuda()
        direction_map = torch.zeros(1024, 1024).cuda()
        n_users = 100
    else:
        density_map = dataset.density_map
        direction_map = dataset.direction_map
        n_users = dataset.nUsers

    global_embedder = MultimodalEmbedding(6, 4, 2, n_users, density_map, direction_map).cuda()
    loadModel(global_embedder, ckpt_embedder)
    global_embedder = global_embedder.eval()

    return model, global_embedder


def makeInterpNoise(traj: torch.Tensor, erase_mask: torch.Tensor):
    """
    :param traj: (B, 3, L) lng, lat, time
    :param erase_mask: (B, 1, L) 1 for erased, 0 for not erased
    :param t: (B)

    usual noise is just a random normal distribution
    but for interpolation, we do have a rough estimation of where the target point is
    for example, to insert a point pi to p1 and p2, with time ti, t1, t2
    the location of pi is likely to be around the linear interpolation of p1 and p2
    denote the linear interpolation of p1 and p2 at time ti as pi'
    then our noise can be a scaled normal distribution with mean at pi' and some feasible std
    """

    B, C, L = traj.shape

    mask = erase_mask.repeat(1, 3, 1).to(torch.bool)
    erased_subtraj = traj[mask].view(B, 3, -1)  # (B, 3, L_erased)
    remain_subtraj = traj[~mask].view(B, 3, -1)  # (B, 3, L_remain)

    L_remain = remain_subtraj.shape[-1]
    L_erased = erased_subtraj.shape[-1]

    time_interp = erased_subtraj[:, 2, :]  # (B, L_erased)
    # time_interp[time_interp < time_interp[:, 0:1]] += 1
    time_remain = remain_subtraj[:, 2, :]  # (B, L_remain)
    # time_remain[time_remain < time_remain[:, 0:1]] += 1
    ids_right = torch.stack([torch.searchsorted(time_remain[i], time_interp[i]) for i in range(B)]).to(torch.long).view(
        -1)  # (B * L_erased)
    ids_left = ids_right - 1  # (B * L_erased)

    ids_left = torch.clamp(ids_left, 0, L_remain - 1)
    ids_right = torch.clamp(ids_right, 0, L_remain - 1)

    ids_batch = torch.arange(B).view(B, 1).repeat(1, L_erased).view(-1)  # (B * L_erased

    lng_left = remain_subtraj[ids_batch, torch.zeros_like(ids_batch), ids_left]  # (B * L_erased)
    lng_right = remain_subtraj[ids_batch, torch.zeros_like(ids_batch), ids_right]  # (B * L_erased)
    lat_left = remain_subtraj[ids_batch, torch.ones_like(ids_batch), ids_left]  # (B * L_erased)
    lat_right = remain_subtraj[ids_batch, torch.ones_like(ids_batch), ids_right]  # (B * L_erased)
    time_left = remain_subtraj[ids_batch, torch.ones_like(ids_batch) * 2, ids_left]  # (B * L_erased)
    time_right = remain_subtraj[ids_batch, torch.ones_like(ids_batch) * 2, ids_right]  # (B * L_erased)

    ratio = (time_interp.reshape(-1) - time_left) / (time_right - time_left)  # (B * L_erased)
    ratio[ratio == float("inf")] = 0
    ratio[ratio == float("-inf")] = 0
    lng_interp = lng_left * (1 - ratio) + lng_right * ratio  # (B * L_erased)
    lat_interp = lat_left * (1 - ratio) + lat_right * ratio  # (B * L_erased)

    lnglat_guess = torch.stack([lng_interp, lat_interp], dim=1).view(B, L_erased, 2).transpose(1, 2)  # (B, 2, L_erased)

    noise = torch.randn_like(lnglat_guess)
    mask[:, 2] = False

    traj_guess = traj.clone()
    traj_guess[mask] = lnglat_guess.reshape(-1)
    return traj_guess, mask, noise


def eval(diff_manager, model, embedder):
    n_trials = 512
    bs = 512
    num_batch = n_trials // bs

    loss_func = nn.MSELoss(reduction="none")
    table = {
        "MSE": "",
        "NDTW": "",
        "JSD": ""
    }

    for erase_rate in [0.3, 0.5, 0.7, 0.9]:
        losses = torch.zeros(n_trials)
        ndtws = torch.zeros(n_trials)
        jsds = torch.zeros(n_trials)

        pbar = tqdm(range(num_batch), desc=f"Evaluating Linear Interpolation For r={erase_rate}")

        getter = XianDataGetter(batch_size=bs, traj_length=512, erase_rate=erase_rate)

        for i in pbar:
            pbar.set_postfix_str("Getting Data")

            traj, interp_mask, uid = next(getter)
            traj_guess, mask, noise = makeInterpNoise(traj, interp_mask)
            traj_input = torch.cat([traj, traj_guess[:, :2, :], interp_mask], dim=1)

            # Compute MSE between the recovered part and the original part
            pbar.set_postfix_str("Computing MSE")

            if embedder is not None:
                with torch.no_grad():
                    global_embed = embedder(traj_input, uid)
            else:
                global_embed = None

            pred_func = getInferFunction(model, traj_guess, mask, interp_mask, global_embed, traj_guess)
            lnglat_rec = diff_manager.diffusionBackward(noise, pred_func)

            traj_recover = traj.clone()
            traj_recover[mask] = lnglat_rec.reshape(-1)

            losses[i * bs: (i + 1) * bs] = loss_func(traj_recover[mask], traj[mask]).view(bs, -1).mean(dim=1)  # compare only the recovered part

            # Compute NDTW between the whole recovered trajectory and the whole original trajectory
            pbar.set_postfix_str(f"Computing NDTW & JSD")
            for b in range(bs):
                ndtws[i * bs + b] = NDTW(traj[b, :, :], traj_recover[b, :, :])
                jsds[i * bs + b] = JSD(getPointDistribution(traj[b:b + 1]), getPointDistribution(traj_recover[b:b + 1]))

        # Latex: $1.81e^{-4}\pm3.3e^{-5}$
        table["MSE"] += f"${losses.mean()*1000:.3f}\\pm{losses.std()*1000:.3f}$ & "
        table["NDTW"] += f"${ndtws.mean()*1000:.3f}\\pm{ndtws.std()*1000:.3f}$ & "
        table["JSD"] += f"${jsds.mean()*1000:.3f}\\pm{jsds.std()*1000:.3f}$ & "

    for k, v in table.items():
        print(k)
        print(v)
        print()

