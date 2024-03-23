_base_ = '../default.py'

expname = None
basedir = './logs/endo'

data = dict(
    datadir=None,
    dataset_type='llff',
    ndc=False,
    width=640,
    height=512,
    factor=1,
    skip_frames=[],
    llffhold=8,
    llff_renderpath='fixidentity'
)

coarse_train = dict(
    N_iters=0,
)

fine_train = dict(
    N_iters=20000,
    N_rand=4096,
    # pg_scale=[2000,4000,6000,8000],
    pg_scale=[],
    ray_sampler='in_roi',
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_tv_k0=1e-6,
    weight_depth=0.01,

    no_depth_refine=False,
    depth_refine_period=1000,
    depth_refine_rounds=2,
    depth_refine_quantile=0.02,

    renew_occ_after=2000,
    renew_occ_every=500,

    lrate_dnet_layers = 1e-3
)

fine_model_and_render = dict(
    num_voxels=256**3,
    rgbnet_dim=9,
    rgbnet_width=64,
    world_bound_scale=1,
    fast_color_thres=1e-3,
    alpha_init=1e-2,

    num_dis_res0=16,
    num_dis_res1=32,
    num_times0=16,
    num_times1=32,
    num_dis_scales=1,
    dis_comp_n=24,
    dis_feat_dim=12,
    dis_net_width=128,
    dis_net_depth=3,
)

reconstruction = dict(
    start_frame=0,
    end_frame=-1,
    left_crop=70,
    right_crop=0,
    depth_smooth_d=28,
    depth_smooth_sv=64,
    depth_smooth_sr=32
)