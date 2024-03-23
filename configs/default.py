from copy import deepcopy

expname = None                    # experiment name
basedir = './logs/'               # where to store ckpts and logs
use_amp = True                 # whether enable amp

''' Template of data options
'''
data = dict(
    datadir=None,                 # path to dataset root folder
    dataset_type=None,            # blender | nsvf | blendedmvs | tankstemple | deepvoxels | co3d
    inverse_y=False,              # intrinsict mode (to support blendedmvs, nsvf, tankstemple)
    flip_x=False,                 # to support co3d
    flip_y=False,                 # to support co3d
    annot_path='',                # to support co3d
    split_path='',                # to support co3d
    sequence_name='',             # to support co3d
    load2gpu_on_the_fly=False,    # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    white_bkgd=False,             # use white background (note that some dataset don't provide alpha and with blended bg color)
    half_res=False,               # [TODO]

    # Below are forward-facing llff specific settings. Not support yet.
    ndc=False,                    # use ndc coordinate (only for forward-facing)
    spherify=False,               # inward-facing
    factor=4,                     # [TODO]
    width=None,                   # enforce image width
    height=None,                  # enforce image height
    llffhold=8,                   # testsplit
    load_depths=False,            # load depth
    llff_renderpath='spiral',
    skip_frames=[],

    # Below are nhr specific settings.
    start_frame=0,
    num_frames=60
)

''' Template of training options
'''
coarse_train = dict(
    N_iters=5000,                 # number of optimization steps
    N_rand=8192,                  # batch size (number of random rays per optimization step)
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    lrate_dis_lines=2e-2,         # lr of displacement components
    # lrate_dis_line=2e-2,        # lr of displacement components
    lrate_dis_cubes=2e-2,         # lr of displacement components
    lrate_dis_basis_mat=1e-3,     # lr of displacement basis
    lrate_dis_net=1e-3,           # lr of displacement net
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    pervoxel_lr=True,             # view-count-based lr
    pervoxel_lr_downrate=1,       # downsampled image for computing view-count-based lr
    ray_sampler='random',         # ray sampling strategies
    weight_main=1.0,              # weight of photometric loss
    weight_depth=1.0,             # weight of depth loss
    weight_entropy_last=0.01,     # weight of background entropy loss
    weight_rgbper=0.1,            # weight of per-point rgb loss
    weight_dis_l1_reg=0.0,        # weight of L1 penalty of displacement fields
    tv_every=1,                   # count total variation loss every tv_every step
    tv_after=0,                   # count total variation loss from tv_from step
    tv_before=0,                  # count total variation before the given number of iterations
    tv_dense_before=0,            # count total variation densely before the given number of iterations
    weight_tv_density=0.0,        # weight of total variation loss of density voxel grid
    weight_tv_k0=0.0,             # weight of total variation loss of color/feature voxel grid
    weight_tv_dis=0.0,            # weight of total variation loss of displacement grid
    pg_scale=[],                  # checkpoints for progressive grid scaling
    renew_occ_after=2000,         # renew occupancy mask after renew_occ_after steps
    renew_occ_every=1500,         # renew occupancy mask every renew_occ_every steps
    skip_zero_grad_fields=[],     # the variable name to skip optimizing parameters w/ zero grad in each iteration
)

fine_train = deepcopy(coarse_train)
fine_train.update(dict(
    N_iters=20000,
    pervoxel_lr=False,
    ray_sampler='in_maskcache',
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    pg_scale=[1000, 2000, 3000, 4000],
    renew_occ_after=10000,
    renew_occ_every=2000,
    skip_zero_grad_fields=['density', 'k0'],
))

''' Template of model and rendering options
'''
coarse_model_and_render = dict(
    num_voxels=1024000,           # expected number of voxel
    num_voxels_base=1024000,      # to rescale delta distance
    num_times0=32,                # the initial number of frames
    num_times1=64,                # the final number of frames
    num_dis_res0=128,             # the initial size of displacement fields
    num_dis_res1=256,             # the final size of displacement fields
    num_dis_scales=1,             # the scales of displacement fields
    mpi_depth=128,                # the number of planes in Multiplane Image (work when ndc=True)
    nearest=False,                # nearest interpolation
    pre_act_density=False,        # pre-activated trilinear interpolation
    in_act_density=False,         # in-activated trilinear interpolation
    bbox_thres=1e-3,              # threshold to determine known free-space in the fine stage
    mask_cache_thres=1e-3,        # threshold to determine a tighten BBox in the fine stage
    rgbnet_dim=0,                 # feature voxel grid dim
    rgbnet_full_implicit=False,   # let the colors MLP ignore feature voxel grid
    rgbnet_direct=True,           # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=128,             # width of the colors MLP
    alpha_init=1e-6,              # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-7,        # threshold of alpha value to skip the fine stage sampled point
    maskout_near_cam_vox=True,    # maskout grid points that between cameras and their near planes
    world_bound_scale=1,          # rescale the BBox enclosing the scene
    stepsize=0.5,                 # sampling stepsize in volume rendering
    dis_comp_n=96,                # rank of displacement
    dis_feat_dim=27,              # displacement feature dimension
    dis_net_width=128,            # displacement net width
    dis_net_depth=3               # displacement net depth
)

fine_model_and_render = deepcopy(coarse_model_and_render)
fine_model_and_render.update(dict(
    num_voxels=160**3,
    num_voxels_base=160**3,
    rgbnet_dim=12,
    alpha_init=1e-2,
    fast_color_thres=1e-4,
    maskout_near_cam_vox=False,
    world_bound_scale=1.05,
))

del deepcopy
