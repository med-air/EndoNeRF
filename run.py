import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, d2vgo
from lib.load_dynamic_data import load_data


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')
    parser.add_argument("--export_canonical", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--recon_pc", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--profiling", action='store_true',
                        help='whether enable profiling mode')
    parser.add_argument("--profiling_duration", type=float, default=3600,
                        help='duration (in seconds) to record profiling')
    parser.add_argument("--profiling_timeintvl", type=float, default=1,
                        help='time interval (in seconds) to record profiling')

    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, render_times, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=0, eval_ROIs=None,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, logging=True):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor != 0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    depths = []
    displs = []
    mses = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    CHUNK_SIZE = 8192 # // 2

    it = tqdm(render_poses) if logging else render_poses
    for i, c2w in enumerate(it):

        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, camera_type=cfg.data.dataset_type)
        ts = torch.ones(CHUNK_SIZE, 1) * render_times[i]
        keys = ['rgb_marched', 'depth', 'displ']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, ts, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(CHUNK_SIZE, 0), rays_d.split(CHUNK_SIZE, 0), viewdirs.split(CHUNK_SIZE, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }

        if eval_ROIs is not None:
            ROI = eval_ROIs[i].to(render_result['rgb_marched'].device)
            render_result['rgb_marched'][ROI < 0.5] = (torch.Tensor(gt_imgs[i]).to(render_result['rgb_marched'].device))[ROI < 0.5]

        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        if 'displ' in render_result:
            displ_map = render_result['displ'].cpu().numpy()
            displs.append(displ_map)

        rgbs.append(rgb)
        depths.append(depth)
        if i==0 and logging:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            mse = np.mean(np.square(rgb - gt_imgs[i]))
            p = -10. * np.log10(mse)
            mses.append(mse)
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs) and logging:
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if len(displs) > 0:
        displs = np.array(displs)
    else:
        displs = None

    if savedir is not None:
        if logging:
            print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            if displs is not None:
                displ_map = displs[i]
                # displ_map = displ_map / np.max(displs)
                displ_map = (displ_map - displs.min()) / (displs.max() - displs.min())
                displ_map = utils.to8b(displ_map)
                filename = os.path.join(savedir, '{:03d}_displ.png'.format(i))
                imageio.imwrite(filename, displ_map)

            # if gt_imgs is not None:
            #     os.makedirs(os.path.join(savedir, 'gt'), exist_ok=True)
            #     filename = os.path.join(savedir, 'gt', '{:03d}.png'.format(i))
            #     imageio.imwrite(filename, utils.to8b(gt_imgs[i]))

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    
    extras = {
        'mses': mses,
        'psnrs': psnrs,
        'ssims': ssims,
        'lpips_vgg': lpips_vgg,
        'lpips_alex': lpips_alex,
        # 'displs': displs
    }

    return rgbs, depths, extras


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = [
            'hwf', 'HW', 'Ks', 'near', 'far', 'bbox',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'times', 'render_times', 'images', 'ROIs', 'depths', 'render_K']
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        # data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
        raise NotImplementedError('Irregular images are not supported yet!')
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
        if data_dict['ROIs'] is not None:
            data_dict['ROIs'] = torch.FloatTensor(data_dict['ROIs'], device='cpu')
        if data_dict['depths'] is not None:
            data_dict['depths'] = torch.FloatTensor(data_dict['depths'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses']).float()
    data_dict['times'] = torch.Tensor(data_dict['times']).float()
    data_dict['render_poses'] = torch.Tensor(data_dict['render_poses']).float()
    data_dict['render_times'] = torch.Tensor(data_dict['render_times']).float()
    if data_dict['bbox'] is not None:
        data_dict['bbox'] = [torch.Tensor(data_dict['bbox'][0]), torch.Tensor(data_dict['bbox'][1])]
    return data_dict


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, camera_type=cfg.data.dataset_type)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
        indexing='ij'
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, times, render_poses, render_K, render_times, images, ROIs, depths = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'times', 'render_poses', 'render_K', 'render_times', 'images', 'ROIs', 'depths'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # prepare saving directories for profiling 
    if args.profiling:
        profiling_dir = os.path.join(cfg.basedir, cfg.expname, 'profiling')
        profiling_val_dir = os.path.join(cfg.basedir, cfg.expname, 'profiling', 'val')
        profiling_demo_dir = os.path.join(cfg.basedir, cfg.expname, 'profiling', 'demo')
        profiling_records = {'val/psnr': [], 'val/mse': []}
        os.makedirs(profiling_dir, exist_ok=True)
        os.makedirs(profiling_val_dir, exist_ok=True)
        os.makedirs(profiling_demo_dir, exist_ok=True)

    # init model and optimizer
    model_kwargs = copy.deepcopy(cfg_model)

    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
        # init model
        if cfg.data.ndc:
            raise NotImplementedError('NDC is not supported')
        else:
            print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
            num_voxels = model_kwargs.pop('num_voxels')
            num_times = model_kwargs['num_times0']
            num_dis_res = model_kwargs['num_dis_res0']
            num_dis_scales = model_kwargs.pop('num_dis_scales')
            if len(cfg_train.pg_scale) and reload_ckpt_path is None:
                num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))
                # num_times = int(num_times / (2**len(cfg_train.pg_scale)))
            model = d2vgo.DynamicDirectVoxGO(
                xyz_min=xyz_min, xyz_max=xyz_max,
                num_voxels=num_voxels, num_times=num_times, num_dis_res=num_dis_res, num_dis_scales=num_dis_scales,
                mask_cache_path=coarse_ckpt_path,
                **model_kwargs)
            if cfg_model.maskout_near_cam_vox:
                model.maskout_near_cam_vox(poses[i_train,:3,3], near)
        model = model.to(device)
        optimizer, lr_names = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        if cfg.data.ndc:
            raise NotImplementedError('NDC is not supported')
        else:
            model_class = d2vgo.DynamicDirectVoxGO
        model = utils.load_model(model_class, reload_ckpt_path).to(device)
        optimizer, lr_names = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
        model, optimizer, start = utils.load_checkpoint(
                model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
        model_kwargs.update(model.get_kwargs())

    # amp grad scalar
    grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'render_depth': True,
        'depth_sampling_std': 5.0
    }
    if args.profiling:
        render_profiling_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': near,
                'far': far,
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': cfg_model.stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            # rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
            raise NotImplementedError('Irregular shape is not supported')
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
            ROIs_tr = ROIs[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device) if ROIs is not None else None
            depths_tr_ori = depths[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device) if depths is not None else None

        if cfg_train.ray_sampler == 'in_maskcache':
            # rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, times_tr, imsz = d2vgo.get_training_rays_in_maskcache_sampling(
            #         rgb_tr_ori=rgb_tr_ori,
            #         train_poses=poses[i_train],
            #         train_times=times[i_train],
            #         HW=HW[i_train], Ks=Ks[i_train],
            #         ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            #         flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
            #         model=model, render_kwargs=render_kwargs)
            raise NotImplementedError('Ray sampler scheme "in_maskcache" is not supported yet.')
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, depths_tr, masks_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, ray_tops = d2vgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                train_times=times[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ROIs=ROIs_tr, depths_tr_ori=depths_tr_ori,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
            # raise NotImplementedError('Ray sampler scheme "flatten" is not supported yet.')
        elif cfg_train.ray_sampler == 'in_bbox':
            # rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, times_tr, imsz, ray_tops = d2vgo.get_training_rays_in_bbox_sampling(
            #         rgb_tr_ori=rgb_tr_ori,
            #         train_poses=poses[i_train],
            #         train_times=times[i_train],
            #         HW=HW[i_train], Ks=Ks[i_train],
            #         ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            #         flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
            #         camera_type=cfg.data.dataset_type,
            #         model=model, rc_device=device, render_kwargs=render_kwargs)
            raise NotImplementedError('Ray sampler scheme "in_bbox" is not supported yet.')
        elif cfg_train.ray_sampler == 'in_roi':
            rgb_tr, depths_tr, rays_o_tr, rays_d_tr, viewdirs_tr, times_tr, imsz, ray_tops = d2vgo.get_training_rays_in_roi_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    train_times=times[i_train],
                    HW=HW[i_train], Ks=Ks[i_train], ROIs=ROIs_tr, depths_tr_ori=depths_tr_ori,
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    camera_type=cfg.data.dataset_type, rc_device=device,
                    divide_rays_by_time=True)
            masks_tr = None
        else:
            # rgb_tr, depths_tr, masks_tr, rays_o_tr, rays_d_tr, viewdirs_tr, times_tr, imsz, ray_tops = d2vgo.get_training_rays(
            #         rgb_tr=rgb_tr_ori,
            #         train_poses=poses[i_train],
            #         train_times=times[i_train],
            #         HW=HW[i_train], Ks=Ks[i_train], ROIs=ROIs_tr, depths_tr_ori=depths_tr_ori,
            #         ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            #         flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, camera_type=cfg.data.dataset_type, rc_device=device)
            raise NotImplementedError('Ray sampler scheme is not supported yet.')
        # index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        # batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, depths_tr, masks_tr, rays_o_tr, rays_d_tr, viewdirs_tr, times_tr, imsz, ray_tops

    rgb_tr, depths_tr, masks_tr, rays_o_tr, rays_d_tr, viewdirs_tr, times_tr, imsz, ray_tops = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            with torch.no_grad():
                model.density[cnt <= 2] = -100
        per_voxel_init()

    # Pre-compute multi-resolutions of displacement field
    upsample_dis_res = (np.round(np.exp(np.linspace(np.log(model_kwargs['num_dis_res0']), np.log(model_kwargs['num_dis_res1']), len(cfg_train.pg_scale) + 1)))).astype(np.int32).tolist()[1:]
    upsample_times = (np.round(np.exp(np.linspace(np.log(model_kwargs['num_times0']), np.log(model_kwargs['num_times1']), len(cfg_train.pg_scale) + 1)))).astype(np.int32).tolist()[1:]

    # Begin training iterations
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1

    if args.profiling:
        profiling_total_time = 0
        pre_profiling_time = 0
        profiling_steps = 0

    for global_step in trange(1+start, 1+cfg_train.N_iters):
        if args.profiling:
            profiling_t0 = time.time()

        # renew occupancy grid
        # model.fast_color_thres = model.fast_color_thres * (100 ** (1 / 20000))
        if global_step > cfg_train.renew_occ_after and (global_step - cfg_train.renew_occ_after) % cfg_train.renew_occ_every == 0 and model.mask_cache is not None: # and (global_step + 500) % 2000 == 0
            self_alpha = F.max_pool3d(model.activate_density(model.density), kernel_size=3, padding=1, stride=1)[0,0]
            model.mask_cache.mask &= (self_alpha > model.fast_color_thres)

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_scales = cfg_train.pg_scale.index(global_step)
            n_rest_scales = len(cfg_train.pg_scale)-n_scales-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            # cur_times = int(cfg_model.num_times / (2**n_rest_scales))

            cur_dis_res = upsample_dis_res[n_scales]
            cur_times = upsample_times[n_scales]

            if isinstance(model, d2vgo.DynamicDirectVoxGO):
                model.scale_volume_grid(cur_voxels, cur_dis_res, cur_times)
            else:
                raise NotImplementedError
            optimizer, lr_names = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.density.data.sub_(1)
        
        # random sample rays
        rays_mask = None
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_b = random.randint(0, len(ray_tops) - 1)
            ray_num = imsz[sel_b]
            ray_i_start = ray_tops[sel_b]
            ray_i_end = ray_i_start + ray_num
            sel_i = torch.randint(ray_i_start, ray_i_end, [cfg_train.N_rand])
            target = rgb_tr[sel_i]
            target_depth = depths_tr[sel_i]
            rays_mask = masks_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
            ts = times_tr[sel_i]

        elif cfg_train.ray_sampler in ['in_bbox', 'in_roi']:
            ray_num = -1
            while ray_num < 0: # ensure there are rays going through the ROI
                sel_b = random.randint(0, len(ray_tops) - 1)
                ray_num = (ray_tops[sel_b + 1] if sel_b < len(ray_tops) - 1 else len(rays_o_tr)) - ray_tops[sel_b]
            ray_i_start = ray_tops[sel_b]
            ray_i_end = ray_i_start + ray_num
            sel_i = torch.randint(ray_i_start, ray_i_end, [cfg_train.N_rand])
            target = rgb_tr[sel_i]
            target_depth = depths_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
            ts = times_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [1])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b][0, sel_r, sel_c, :]
            rays_o = rays_o_tr[sel_b][0, sel_r, sel_c, :]
            rays_d = rays_d_tr[sel_b][0, sel_r, sel_c, :]
            viewdirs = viewdirs_tr[sel_b][0, sel_r, sel_c, :]
            ts  = torch.tile(times_tr[sel_b], [cfg_train.N_rand, 1])
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            target_depth = target_depth.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
            ts = ts.to(device)
            if rays_mask is not None:
                rays_mask = rays_mask.to(device)

        # forward pass
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            # render_result = model(rays_o, rays_d, viewdirs, ts, depth=target_depth, global_step=global_step, **render_kwargs)
            render_result = model(rays_o, rays_d, viewdirs, ts, depth=None, global_step=global_step, **render_kwargs)


        loss = 0
        if rays_mask is None:
            rgb_loss = F.mse_loss(render_result['rgb_marched'], target)
        else:
            rgb_loss = F.mse_loss(render_result['rgb_marched'] * rays_mask, target * rays_mask)
        loss += cfg_train.weight_main * rgb_loss
        if cfg_train.weight_depth > 0:
            if rays_mask is None:
                loss += cfg_train.weight_depth * F.huber_loss(render_result['depth'], target_depth)
            else:
                loss += cfg_train.weight_depth * F.huber_loss(render_result['depth'] * rays_mask, target_depth * rays_mask)
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss

        grad_scaler.scale(loss).backward()

        grad_scaler.unscale_(optimizer)

        if global_step < cfg_train.tv_before and global_step > cfg_train.tv_after and global_step % cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)

        # optimizer.step()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1 / decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            if not 'dis_' in lr_names[i_opt_g]:
                param_group['lr'] = param_group['lr'] * decay_factor
            else:
                # update lr for displacment fields
                param_state = optimizer.state[param_group['params'][0]]
                if not 'step' in param_state:
                    continue
                else:
                    opt_step = param_state['step']
                    if opt_step in [1000, 2000]:
                        param_group['lr'] = param_group['lr'] * 0.33

        
        #####################################
        # depth refinement
        #####################################
        refinement_round = global_step // cfg_train.depth_refine_period
        if not cfg_train.no_depth_refine and depths is not None and global_step % cfg_train.depth_refine_period == 0 and refinement_round <= cfg_train.depth_refine_rounds:
            print('Render RGB and depth maps for refinement...')

            render_depth_kwargs = {
                'model': model,
                'ndc': cfg.data.ndc,
                'render_kwargs': {
                    'near': near,
                    'far': far,
                    'bg': 1 if cfg.data.white_bkgd else 0,
                    'stepsize': cfg_model.stepsize,
                    'inverse_y': cfg.data.inverse_y,
                    'flip_x': cfg.data.flip_x,
                    'flip_y': cfg.data.flip_y,
                    'render_depth': True,
                },
            }
            
            refinement_save_path = os.path.join(cfg.basedir, cfg.expname, 'refinement{:04d}'.format(refinement_round))
            if not os.path.exists(refinement_save_path):
                os.makedirs(refinement_save_path)
            depth_prev_save_path = os.path.join(refinement_save_path, 'depth_prev')
            depth_refined_save_path = os.path.join(refinement_save_path, 'depth_refined')
            if not os.path.exists(depth_prev_save_path):
                os.makedirs(depth_prev_save_path)
            if not os.path.exists(depth_refined_save_path):
                os.makedirs(depth_refined_save_path)

            with torch.no_grad():
                downsample_factor = 4
                rgbs_t, depths_t, _ = render_viewpoints(
                        render_poses=poses[i_train],
                        render_times=times[i_train],
                        HW=HW[i_train],
                        Ks=Ks[i_train],
                        render_factor=downsample_factor,
                        # gt_imgs=[images[i_train].cpu().numpy()],
                        savedir=None, eval_ROIs=None,
                        logging=False, **render_depth_kwargs)

                rgbs_t = torch.tensor(rgbs_t).to(depths_tr.device)
                depths_t = torch.tensor(depths_t).to(depths_tr.device)

                rgbs_t = F.interpolate(rgbs_t.permute(0, 3, 1, 2), scale_factor=downsample_factor, mode='bilinear')
                rgbs_t = rgbs_t.permute(0, 2 ,3, 1)

                depths_t = F.interpolate(depths_t.permute(0, 3, 1, 2), scale_factor=downsample_factor, mode='bilinear')
                depths_t = depths_t.permute(0, 2 ,3, 1)
                
                ROIs_tr = ROIs[i_train].to(depths_tr.device) if ROIs is not None else None

                quantile_ls = []
                depth_diff_ls = []
                depth_to_refine_ls = []

                top = 0
                for ROI, depth_map in zip(ROIs_tr, depths_t):
                    mask = (ROI > 0.5).bool()
                    n = mask.sum().cpu().item()

                    depth_tr_ = depths_tr[top:top+n]
                    depth_t_ = depth_map[mask].squeeze(-1)
                    depth_diff = torch.pow(depth_t_ - depth_tr_, 2)
                    quantile = torch.quantile(depth_diff, 1.0 - cfg_train.depth_refine_quantile, dim=0, keepdim=True)
                    depth_to_refine = (depth_diff > quantile)
                    depth_tr_[depth_to_refine] = depth_t_[depth_to_refine]
                    
                    depths_tr[top:top+n].copy_(depth_tr_)

                    top = top + n

                    quantile_ls.append(quantile.cpu().numpy())
                    depth_diff_ls.append(depth_diff.cpu().numpy())
                    depth_to_refine_ls.append(depth_to_refine.cpu().numpy())

                save_dict = {
                    'rounds': refinement_round,
                    'quantile': quantile_ls,
                    'depth_diff': depth_diff_ls,
                    'depth_to_refine': depth_to_refine_ls
                }
                torch.save(save_dict, os.path.join(refinement_save_path, 'depth_refine_info.tar'))

                del rgbs_t, depths_t, depth_to_refine, depth_diff, quantile, quantile_ls, depth_diff_ls, depth_to_refine_ls, ROIs_tr

                print('\nRefinement finished, intermediate results saved at', refinement_save_path)


        #####################################
        # record profiling
        #####################################
        if args.profiling:
            dt = time.time() - profiling_t0
            profiling_total_time += dt

            if profiling_total_time <= args.profiling_duration and (profiling_total_time - pre_profiling_time) >= args.profiling_timeintvl:
                pre_profiling_time = profiling_total_time
                profiling_steps += 1
                torch.cuda.empty_cache()

                val_img_i = [profiling_steps % i_val.shape[0]]
                demo_img_i = [profiling_steps % render_poses.shape[0]]
                
                with torch.no_grad():                    
                    rgb_val, _, evals = render_viewpoints(
                            render_poses=poses[i_val[val_img_i]],
                            render_times=times[i_val[val_img_i]],
                            HW=HW[i_val[val_img_i]],
                            Ks=Ks[i_val[val_img_i]],
                            gt_imgs=[images[i_val[val_img_i]].cpu().numpy()],
                            savedir=None, eval_ROIs=None,
                            logging=False, **render_profiling_kwargs)
                    
                    rgb_demo, _, _ = render_viewpoints(
                            render_poses=render_poses[demo_img_i],
                            render_times=render_times[demo_img_i],
                            HW=HW[i_test][[0]].repeat(1, 0),
                            Ks=(Ks[i_test][[0]] if render_K is None else render_K).repeat(1, 0),
                            render_factor=args.render_video_factor,
                            gt_imgs=None, savedir=None, eval_ROIs=None, logging=False, **render_profiling_kwargs)

                imageio.imwrite(os.path.join(profiling_val_dir, 'val_t{:06d}.png'.format(int(profiling_total_time))), utils.to8b(rgb_val[0]))
                imageio.imwrite(os.path.join(profiling_demo_dir, 'demo_t{:06d}.png'.format(int(profiling_total_time))), utils.to8b(rgb_demo[0]))
                profiling_records['val/mse'].append((profiling_total_time, evals['mses'][0].item()))
                profiling_records['val/psnr'].append((profiling_total_time, evals['psnrs'][0].item()))
            
                np.save(os.path.join(profiling_dir, 'profiling_records.npy'), profiling_records)

        # calculate PSNR
        psnr = utils.mse2psnr(rgb_loss.detach())
        psnr_lst.append(psnr.item())
        
        # check log & save
        if global_step % args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            psnr_lst = []

        if global_step % args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)
        
        torch.cuda.empty_cache()

    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)


def train(args, cfg, data_dict):
    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching
    eps_coarse = time.time()
    if data_dict['bbox'] is None:
        xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    else:
        xyz_min_coarse, xyz_max_coarse = data_dict['bbox']
    if cfg.coarse_train.N_iters > 0:
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, stage='coarse')
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.data.ndc:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        if coarse_ckpt_path is not None:
            xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                    model_class=d2vgo.DynamicDirectVoxGO, model_path=coarse_ckpt_path,
                    thres=cfg.fine_model_and_render.bbox_thres)
        else:
            xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()

    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=None) # coarse_ckpt_path
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    training_time_logs = [f'fine detail reconstruction in {eps_time_str}']

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')

    training_time_logs.append(f'training procedure in {eps_time_str}')

    log_savepath = os.path.join(cfg.basedir, cfg.expname, f'training_time.txt')
    with open(log_savepath, 'w+') as f:
        f.write('\n'.join(training_time_logs))


if __name__=='__main__':
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        if data_dict['bbox'] is None:
            xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        else:
            xyz_min, xyz_max = data_dict['bbox']
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y, camera_type=cfg.data.dataset_type)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            if data_dict['bbox'] is None:
                xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
            else:
                xyz_min, xyz_max = data_dict['bbox']
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(d2vgo.DynamicDirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0).squeeze().permute(1,2,3,0).cpu().numpy()
            xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                    model_class=d2vgo.DynamicDirectVoxGO, model_path=ckpt_path,
                    thres=cfg.fine_model_and_render.bbox_thres)
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            xyz_min_fine=xyz_min_fine.cpu().numpy(), xyz_max_fine=xyz_max_fine.cpu().numpy())
        print('done')
        sys.exit()

    # train
    if not args.render_only:
        train(args, cfg, data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video or args.export_canonical or args.recon_pc:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            raise NotImplementedError('NDC is not supported')
        else:
            model_class = d2vgo.DynamicDirectVoxGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        model.eval()
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
                'render_displ': False
            },
        }

    # export canonical frame
    if args.export_canonical:
        print('Export canonical visualization...')
        with torch.no_grad():
            alpha = model.activate_density(model.density).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_canonical, alpha=alpha, rgb=rgb)
        print('done')

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths, _ = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                render_times=data_dict['times'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir,
                eval_ROIs=data_dict['ROIs'][data_dict['i_train']] if data_dict['ROIs'] is not None else None,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, depths, extras = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                render_times=data_dict['times'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir,
                eval_ROIs=data_dict['ROIs'][data_dict['i_test']] if data_dict['ROIs'] is not None else None,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

        with open(os.path.join(testsavedir, 'metrics.json')) as f:
            json.dump(extras, f, indent=2)

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        # render_times = data_dict['times'][data_dict['i_train']][74].unsqueeze(0).repeat(len(data_dict['render_poses']), 1)
        rgbs, depths, extras = render_viewpoints(
                render_poses=data_dict['render_poses'],
                render_times=data_dict['render_times'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=(data_dict['Ks'][data_dict['i_test']][[0]] if data_dict['render_K'] is None else data_dict['render_K']).repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                savedir=testsavedir,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

        if 'displs' in extras:
            displs = extras['displs']
            displs = (displs - displs.min()) / (displs.max() - displs.min())
            imageio.mimwrite(os.path.join(testsavedir, 'video.dis.mp4'), utils.to8b(displs), fps=30, quality=8)

    # reconstruct point clouds
    if args.recon_pc:
        import open3d as o3d
        import cv2

        savedir = os.path.join(cfg.basedir, cfg.expname, f'recon_{ckpt_name}')
        os.makedirs(savedir, exist_ok=True)

        recon_idx = np.arange(len(data_dict['render_poses']))[cfg.reconstruction.start_frame:cfg.reconstruction.end_frame + 1 if cfg.reconstruction.end_frame >= 0 else -1]
        recon_poses = data_dict['render_poses'][recon_idx]
        recon_times = data_dict['render_times'][recon_idx]

        rgbs, depths, _ = render_viewpoints(
                render_poses=recon_poses,
                render_times=recon_times,
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(recon_poses), 0),
                Ks=(data_dict['Ks'][data_dict['i_test']][[0]] if data_dict['render_K'] is None else data_dict['render_K']).repeat(len(recon_poses), 0),
                savedir=None,
                **render_viewpoints_kwargs)

        depths = depths / np.max(depths)

        for i in range(len(rgbs)):
            rgb_np = rgbs[i]
            depth_np = np.ascontiguousarray(depths[i].squeeze(-1), dtype=np.float32)

            rgb_np = rgb_np[:, cfg.reconstruction.left_crop:-(cfg.reconstruction.right_crop + 1), :]
            depth_np = depth_np[:, cfg.reconstruction.left_crop:-(cfg.reconstruction.right_crop + 1)]

            depth_np = cv2.bilateralFilter(depth_np, cfg.reconstruction.depth_smooth_d, cfg.reconstruction.depth_smooth_sv, cfg.reconstruction.depth_smooth_sr)

            rgb_im = o3d.geometry.Image(utils.to8b(rgb_np))
            depth_im = o3d.geometry.Image(utils.to8b(depth_np))

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(int(data_dict['hwf'][1]), int(data_dict['hwf'][0]), data_dict['hwf'][2], data_dict['hwf'][2], data_dict['hwf'][1] / 2, data_dict['hwf'][0] / 2)
            )

            o3d.io.write_point_cloud(os.path.join(savedir, f"frame_{recon_idx[i]:06d}_pc.ply"), pcd)

