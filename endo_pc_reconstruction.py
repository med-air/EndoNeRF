from run_endonerf import config_parser, create_nerf
import torch
# from load_blender import pose_spherical
from run_endonerf import render_path
from run_endonerf_helpers import to8b
import numpy as np
import matplotlib.pyplot as plt
# import mcubes
# import trimesh
import os
import configargparse
import open3d as o3d
import cv2


'''
Setup
'''

# set cuda
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
PointCloud reconstruction
'''

###################################################################################################
# Usage Example
###################################################################################################

# python endo_pc_reconstruction.py --config_file configs/example.txt --n_frames 120

###################################################################################################


def generate_rgbd(time, nerf_args, render_poses=None):

    # render_poses = torch.unsqueeze(torch.tensor([[0,-1.0,0,0],[-1.0,0,0.0,0],[0,0,-1.0,0],[0,0,0.0,1.0]]), 0).to(device)

    if render_poses is None:
        render_poses = torch.unsqueeze(torch.tensor([[1.0,0,0,0],[0,1.0,0.0,0],[0,0,1.0,0],[0,0,0.0,1.0]]), 0).to(device)
    render_times = torch.Tensor([time]).to(device)

    with torch.no_grad():
        rgbs, disp = render_path(render_poses, render_times, hwf, nerf_args.chunk, render_kwargs_test, render_factor=nerf_args.render_factor)
    rgbs = to8b(rgbs)
        
    return rgbs[0], disp[0]


def reconstruct_pointcloud(test_time, nerf_args, vis_rgbd=False, depth_filter=None, verbose=True, crop_left_size=0):
    rgb_np, disp_np = generate_rgbd(test_time, nerf_args)
    depth_np = 1.0 / (disp_np + 1e-6)
    
    if crop_left_size > 0:
        rgb_np = rgb_np[:, crop_left_size:, :]
        depth_np = depth_np[:, crop_left_size:]

    if depth_filter is not None:
        depth_np = cv2.bilateralFilter(depth_np, depth_filter[0], depth_filter[1], depth_filter[2])

    if verbose:
        print('min disp:', disp_np.min(), 'max disp:', disp_np.max())
        print('min depth:', depth_np.min(), 'max depth:', depth_np.max())

    rgb_im = o3d.geometry.Image(rgb_np.astype(np.uint8))
    depth_im = o3d.geometry.Image(depth_np)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)

    if vis_rgbd:
        plt.subplot(1, 2, 1)
        plt.title('RGB image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth image')
        plt.imshow(rgbd_image.depth)
        plt.colorbar()
        plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(hwf[1],hwf[0], hwf[2], hwf[2], hwf[1] / 2, hwf[0] / 2)
    )

    return pcd



if __name__ == '__main__':
    cfg_parser = configargparse.ArgumentParser()
    cfg_parser.add_argument('--config_file', type=str, 
                        help='config file path')
    cfg_parser.add_argument('--reload_ckpt', type=str, default='',
                        help='model ckpt to reload')
    cfg_parser.add_argument("--no_pc_saved", action='store_true',
                        help='donot save reconstructed point clouds?')
    cfg_parser.add_argument('--out_postfix', type=str, default='',
                        help='the postfix append to the output directory name')
    cfg_parser.add_argument("--vis_rgbd", action='store_true', 
                        help='visualize RGBD output from NeRF?')
    cfg_parser.add_argument("--start_t", type=float, default=0.0,
                        help='time of start frame')
    cfg_parser.add_argument("--end_t", type=float, default=1.0,
                        help='time of end frame')
    cfg_parser.add_argument("--n_frames", type=int, default=1,
                        help='num of frames')
    cfg_parser.add_argument("--depth_smoother", action='store_true',
                        help='apply bilateral filtering on depth maps?')
    cfg_parser.add_argument("--depth_smoother_d", type=int, default=32,
                        help='diameter of bilateral filter for depth maps')
    cfg_parser.add_argument("--depth_smoother_sv", type=float, default=64,
                        help='The greater the value, the depth farther to each other will start to get mixed')
    cfg_parser.add_argument("--depth_smoother_sr", type=float, default=32,
                        help='The greater its value, the more further pixels will mix together')
    cfg_parser.add_argument("--crop_left_size", type=int, default=75,
                        help='the size of pixels to crop')

    cfg = cfg_parser.parse_args()
    
    nerf_parser = config_parser()
    nerf_args = nerf_parser.parse_args(f'--config {cfg.config_file}')

    if cfg.reload_ckpt:
        setattr(nerf_args, 'ft_path', os.path.join(nerf_args.basedir, nerf_args.expname, cfg.reload_ckpt))

    # set render params for DaVinci endoscopic
    hwf = [512, 640, 569.46820041]
    _, render_kwargs_test, epoch, _, _, _ = create_nerf(nerf_args)
    render_kwargs_test.update({'near' : 0., 'far' : 1.})

    # output directory
    if not cfg.no_pc_saved:
        out_dir = os.path.join(nerf_args.basedir, nerf_args.expname, f"reconstructed_pcds_{epoch}" + (f"_{cfg.out_postfix}" if cfg.out_postfix else ""))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cfg_parser.write_config_file(cfg, [os.path.join(out_dir, 'args.txt')])

    # build depth filter
    if cfg.depth_smoother:
        depth_smoother = (cfg.depth_smoother_d, cfg.depth_smoother_sv, cfg.depth_smoother_sr)
    else:
        depth_smoother = None

    # reconstruct pointclouds
    print('Reconstructing point clouds...')

    pcds = []
    if cfg.n_frames == 1:
        print('>>> t=', cfg.start_t)
        pcd = reconstruct_pointcloud(cfg.start_t, nerf_args, cfg.vis_rgbd, depth_filter=depth_smoother, crop_left_size=cfg.crop_left_size)
        pcds.append(pcd)
    else:
        for test_time in np.linspace(cfg.start_t, cfg.end_t, cfg.n_frames):
            print('>>> t=', test_time)
            pcd = reconstruct_pointcloud(test_time, nerf_args, cfg.vis_rgbd, depth_filter=depth_smoother, crop_left_size=cfg.crop_left_size)
            pcds.append(pcd)

    if not cfg.no_pc_saved:
        print('Saving point clouds...')

        for i, pcd in enumerate(pcds):
        
            # Flip it, otherwise the pointcloud will be upside down
            # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            
            fn = os.path.join(out_dir, f"frame_{i:06d}_pc.ply")
            o3d.io.write_point_cloud(fn, pcd)
        
        print('Point clouds saved to', out_dir)
        

    
