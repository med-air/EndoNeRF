import open3d as o3d
import numpy as np
from particle_io import ParticleIO
import torch
from pytorch3d.ops import ball_query, knn_gather
import argparse




'''
Match particles to the skin
'''
def normalize_pts(pts):
    center = np.mean(pts, axis=0)
    pts = pts - center
    vmin, vmax = np.min(pts, axis=0), np.max(pts, axis=0)
    pts = pts / (vmax - vmin)
    return pts

def match_skin(par_pos, skin_file, skin_thickness=0.05, fill_color=np.array([0, 0, 0]), vis=False):
    print('Start matching skin...')
    skin_mesh = o3d.io.read_triangle_mesh(skin_file)

    skin_pc = skin_mesh.sample_points_uniformly(number_of_points=1000000)
    skin_pc.points = o3d.utility.Vector3dVector(normalize_pts(np.asarray(skin_pc.points)))
    skin_pc.estimate_normals()
    
    skin_pc_colors = np.asarray(skin_pc.colors)

    par_pos = normalize_pts(par_pos)

    # GPU-accelerated skin matching
    device = torch.device('cuda:0')

    skin_pc_gpu = torch.Tensor(np.asarray(skin_pc.points)).to(device)
    skin_pc_gpu = skin_pc_gpu.unsqueeze(0)

    par_pts_gpu = torch.Tensor(par_pos).to(device)
    par_pts_gpu = par_pts_gpu.unsqueeze(0)

    _, idx, _ = ball_query(par_pts_gpu, skin_pc_gpu, K=50, radius=skin_thickness)

    skin_pc_colors = np.concatenate([np.array([[0, 0, 0]]), skin_pc_colors], axis=0) # the first is the empty color, i.e., (0, 0, 0)
    skin_pc_colors_gpu = torch.Tensor(skin_pc_colors).to(device)
    skin_pc_colors_gpu = skin_pc_colors_gpu.unsqueeze(0)

    idx = idx + 1 # (1, P1, K), remap -1 to 0
    q_nums = (idx > 0).sum(dim=-1, keepdim=True) # (1, P1, 1)
    q_colors = knn_gather(skin_pc_colors_gpu, idx) # (1, P1, K, 3)
    q_colors_avg = q_colors.sum(dim=2) # (1, P1, 3)
    q_colors_avg = (q_colors_avg / q_nums).squeeze(0) # (P1, 3)
    q_colors_nan = torch.isnan(q_colors_avg).any(dim=1)  # (P1,)
    q_colors_avg[q_colors_nan] = torch.Tensor(fill_color).to(device)

    color_arr = q_colors_avg.cpu().numpy()

    print('Skin generated.')

    if vis:
        print('Visualizing...')
        par_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(par_pos))
        par_pc.colors = o3d.utility.Vector3dVector(color_arr)

        o3d.visualization.draw_geometries([par_pc])
    
    return color_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--skin_file", type=str, required=True,
                        help='The mesh file path used in the simulation path')
    parser.add_argument("--particle_file", type=str, default='./sim_results/0001.npz',
                        help='A particle file of simulation output')
    parser.add_argument("--out_file", type=str, default='./particle_skin.npy',
                        help='The output particle skin file')
    parser.add_argument("--vis", action='store_true',
                        help='Visualize results with Open3D')
    args = parser.parse_args()

    x, v, idx, types = ParticleIO.read_particles_3d(args.particle_file)

    is_softbody = (types == 0).squeeze(-1)

    # colors = np.ones_like(x) * 0.5
    colors = np.ones((idx.max() + 1, 3)) * 0.5

    softbody_par = x[is_softbody]
    softbody_par_idx = idx[is_softbody].squeeze(-1)

    skin_colors = match_skin(softbody_par, args.skin_file, skin_thickness=0.05, fill_color=np.array([1., 0, 0]), vis=args.vis)
    colors[softbody_par_idx] = skin_colors

    colors = (colors * 255).astype(np.uint8)

    np.save(args.out_file, colors)