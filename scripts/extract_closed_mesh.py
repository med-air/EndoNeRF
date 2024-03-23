import open3d as o3d
import numpy as np
import os
import argparse
from pathlib import Path


'''
Fill a surface to a solid
'''

def scan_silhouette_mesh(mesh, eps=0.1):
    bound_edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
    bound_edges = np.asarray(bound_edges)

    adj_dict = {}
    for edge in bound_edges:
        a, b = edge[0], edge[1]

        if a not in adj_dict:
            adj_dict[a] = [b]
        else:
            adj_dict[a].append(b)
        
        if b not in adj_dict:
            adj_dict[b] = [a]
        else:
            adj_dict[b].append(a)
        
        if len(adj_dict[a]) > 2:
            print(f'WARNING: Vertex {a} included in more than 2 boundary edges')
        if len(adj_dict[b]) > 2:
            print(f'WARNING: Vertex {b} included in more than 2 boundary edges')

    fringe = [bound_edges[0, 0]]
    visited = []
    while len(fringe) > 0:
        ele = fringe.pop()
        visited.append(ele)
        adj = adj_dict[ele]
        for k in adj:
            if k not in visited:
                fringe.append(k)
                continue
    visited.append(bound_edges[0, 0])
    
    return np.array(visited)

def close_mesh(mesh, scale_max_depth=1.5, filled_frustum=False, frustum_far_scale=2.5):
    ''' Require mesh is located in a right-handed coordinate system. The surface is forwarding +z.
    '''
    
    vtx = np.asarray(mesh.vertices)
    vtx_colors = np.asarray(mesh.vertex_colors)
    num_vtx = vtx.shape[0]

    max_depth = np.min(vtx, axis=1)[2] * scale_max_depth
    fill_pts = []
    fill_pts_colors = []
    fill_triangles = []

    bound = scan_silhouette_mesh(mesh)
    center = np.mean(vtx[bound], axis=0)
    center[2] = max_depth
    center_color = np.mean(vtx_colors[bound], axis=0)
    fill_pts.append(center)
    fill_pts_colors.append(center_color)

    for j in range(bound.shape[0]):
        idx = bound[j]

        if not filled_frustum:
            new_pt = np.array([vtx[idx, 0], vtx[idx, 1], max_depth])
        else:
            new_pt = np.array([center[0] + (vtx[idx, 0] - center[0]) * frustum_far_scale, center[1] + (vtx[idx, 1] - center[1]) * frustum_far_scale, max_depth])
        fill_pts.append(new_pt)
        fill_pts_colors.append(np.array([vtx_colors[idx, 0], vtx_colors[idx, 1], vtx_colors[idx, 2]]))

        if j < bound.shape[0] - 1:
            idx_next = bound[j + 1]
            fill_triangles.append(np.array([idx, idx_next, num_vtx + len(fill_pts) - 1]))

        if j > 0:
            fill_triangles.append(np.array([num_vtx + len(fill_pts) - 2, idx, num_vtx + len(fill_pts) - 1]))
            fill_triangles.append(np.array([num_vtx + len(fill_pts) - 2, num_vtx, num_vtx + len(fill_pts) - 1]))

            
    new_pts = np.concatenate([vtx, np.stack(fill_pts)])
    new_pts_colors = np.concatenate([vtx_colors, np.stack(fill_pts_colors)])
    new_triangles = np.concatenate([np.asarray(mesh.triangles), np.stack(fill_triangles)])

    mesh.vertices = o3d.utility.Vector3dVector(new_pts)
    mesh.vertex_colors = o3d.utility.Vector3dVector(new_pts_colors)
    mesh.triangles = o3d.utility.Vector3iVector(new_triangles)

    

if __name__ == '__main__':
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--pc_filepath", type=str, required=True,
                        help='Point cloud file path')
    parser.add_argument("--output_path", type=str, default='./',
                        help='Output directory path')
    parser.add_argument("--pc_ds_rate", type=int, default=10,
                        help='Point cloud downsample rate')
    parser.add_argument("--mesh_simplify_level", type=int, default=5,
                        help='Mesh simplificiation level')
    parser.add_argument("--scale_z", type=float, default=1.2,
                        help='Scale of the max depth (thickness)')
    parser.add_argument("--filled_frustum", action='store_true',
                        help='Filled the mesh surface into cone shape')
    parser.add_argument("--frustum_far_scale", type=float, default=2.5,
                        help='Scale of the frustum far plane')
    parser.add_argument("--vis", action='store_true',
                        help='Visualize results with Open3D')
    args = parser.parse_args()


    '''Rotate and normalize point clouds
    '''
    pc_fn = Path(args.pc_filepath).stem
    pc = o3d.io.read_point_cloud(args.pc_filepath)
    if args.pc_ds_rate >= 0:
        pc = pc.uniform_down_sample(args.pc_ds_rate)
    # Rotate to forward facing
    pc.rotate(pc.get_rotation_matrix_from_xyz((-np.pi,0,0)))
    # Shift to origin
    pc.translate(-pc.get_center())
    # Normalize
    bbox = pc.get_max_bound() - pc.get_min_bound()
    scale = 1 / bbox.max()
    pc.scale(scale, np.array([0, 0, 0]))

    '''Poisson surface reconstruction
    '''
    print('Poisson surface reconstruction...')

    pc.estimate_normals()
    pc.orient_normals_consistent_tangent_plane(50)

    mesh_recon, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8, width=0, scale=1)
    if args.mesh_simplify_level > 1:
        mesh_recon = mesh_recon.simplify_quadric_decimation(np.asarray(mesh_recon.triangles).shape[0] // args.mesh_simplify_level)

    mesh_recon_file = os.path.join(args.output_path, f'{pc_fn}_mesh.obj')
    o3d.io.write_triangle_mesh(mesh_recon_file, mesh_recon)

    print(f'Saved surface reconstruction to {mesh_recon_file}')

    if args.vis:
        o3d.visualization.draw_geometries([mesh_recon], mesh_show_wireframe=True, mesh_show_back_face=True)

    '''Closed mesh
    '''
    print('Closed mesh reconstruction...')

    mesh_recon = o3d.io.read_triangle_mesh(os.path.join(args.output_path, f'{pc_fn}_mesh.obj'))
    close_mesh(mesh_recon, args.scale_z, args.filled_frustum, args.frustum_far_scale)

    closed_mesh_file = os.path.join(args.output_path, f'{pc_fn}_closed_mesh.obj')
    o3d.io.write_triangle_mesh(closed_mesh_file, mesh_recon)

    print(f'Saved closed mesh to {closed_mesh_file}')

    if args.vis:
        o3d.visualization.draw_geometries([mesh_recon], mesh_show_wireframe=True, mesh_show_back_face=True)


