import taichi as ti
import os
import sys
import time
from pathlib import Path
import argparse
import numpy as np
import random
import glob

from utils import create_output_folder

from renderer import Renderer

from particle_io import ParticleIO



parser = argparse.ArgumentParser()
parser.add_argument("--particle_skin", type=str, default='./particle_skin.npy',
                    help='The particle skin file')
parser.add_argument("--sim_dir", type=str, default='./sim_results/',
                    help='Simulation result folder containing .npz files')
parser.add_argument("--out_dir", type=str, default='./render_results/',
                    help='The render output directory')
args = parser.parse_args()

render_out_dir = create_output_folder(args.out_dir, no_postfix=True)
sim_input_dir = args.sim_dir
skin_file = args.particle_skin



res = 512
particle_radius = 0.8
shutter_time = 0
max_particles = 128  # in million
spp = 200
begin_frame = 0
end_frame = 119
skip_frame_step = 1
fixed_frame = None
force_save = True
num_cams = 360
cam_dist = 12
fov = 0.23
object_position = (0.4, 0.5, 0.5)
cam_pose = [
    [0.401,  1.3, 0.7],
    object_position
]  # camera center, look-at

random.seed(0)


def build_scene(par_file, skin_colors):
    x, v, idx, types = ParticleIO.read_particles_3d(par_file)

    colors = skin_colors[idx.squeeze(-1)]

    return x, v, colors

if __name__ == '__main__':
    '''
    Render simulations
    '''

    ti.init(arch=ti.cuda, device_memory_GB=9)

    renderer = Renderer(dx=1 / res,
                        sphere_radius=particle_radius / res,
                        shutter_time=shutter_time,
                        max_num_particles_million=max_particles,
                        taichi_logo=False)
    renderer.set_fov(fov)
    renderer.set_light_directions([
        [0, -1.0, 0]
    ])
    renderer.set_light_color([1.0, 1.0, 1.0])
    renderer.set_light_power(1.)

    fn_ls = [fn for fn in os.listdir(sim_input_dir) if fn.endswith('.npz')]
    if begin_frame > len(fn_ls) or begin_frame < 0:
        begin_frame = len(fn_ls)
    if end_frame > len(fn_ls) or end_frame < 0:
        end_frame = len(fn_ls)

    input_sim_files = sorted(glob.glob(os.path.join(sim_input_dir, '*.npz')))

    skin_colors = np.load(skin_file)

    for f in range(begin_frame, end_frame + 1, skip_frame_step):
        print('frame', f, end=' ')

        t = time.time()

        if fixed_frame is None:
            # cur_render_input = f'{sim_input_dir}/{f:05d}.npz'
            cur_render_input = input_sim_files[f]
        else:
            cur_render_input = f'{sim_input_dir}/{fixed_frame}.npz'
        if not os.path.exists(cur_render_input):
            print(f'warning, {cur_render_input} not existed, skip!')
            continue

        output_fn = f'{render_out_dir}/{f:05d}.png'
        if os.path.exists(output_fn) and not force_save:
            print('skip.')
            continue
        else:
            print('rendering...')
        Path(output_fn).touch()

        renderer.floor_height[None] = 0.2

        np_x, np_v, np_color = build_scene(cur_render_input, skin_colors)
        renderer.initialize_particles_from_mpm(np_x, np_v, np_color)

        total_voxels = renderer.total_non_empty_voxels()
        total_inserted_particles = renderer.total_inserted_particles()
        print('Total particles (w/ motion blur)', total_inserted_particles)
        print('Total nonempty voxels', total_voxels)
        print('Average particle_list_length', total_inserted_particles / total_voxels)

        cam_pos, cam_lookat = cam_pose
        renderer.set_camera_pos(cam_pos[0], cam_pos[1], cam_pos[2])
        renderer.set_look_at(cam_lookat[0], cam_lookat[1], cam_lookat[2])

        img = renderer.render_frame(spp=spp)

        ti.imwrite(img, output_fn)
        ti.print_memory_profile_info()
        print(f'Frame rendered. {spp} take {time.time() - t} s.')