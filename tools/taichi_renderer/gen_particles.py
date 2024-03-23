import os
import partio
import numpy as np
import glob
from pathlib import Path
import argparse


'''
Read particles to numpy array
'''
def read_geo_to_array(fn):
    try:
        p = partio.read(fn)
    except Exception as e:
        return None, None, None, None, None
    
    pos_attr = p.attributeInfo("position")
    v_attr = p.attributeInfo("v")
    idx_attr = p.attributeInfo("index")
    type_attr = p.attributeInfo("type")

    particle_pos = []
    particle_v = []
    particle_idx = []
    particle_type = []

    N = p.numParticles()

    for i in range(N):
        t = p.get(type_attr, i)
        particle_type.append(t)

        pos = p.get(pos_attr, i)
        particle_pos.append(pos)

        idx = p.get(idx_attr, i)
        particle_idx.append(idx)

        v = p.get(v_attr, i)
        particle_v.append(v)

    particle_type = np.array(particle_type)
    particle_pos = np.array(particle_pos)
    particle_v = np.array(particle_v)
    particle_idx = np.array(particle_idx)

    ranges_x = [np.min(particle_pos, axis=0), np.max(particle_pos, axis=0)]
    ranges_v = [np.min(particle_v, axis=0), np.max(particle_v, axis=0)]
    ranges = np.stack([ranges_x, ranges_v], axis=0)

    return ranges, particle_pos, particle_v, particle_idx, particle_type


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bgeo_dir", type=str, required=True,
                        help='directory containing simulation results (.bgeo)')
    parser.add_argument("--output_path", type=str, default='./sim_results/',
                        help='Output directory path')
    args = parser.parse_args()

    particle_files = glob.glob(os.path.join(args.bgeo_dir, '/*.bgeo'))
    particle_files = sorted(particle_files)
    out_dir = args.output_path

    for iter, fn in enumerate(particle_files):
        if iter % 10 == 0:
            print(f'{iter}/{len(particle_files)}')

        ranges, particle_pos, particle_v, particle_idx, particle_type = read_geo_to_array(fn)

        output_fn = os.path.join(out_dir, f'{Path(fn).stem}')
        np.savez(output_fn, ranges=ranges, x=particle_pos, v=particle_v, type=particle_type, idx=particle_idx)

        del ranges, particle_pos, particle_v, particle_idx, particle_type
