import taichi as tc
from math import *

r = 240

if __name__ == '__main__':
  use_flip = False
  mpm = tc.dynamics.MPM(
      res=(r + 1, r + 1, r + 1),
      base_delta_t=5e-5,
      num_frames=100,
      penalty=1e3,
      cfl=0.5,
      rpic_damping=1,)

  levelset = mpm.create_levelset()
  levelset.add_plane(tc.Vector(0, 1, 0), -0.25)
  # levelset.add_plane(tc.Vector(1, 0, 0), -0.16)
  levelset.set_friction(-1)
  mpm.set_levelset(levelset, False)

  tex = tc.Texture(
      'mesh',
      resolution=(2 * r, 2 * r, 2 * r),
      translate=(0.4, 0.5, 0.55),
      scale=(0.4, 0.4, 0.4),
      adaptive=False,
      filename='$mpm/soft_tissues_s2.obj') * 10

  mpm.add_particles(
      type='von_mises',
      pd=True,
      density_tex=tex.id,
      initial_velocity=(0, 0, 0),
      density=400,
      color=(0.8, 0.7, 1.0),
      initial_position=(0.5, 0.5, 0.5),
      youngs_modulus=4e5,
      poisson_ratio=0.4,
      yield_stress=25)

  tgt_dy = 0.08
  tgt_dx = 0.18
  x_pos = 0.4
  y_pos = 0.35
  z_pos = 0.55
  velo_y = -0.15
  velo_x = -0.3
  velo_z = 0.1
  def position_function(t):
    y_dist = velo_y * t
    if abs(y_dist) < tgt_dy:
      return tc.Vector(x_pos, y_pos + y_dist, z_pos)
    else:
      t = t - tgt_dy / abs(velo_y)
      x_dist = velo_x * t
      if abs(x_dist) < tgt_dx:
        return tc.Vector(x_pos + x_dist, y_pos - tgt_dy, z_pos + velo_z * t)
      else:
        dt = tgt_dx / abs(velo_x)
        t = t - dt
        return tc.Vector(x_pos - tgt_dx, y_pos - tgt_dy, z_pos + velo_z * dt - velo_z * t)

  def rotation_function(t):
    return tc.Vector(0, 0, 0.)

  mpm.add_particles(
      type='rigid',
      density=40,
      scale=(1, 1, 1),
      friction=0,
      scripted_position=tc.function13(position_function),
      scripted_rotation=tc.function13(rotation_function),
      codimensional=False,
      mesh_fn='$mpm/psm_part_newgripper.obj')

  mpm.simulate(clear_output_directory=True)
