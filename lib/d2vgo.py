import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo

from lib.dvgo import DirectVoxGO, get_rays_of_a_view, render_utils_cuda, total_variation_cuda, Alphas2Weights, Raw2Alpha


'''Model
'''


class DynamicDirectVoxGO(DirectVoxGO):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 num_dis_res=0, num_times=0, num_dis_scales=0,
                 dis_comp_n=8, dis_feat_dim=6, dis_feat_pe=2,
                 dis_net_width=128, dis_net_depth=3,
                 zero_canonical=False,
                 **kwargs):
        super(DynamicDirectVoxGO, self).__init__(xyz_min, xyz_max,
                 num_voxels, num_voxels_base,
                 alpha_init,
                 mask_cache_path, mask_cache_thres,
                 fast_color_thres,
                 rgbnet_dim, rgbnet_direct, rgbnet_full_implicit,
                 rgbnet_depth, rgbnet_width,
                 viewbase_pe,
                 **kwargs)

        self.zero_canonical = zero_canonical
        self.dis_feat_pe = dis_feat_pe
        self.dis_comp_n = dis_comp_n
        self.dis_feat_dim = dis_feat_dim
        self.dis_net_width = dis_net_width
        self.dis_net_depth = dis_net_depth
        
        self.num_dis_res = num_dis_res
        self.num_times = num_times
        self.num_dis_scales = num_dis_scales

        # init positional encoding for displacement features
        self.register_buffer('disfreq', torch.FloatTensor([(2 ** i) for i in range(dis_feat_pe)]))
        dis_feat_dim0 = (dis_feat_dim + dis_feat_dim * dis_feat_pe * 2)

        # init displacement net
        assert dis_net_depth > 2, 'The displacement network depth should be greater than 2!'
        self.dis_net_depth = dis_net_depth
        self.dis_net = nn.ModuleList([nn.Linear(dis_feat_dim0, dis_net_width),])
        for i in range(dis_net_depth - 2):
            self.dis_net.append(nn.Linear(dis_net_width, dis_net_width)) # hidden layer
        self.dis_net.append(nn.Linear(dis_net_width, 3)) # out layer

        # init displacement basis
        self.dis_basis_mat = nn.Linear(4 * dis_comp_n * num_dis_scales, dis_feat_dim, bias=False)

        self.cube_ids = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        self.line_ids = [3, 2, 1, 0]
        shapes = [self.num_dis_res, self.num_dis_res, self.num_dis_res, self.num_times]
        cubes = []
        lines = []

        for _ in range(self.num_dis_scales):
            for i in range(len(self.line_ids)):
                line_i, cube_i = self.line_ids[i], self.cube_ids[i], 
                cubes.append(torch.nn.Parameter(0.1 * torch.randn((1, dis_comp_n, shapes[cube_i[2]], shapes[cube_i[1]], shapes[cube_i[0]]))))
                lines.append(torch.nn.Parameter(0.1 * torch.randn((1, dis_comp_n, shapes[line_i], 1, 1)))) # (fake 3d to use grid_sample)

            shapes = [shapes[0] * 2, shapes[1] * 2, shapes[2] * 2, shapes[3] * 2]
        
        self.dis_cubes = torch.nn.ParameterList(cubes)
        self.dis_lines = torch.nn.ParameterList(lines)

    def get_kwargs(self):
        kwargs = super(DynamicDirectVoxGO, self).get_kwargs()
        kwargs['num_dis_res'] = self.num_dis_res
        kwargs['num_times'] = self.num_times
        kwargs['num_dis_scales'] = self.num_dis_scales
        kwargs['zero_canonical'] = self.zero_canonical
        kwargs['dis_comp_n'] = self.dis_comp_n
        kwargs['dis_feat_dim'] = self.dis_feat_dim
        kwargs['dis_feat_pe'] = self.dis_feat_pe
        kwargs['dis_net_width'] = self.dis_net_width
        kwargs['dis_net_depth'] = self.dis_net_depth

        return kwargs

    def hit_bbox(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the bbox'''
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        
        hit = render_utils_cuda.rays_hit_test_bbox(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        return hit.reshape(shape)

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, num_dis_res, num_times):
        super(DynamicDirectVoxGO, self).scale_volume_grid(num_voxels)

        if self.num_dis_scales > 1:
            return

        print('dvgo: scale_volume_grid scale times from', self.num_times, 'to', num_times)
        print('dvgo: scale_volume_grid scale displacement from', self.num_dis_res, 'to', num_dis_res)
        
        self.num_times = num_times
        self.num_dis_res = num_dis_res

        shapes = [self.num_dis_res, self.num_dis_res, self.num_dis_res, self.num_times]
        for i in range(len(self.line_ids)):
            line_i, cube_i = self.line_ids[i], self.cube_ids[i], 
            self.dis_cubes[i] = torch.nn.Parameter(F.interpolate(self.dis_cubes[i].data, size=(shapes[cube_i[2]], shapes[cube_i[1]], shapes[cube_i[0]]), mode='trilinear', align_corners=True))
            self.dis_lines[i] = torch.nn.Parameter(F.interpolate(self.dis_lines[i].data, size=(shapes[line_i], 1, 1), mode='trilinear', align_corners=True))

    def compute_displacement(self, xyz, t):
        '''Estimate displacement
        @xyz:   [M, 3] spatial coordinates.
        @t  :   [M, 1] time coordinates.
        '''

        N = xyz.shape[0]

        if not N > 0:
            dis = torch.zeros(N, 3)
            vec_feat = torch.zeros(self.dis_comp_n, N)
        else:
            # if self.training:
            #     t = t + (torch.rand_like(t) - 0.5) * (1.0 / self.num_times)
            
            xyz = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2. - 1. # scale to [-1, 1]
            t = t * 2. - 1. # scale to [-1, 1]

            x = torch.stack((xyz[..., 0], xyz[..., 1], xyz[..., 2], t[..., 0]), dim=-1)
            cube_coords = torch.stack((x[..., self.cube_ids[0]], x[..., self.cube_ids[1]], x[..., self.cube_ids[2]], x[..., self.cube_ids[3]])).detach().view(4, -1, 1, 1, 3) # [4, N, 1, 1, 3]
            line_coords = torch.stack((x[..., self.line_ids[0]], x[..., self.line_ids[1]], x[..., self.line_ids[2]], x[..., self.line_ids[3]]))
            line_coords = torch.stack((torch.zeros_like(line_coords), torch.zeros_like(line_coords), line_coords), dim=-1).detach().view(4, -1, 1, 1, 3) # [4, N, 1, 1, 3], fake 3d coord

            cube_feat, line_feat = [], []

            for i in range(len(self.dis_cubes)):
                cube = self.dis_cubes[i]
                line = self.dis_lines[i]
                cube_feat.append(F.grid_sample(cube, cube_coords[[i % 4]], align_corners=True, mode='bilinear').view(-1, N)) # [1, R, N, 1] --> [R, N]
                line_feat.append(F.grid_sample(line, line_coords[[i % 4]], align_corners=True, mode='bilinear').view(-1, N)) # [R, N]
            
            vec_feat = torch.cat(cube_feat, dim=0) * torch.cat(line_feat, dim=0) # [4 * R, N]

            dis = self.dis_basis_mat(vec_feat.T) # [N, 4 * R] --> [N, dis_feat_dim]

            disfeat_emb = (dis.unsqueeze(-1) * self.disfreq).flatten(-2)
            disfeat_emb = torch.cat([dis, disfeat_emb.sin(), disfeat_emb.cos()], -1)
            dis = disfeat_emb.flatten(0, -2)

            for l in range(self.dis_net_depth):
                dis = self.dis_net[l](dis)
                if l != self.dis_net_depth - 1:
                    dis = F.relu(dis, inplace=True)

        return {
            'dis': dis,
            'dis_enc': vec_feat
        }
    
    def render_rays(self, N, ray_pts, ray_id, step_id, viewdirs, global_step=None, t_min=0, t_max=0, dis=None, **render_kwargs):
        '''Volume rendering'''

        ret_dict = {}
        
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        stepdist = render_kwargs['stepsize'] * self.voxel_size

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            if dis is not None:
                dis = dis[mask]

        # query for alpha w/ post-activation
        density = self.grid_sampler(ray_pts, self.density)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]
            if dis is not None:
                dis = dis[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            if dis is not None:
                dis = dis[mask]

        # query for color
        if not self.rgbnet_full_implicit:
            k0 = self.grid_sampler(ray_pts, self.k0)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[:, 3:]
                k0_diffuse = k0[:, :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            if self.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb = torch.sigmoid(rgb_logit + k0_diffuse)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
        })

        if render_kwargs.get('render_depth', False):
            # with torch.no_grad():
            depth = segment_coo(
                    src=(weights * step_id * stepdist),
                    index=ray_id,
                    out=torch.zeros([N]),
                    reduce='sum') + t_min 
            ret_dict.update({'depth': depth})
        
        if render_kwargs.get('render_displ', False):
            with torch.no_grad():
                displ = torch.zeros_like(rgb_marched)
                if dis is not None:
                    displ = segment_coo(
                            src=(weights.unsqueeze(-1) * dis),
                            index=ray_id,
                            out=torch.zeros([N, 3]),
                            reduce='sum')
                ret_dict.update({'displ': displ})

        return ret_dict


    def sample_ray_depth_cueing(self, rays_o, rays_d, depth, depth_sampling_std, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        depth = depth.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_depth_cueing_on_rays(
            rays_o, rays_d, depth, self.xyz_min, self.xyz_max, depth_sampling_std, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id, t_min, t_max


    def forward(self, rays_o, rays_d, viewdirs, times, depth=None, global_step=None, **render_kwargs):
        '''Forward pass
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        @times:    [N, 1] times to compute positional embedding for MLP
        '''

        assert torch.unique(times).shape[0] == 1, "Only accepts all points from same time"

        N = len(rays_o)

        # sample points on rays
        if depth is not None:
            ray_pts, ray_id, step_id, t_min, t_max = self.sample_ray_depth_cueing(
                rays_o=rays_o, rays_d=rays_d, depth=depth, is_train=global_step is not None, **render_kwargs)
        else:
            ray_pts, ray_id, step_id, t_min, t_max = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)

        times = times[ray_id]
        displ = None
        if times.shape[0] > 0:
            dis_ret = self.compute_displacement(ray_pts, times)
            displ = dis_ret['dis'] #* 0.3

            if self.zero_canonical and times[0] == 0:
                displ = displ * torch.zeros_like(ray_pts)

            ray_pts = ray_pts + displ

        ret = self.render_rays(N, ray_pts, ray_id, step_id, viewdirs, global_step, t_min, t_max, displ, **render_kwargs)

        return ret
        


''' Ray and batch
'''


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, train_times, HW, Ks, ROIs, depths_tr_ori, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    depths_tr = torch.zeros([N], device=DEVICE)
    masks_tr = torch.zeros([N], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    tops = []
    for c2w, ts, img, (H, W), K, ROI, depth_map in zip(train_poses, train_times, rgb_tr_ori, HW, Ks, ROIs, depths_tr_ori):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        mask = (ROI > 0.5).to(DEVICE).float()

        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        depths_tr[top:top+n].copy_(depth_map.flatten(0,1))
        masks_tr[top:top+n].copy_(mask.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        tops.append(top)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, depths_tr, masks_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, tops


@torch.no_grad()
def get_training_rays_in_roi_sampling(rgb_tr_ori, train_poses, train_times, HW, Ks, ROIs, depths_tr_ori, ndc, inverse_y, flip_x, flip_y, camera_type, rc_device, divide_rays_by_time=False):
    print('get_training_rays_in_roi_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    train_poses = train_poses.to(rc_device)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    depths_tr = torch.zeros([N], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    times_tr = torch.zeros((rgb_tr.shape[0], 1), device=DEVICE)
    imsz = []
    top = 0
    tops = []
    pre_ts = torch.tensor([-1.]) # used to check if time is changed
    for c2w, ts, img, (H, W), K, ROI, depth_map in zip(train_poses, train_times, rgb_tr_ori, HW, Ks, ROIs, depths_tr_ori):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, camera_type=camera_type)        
        mask = (ROI > 0.5).to(DEVICE).bool()
        n = mask.sum().cpu().item()
        rgb_tr[top:top+n].copy_(img[mask])
        depths_tr[top:top+n].copy_(depth_map[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        times_tr[top:top+n].copy_(torch.tile(ts, (n, 1)).to(DEVICE))
        imsz.append(n)
        if not divide_rays_by_time or (pre_ts != ts):
            pre_ts = ts
            tops.append(top)
        top = top + n

        del rays_o, rays_d, viewdirs, mask

    del train_poses

    print('get_training_rays_in_roi_sampling: ratio', top / N)
    
    rgb_tr = rgb_tr[:top]
    depths_tr = depths_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    times_tr = times_tr[:top]

    eps_time = time.time() - eps_time
    print('get_training_rays_in_roi_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, depths_tr, rays_o_tr, rays_d_tr, viewdirs_tr, times_tr, imsz, tops

