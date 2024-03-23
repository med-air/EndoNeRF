import numpy as np

from .load_dynamic_llff import load_llff_data

def load_data(args):

    K, bbox, depths, ROIs, render_K = None, None, None, None, None

    print(f'Dataset type: {args.dataset_type}')

    if args.dataset_type == 'llff':
        images, ROIs, depths, poses, times, bds, render_poses, render_times, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75, spherify=args.spherify, fg_mask=True, use_depth=True,
                                                                  render_path=args.llff_renderpath, davinci_endoscopic=True)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[1:-1:args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val and i not in args.skip_frames)])
        # i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in args.skip_frames)])


        print('DEFINING BOUNDS')
        
        close_depth, inf_depth = np.ndarray.min(bds) * .9, np.ndarray.max(bds) * 1.

        if not args.ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    # Initialize camera intrinsics from hwf if intrinsic matrices are not provided
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    times = times[..., np.newaxis]
    print('min time:', times.min(), 'max time:', times.max())
    render_times = render_times[..., np.newaxis]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks, near=near, far=far, bbox=bbox,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses, render_K=render_K,
        images=images, depths=depths, ROIs=ROIs,
        irregular_shape=irregular_shape,
        times=times, render_times=render_times,
    )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far

