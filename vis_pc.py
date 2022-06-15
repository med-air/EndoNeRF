import open3d as o3d
import numpy as np
import imageio
import os
import configargparse


###################################################################################################
# Usage Example
###################################################################################################

# python vis_pc.py --pc_dir logs/example_training/reconstructed_pcds_100000

###################################################################################################


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


class PointCloudSequenceVisualizer:
    def __init__(self, pcd_list, stall_count=0, save_dir='./', no_autoplay=False, no_loop=False, rec_video_fps=30, cam_move='none'):
        self.pcd_list = pcd_list

        self.stall_count = stall_count
        self.stall_index = 0

        self.playing = not no_autoplay
        self.loop = not no_loop

        self.cam_movement = cam_move
        self.cam_move_params = {
            'swing': {'move_dir': 1, 'move_accum': 0}
        }

        self.recording = False
        self.rec_video_fps = rec_video_fps
        self.rec_buffer = []

        self.frame_idx = 0

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()

        self.save_dir = save_dir

        # initialize point clouds
        self.geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(self.pcd_list[self.frame_idx].points).copy()))
        self.geometry.colors = o3d.utility.Vector3dVector(np.asarray(self.pcd_list[self.frame_idx].colors).copy())
        self.vis.add_geometry(self.geometry)

        key_to_callback = {}
        key_to_callback[ord("A")] = lambda _: self._prev_frame()
        key_to_callback[ord("D")] = lambda _: self._next_frame()
        key_to_callback[ord("P")] = lambda _: self._pause_loop()
        key_to_callback[ord("R")] = lambda _: self._reset_cam_pose()
        key_to_callback[ord("O")] = lambda _: print('Current frame idx:', self.frame_idx)
        key_to_callback[ord("C")] = lambda _: self._save_cam_pose()
        key_to_callback[ord("L")] = lambda _: self._load_cam_pose()
        key_to_callback[ord("S")] = lambda _: self._capture_screenshot()
        key_to_callback[ord("V")] = lambda _: self._video_record()
        for k in key_to_callback:
            self.vis.register_key_callback(k, key_to_callback[k])

        print('####### Manual #######')
        print('A: previous frame')
        print('D: next frame')
        print('P: pause')
        print('R: reset camera pose')
        print('O: output frame index')
        print('C: save camera pose')
        print('L: load camera view')
        print('S: save screenshot')
        print('V: turn on/off screen recording')
        print('######################')

        self.vis.register_animation_callback(lambda _: self._loop_update_cb())

    def _loop_update_cb(self):
        if self.stall_index < self.stall_count:
            self.stall_index += 1
            return False
        else:
            self.stall_index = 0

        self.geometry.points = self.pcd_list[self.frame_idx].points
        self.geometry.colors = self.pcd_list[self.frame_idx].colors

        if self.playing:
            if self.recording:
                frame = self.vis.capture_screen_float_buffer(do_render=True)
                self.rec_buffer.append(frame)

            self.frame_idx += 1

            if self.frame_idx >= len(self.pcd_list):
                if not self.loop:
                    self.playing = False
                    self.frame_idx = len(self.pcd_list) - 1
                else:
                    self.frame_idx = 0
            
            self._update_camera_movement()

        return True

    def _update_camera_movement(self):
        ctr = self.vis.get_view_control()
        if self.cam_movement == 'swing':
            ctr.rotate(4.0 * self.cam_move_params['swing']['move_dir'], 0.0)
            self.cam_move_params['swing']['move_accum'] = self.cam_move_params['swing']['move_accum'] + self.cam_move_params['swing']['move_dir']

            if abs(self.cam_move_params['swing']['move_accum']) >= 30:
                self.cam_move_params['swing']['move_dir'] = -1 * self.cam_move_params['swing']['move_dir']
                self.cam_move_params['swing']['move_accum'] = 0


    def _reset_cam_pose(self):
        ctr = self.vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        init_param.extrinsic = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
        ctr.convert_from_pinhole_camera_parameters(init_param)

        self.stall_index = self.stall_count

        return True

    def _video_record(self):
        self.recording = not self.recording

        if not self.recording and len(self.rec_buffer) > 0:
            frames = [np.array(f) for f in self.rec_buffer]
            frames = np.stack(frames, 0)
            self.rec_buffer = []

            imageio.mimwrite(os.path.join(self.save_dir, 'rec_video.mp4'), to8b(frames), fps=self.rec_video_fps, quality=8)
        
        if not self.recording:
            print('Recording stopped. Video saved.')
        else:
            print('Start recording...')

        return True

    def _save_cam_pose(self):
        ctr = self.vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()

        filename = input('Camera pose save path: ')
        o3d.io.write_pinhole_camera_parameters(filename, param)

        return True
    
    def _load_cam_pose(self):
        filename = input('Camera pose load path: ')
        param = o3d.io.read_pinhole_camera_parameters(filename)
        # intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic

        ctr = self.vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        init_param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(init_param)

        return True
    
    def _capture_screenshot(self):
        filename = os.path.join(self.save_dir, 'frame_%s.png' % self.frame_idx)
        self.vis.capture_screen_image(filename, do_render=True)

        return True

    def _pause_loop(self):
        self.playing = not self.playing

        if not self.loop and self.frame_idx == len(self.pcd_list) - 1:
            self.playing = False
            self.frame_idx = 0
    
    def _next_frame(self):
        self.frame_idx += 1

        if self.frame_idx >= len(self.pcd_list):
            self.frame_idx = 0
                
        self.stall_index = self.stall_count
    
    def _prev_frame(self):
        self.frame_idx -= 1

        if self.frame_idx < 0:
            self.frame_idx = len(self.pcd_list) - 1
        
        self.stall_index = self.stall_count
    
    def run(self):
        self._reset_cam_pose()

        self.vis.run()
        self.vis.destroy_window()


if __name__ == '__main__':
    cfg_parser = configargparse.ArgumentParser()
    cfg_parser.add_argument('--pc_dir', type=str, 
                        help='point clouds directory')
    cfg_parser.add_argument('--vis_stall', type=int, default=10,
                        help='control visualization speed (bigger => slower)')
    cfg_parser.add_argument('--data_format', type=str, default='n', choices=['n', 's'])
    cfg_parser.add_argument('--save_dir', type=str, default='./',
                        help='directory for saving screenshots')
    cfg_parser.add_argument("--no_loop", action='store_true', 
                        help='loop playing?')
    cfg_parser.add_argument("--no_autoplay", action='store_true', 
                        help='auto playing?')
    cfg_parser.add_argument('--rec_video_fps', type=int, default=30,
                        help='FPS of video recording')
    cfg_parser.add_argument('--cam_move', type=str, default='none',
                        help='Movement of cameras: none / swing')

    cfg = cfg_parser.parse_args()

    if cfg.data_format == 'n':
        # Point cloud files are stored in a directory:
        pcd_fns = [os.path.join(cfg.pc_dir, fn) for fn in sorted(os.listdir(cfg.pc_dir)) if fn.endswith('.ply')]
    elif cfg.data_format == 's':
        # Output of surfelwarp
        pcd_fns = [os.path.join(cfg.pc_dir, dn, 'live.ply') for dn in sorted(os.listdir(cfg.pc_dir), key=lambda elem: int(elem[6:]))]

    pcd_list = []

    print('Loading point clouds...')
    for fn in pcd_fns:
        pcd = o3d.io.read_point_cloud(fn)
        pcd_list.append(pcd)

    print('Total:', len(pcd_list), 'point clouds loaded.')

    pc_vis = PointCloudSequenceVisualizer(pcd_list, cfg.vis_stall, cfg.save_dir, cfg.no_autoplay, cfg.no_loop, cfg.rec_video_fps, cfg.cam_move)
    pc_vis.run()

    

