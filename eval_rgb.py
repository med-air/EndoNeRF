
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from math import exp
import configargparse
import random, time
import imageio
import lpips


'''
SSIM utils
'''

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class ssim_utils:
    @staticmethod
    def ssim(img1, img2, window_size = 11, size_average = True):
        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel)
        
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        
        return _ssim(img1, img2, window, window_size, channel, size_average)


'''
Metrics
'''

lpips_alex = lpips.LPIPS(net='alex') # best forward scores
lpips_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def img2mse(x, y, reduction='mean'):
    diff = torch.mean((x - y) ** 2, -1)
    if reduction == 'mean':
        return torch.mean(diff)
    elif reduction == 'sum':
        return torch.sum(diff)
    elif reduction == 'none':
        return diff

def mse2psnr(x):
    if isinstance(x, float):
        x = torch.tensor([x])
    return -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

def ssim(img1, img2, window_size = 11, size_average = True, format='NCHW'):
    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    return ssim_utils.ssim(img1, img2, window_size, size_average)

def lpips(img1, img2, net='alex', format='NCHW'):
    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    if net == 'alex':
        model = lpips_alex.to(img1.device)
        return model(img1, img2)
    elif net == 'vgg':
        model = lpips_vgg.to(img1.device)
        return model(img1, img2)

def to8b(x):
    return (255*(x-x.min())/(x.max()-x.min())).astype(np.uint8)

def export_images(rgbs, save_dir, H=0, W=0):
    rgb8s = []
    for i, rgb in enumerate(rgbs):
        # Resize
        if H > 0 and W > 0:
            rgb = rgb.reshape([H, W])

        filename = os.path.join(save_dir, '{:03d}.npy'.format(i))
        np.save(filename, rgb)
        
        # Convert to image
        rgb8 = to8b(rgb)
        filename = os.path.join(save_dir, '{:03d}.png'.format(i))
        imageio.imwrite(filename, rgb8)
        rgb8s.append(rgb8)
    
    return np.stack(rgb8s, 0)

def export_video(rgbs, save_path, fps=30, quality=8):
    imageio.mimwrite(save_path, to8b(rgbs), fps=fps, quality=quality)


'''
Eval
'''

if __name__ == '__main__':
    cfg_parser = configargparse.ArgumentParser()
    cfg_parser.add_argument('--gt_dir', type=str, required=True)
    cfg_parser.add_argument('--mask_dir', type=str, required=True)
    cfg_parser.add_argument('--img_dir', type=str, required=True)
    cfg = cfg_parser.parse_args()

    gt_dir = cfg.gt_dir
    mask_dir = cfg.mask_dir
    img_dir = cfg.img_dir
    start_index = 1
    skip_gt = 0

    gt_list = [imageio.imread(os.path.join(gt_dir, fn)) for fn in sorted(os.listdir(gt_dir)) if fn.endswith('.png')]
    mask_list = [imageio.imread(os.path.join(mask_dir, fn)) for fn in sorted(os.listdir(mask_dir)) if fn.endswith('.png')]
    img_list = [imageio.imread(os.path.join(img_dir, fn)) for fn in sorted(os.listdir(img_dir)) if fn.endswith('.png')]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    masks = np.stack(mask_list, axis=0).astype(np.float64)[skip_gt:] / 255.0
    gts = np.stack(gt_list, axis=0).astype(np.float64)[skip_gt:] / 255.0
    imgs = np.stack(img_list, axis=0).astype(np.float64) / 255.0

    if gts.shape[0] > imgs.shape[0]:
        gts =  gts[:imgs.shape[0]]
        masks =  masks[:imgs.shape[0]]

    print('Shapes (gt, imgs, masks):', gts.shape, imgs.shape, masks.shape)

    masks = torch.Tensor(1.0 - masks).to(device).unsqueeze(-1)
    gts = torch.Tensor(gts).to(device) * masks
    imgs = torch.Tensor(imgs).to(device) * masks
    masks = masks[start_index:]
    gts = gts[start_index:]
    imgs = imgs[start_index:]


    mse = img2mse(imgs, gts)
    psnr = mse2psnr(mse)
    ssim_ = ssim(imgs, gts, format='NHWC')
    lpips_ = lpips(imgs, gts, format='NHWC')

    print('PSNR:', psnr.item())
    print('SSIM:', ssim_.item())
    print('LPIPS:', torch.mean(lpips_).item())
