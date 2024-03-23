import os, math
import numpy as np
import scipy.signal
from typing import List, Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_adam import MaskedAdam


''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    lr_names = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            lr_names.append(k)
            param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return MaskedAdam(param_group), lr_names


''' Checkpoint utils
'''
def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer):
    ckpt = torch.load(ckpt_path)
    start = ckpt['global_step']
    if 'occ_xyz_min' in ckpt['model_state_dict']: del ckpt['model_state_dict']['occ_xyz_min']
    if 'occ_xyz_max' in ckpt['model_state_dict']: del ckpt['model_state_dict']['occ_xyz_max']
    model.load_state_dict(ckpt['model_state_dict'])
    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, optimizer, start


def load_model(model_class, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'])
    if 'occ_xyz_min' in ckpt['model_state_dict']: del ckpt['model_state_dict']['occ_xyz_min']
    if 'occ_xyz_max' in ckpt['model_state_dict']: del ckpt['model_state_dict']['occ_xyz_max']
    model.load_state_dict(ckpt['model_state_dict'])
    return model


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()



''' Morton Code
'''

import sys

_DIVISORS = [180.0 / 2 ** n for n in range(32)]


def __part1by1_32(n):
    n &= 0x0000ffff                  # base10: 65535,      binary: 1111111111111111,                 len: 16
    n = (n | (n << 8))  & 0x00FF00FF # base10: 16711935,   binary: 111111110000000011111111,         len: 24
    n = (n | (n << 4))  & 0x0F0F0F0F # base10: 252645135,  binary: 1111000011110000111100001111,     len: 28
    n = (n | (n << 2))  & 0x33333333 # base10: 858993459,  binary: 110011001100110011001100110011,   len: 30
    n = (n | (n << 1))  & 0x55555555 # base10: 1431655765, binary: 1010101010101010101010101010101,  len: 31

    return n


def __part1by2_32(n):
    n &= 0x000003ff                  # base10: 1023,       binary: 1111111111,                       len: 10
    n = (n ^ (n << 16)) & 0xff0000ff # base10: 4278190335, binary: 11111111000000000000000011111111, len: 32
    n = (n ^ (n << 8))  & 0x0300f00f # base10: 50393103,   binary: 11000000001111000000001111,       len: 26
    n = (n ^ (n << 4))  & 0x030c30c3 # base10: 51130563,   binary: 11000011000011000011000011,       len: 26
    n = (n ^ (n << 2))  & 0x09249249 # base10: 153391689,  binary: 1001001001001001001001001001,     len: 28

    return n


def __unpart1by1_32(n):
    n &= 0x55555555                  # base10: 1431655765, binary: 1010101010101010101010101010101,  len: 31
    n = (n ^ (n >> 1))  & 0x33333333 # base10: 858993459,  binary: 110011001100110011001100110011,   len: 30
    n = (n ^ (n >> 2))  & 0x0f0f0f0f # base10: 252645135,  binary: 1111000011110000111100001111,     len: 28
    n = (n ^ (n >> 4))  & 0x00ff00ff # base10: 16711935,   binary: 111111110000000011111111,         len: 24
    n = (n ^ (n >> 8))  & 0x0000ffff # base10: 65535,      binary: 1111111111111111,                 len: 16

    return n


def __unpart1by2_32(n):
    n &= 0x09249249                  # base10: 153391689,  binary: 1001001001001001001001001001,     len: 28
    n = (n ^ (n >> 2))  & 0x030c30c3 # base10: 51130563,   binary: 11000011000011000011000011,       len: 26
    n = (n ^ (n >> 4))  & 0x0300f00f # base10: 50393103,   binary: 11000000001111000000001111,       len: 26
    n = (n ^ (n >> 8))  & 0xff0000ff # base10: 4278190335, binary: 11111111000000000000000011111111, len: 32
    n = (n ^ (n >> 16)) & 0x000003ff # base10: 1023,       binary: 1111111111,                       len: 10

    return n


def __part1by1_64(n):
    n &= 0x00000000ffffffff                  # binary: 11111111111111111111111111111111,                                len: 32
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF # binary: 1111111111111111000000001111111111111111,                        len: 40
    n = (n | (n << 8))  & 0x00FF00FF00FF00FF # binary: 11111111000000001111111100000000111111110000000011111111,        len: 56
    n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0F # binary: 111100001111000011110000111100001111000011110000111100001111,    len: 60
    n = (n | (n << 2))  & 0x3333333333333333 # binary: 11001100110011001100110011001100110011001100110011001100110011,  len: 62
    n = (n | (n << 1))  & 0x5555555555555555 # binary: 101010101010101010101010101010101010101010101010101010101010101, len: 63

    return n


def __part1by2_64(n):
    n &= 0x1fffff                            # binary: 111111111111111111111,                                         len: 21
    n = (n | (n << 32)) & 0x1f00000000ffff   # binary: 11111000000000000000000000000000000001111111111111111,         len: 53
    n = (n | (n << 16)) & 0x1f0000ff0000ff   # binary: 11111000000000000000011111111000000000000000011111111,         len: 53
    n = (n | (n << 8))  & 0x100f00f00f00f00f # binary: 1000000001111000000001111000000001111000000001111000000001111, len: 61
    n = (n | (n << 4))  & 0x10c30c30c30c30c3 # binary: 1000011000011000011000011000011000011000011000011000011000011, len: 61
    n = (n | (n << 2))  & 0x1249249249249249 # binary: 1001001001001001001001001001001001001001001001001001001001001, len: 61

    return n


def __unpart1by1_64(n):
    n &= 0x5555555555555555                  # binary: 101010101010101010101010101010101010101010101010101010101010101, len: 63
    n = (n ^ (n >> 1))  & 0x3333333333333333 # binary: 11001100110011001100110011001100110011001100110011001100110011,  len: 62
    n = (n ^ (n >> 2))  & 0x0f0f0f0f0f0f0f0f # binary: 111100001111000011110000111100001111000011110000111100001111,    len: 60
    n = (n ^ (n >> 4))  & 0x00ff00ff00ff00ff # binary: 11111111000000001111111100000000111111110000000011111111,        len: 56
    n = (n ^ (n >> 8))  & 0x0000ffff0000ffff # binary: 1111111111111111000000001111111111111111,                        len: 40
    n = (n ^ (n >> 16)) & 0x00000000ffffffff # binary: 11111111111111111111111111111111,                                len: 32
    return n


def __unpart1by2_64(n):
    n &= 0x1249249249249249                  # binary: 1001001001001001001001001001001001001001001001001001001001001, len: 61
    n = (n ^ (n >> 2))  & 0x10c30c30c30c30c3 # binary: 1000011000011000011000011000011000011000011000011000011000011, len: 61
    n = (n ^ (n >> 4))  & 0x100f00f00f00f00f # binary: 1000000001111000000001111000000001111000000001111000000001111, len: 61
    n = (n ^ (n >> 8))  & 0x1f0000ff0000ff   # binary: 11111000000000000000011111111000000000000000011111111,         len: 53
    n = (n ^ (n >> 16)) & 0x1f00000000ffff   # binary: 11111000000000000000000000000000000001111111111111111,         len: 53
    n = (n ^ (n >> 32)) & 0x1fffff           # binary: 111111111111111111111,                                         len: 21
    return n


if getattr(sys, 'maxint', 0) and sys.maxint <= 2 ** 31 - 1:
    __part1by1 = __part1by1_32
    __part1by2 = __part1by2_32
    __unpart1by1 = __unpart1by1_32
    __unpart1by2 = __unpart1by2_32
else:
    __part1by1 = __part1by1_64
    __part1by2 = __part1by2_64
    __unpart1by1 = __unpart1by1_64
    __unpart1by2 = __unpart1by2_64


def interleave2(*args):
    if len(args) != 2:
        raise ValueError('Usage: interleave2(x, y)')
    for arg in args:
        if not isinstance(arg, int):
            print('Usage: interleave2(x, y)')
            raise ValueError("Supplied arguments contain a non-integer!")

    return __part1by1(args[0]) | (__part1by1(args[1]) << 1)


def interleave3(*args):
    if len(args) != 3:
        raise ValueError('Usage: interleave3(x, y, z)')
    for arg in args:
        if not isinstance(arg, int):
            print('Usage: interleave3(x, y, z)')
            raise ValueError("Supplied arguments contain a non-integer!")

    return __part1by2(args[0]) | (__part1by2(args[1]) << 1) | (
        __part1by2(args[2]) << 2)


def interleave(*args):
    if len(args) < 2 or len(args) > 3:
        print('Usage: interleave(x, y, (optional) z)')
        raise ValueError(
            "You must supply two or three integers to interleave!")

    method = globals()["interleave" + str(len(args))]

    return method(*args)


def deinterleave2(n):
    if not isinstance(n, int):
        print('Usage: deinterleave2(n)')
        raise ValueError("Supplied arguments contain a non-integer!")

    return __unpart1by1(n), __unpart1by1(n >> 1)


def deinterleave3(n):
    if not isinstance(n, int):
        print('Usage: deinterleave2(n)')
        raise ValueError("Supplied arguments contain a non-integer!")

    return __unpart1by2(n), __unpart1by2(n >> 1), __unpart1by2(n >> 2)


''' Fourier transformations
'''

def fft2_center(img):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(img, dim=(-1,-2))), dim=(-1,-2))

def fftn_center(img, s=None, dim=None, norm=None):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(img)))

def ifftn_center(V, s=None, dim=None, norm=None):
    V = torch.fft.ifftshift(V)
    V = torch.fft.ifftn(V, s, dim, norm)
    V = torch.fft.ifftshift(V)
    return V

def ht2_center(img):
    f = fft2_center(img)
    return f.real-f.imag

def htn_center(img):
    f = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(img)))
    return f.real-f.imag

def iht2_center(img):
    img = fft2_center(img)
    img /= (img.shape[-1]*img.shape[-2])
    return img.real - img.imag

def ihtn_center(V, s=None, dim=None, norm=None):
    V = torch.fft.fftshift(V)
    V = torch.fft.fftn(V, s, dim, norm)
    V = torch.fft.fftshift(V)
    V /= torch.prod(torch.tensor(V.shape))
    return V.real - V.imag