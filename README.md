# Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery, MICCAI 2022.

Implementation for MICCAI 2022 paper **[Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery](https://github.com/med-air/EndoNeRF)** by Yuehao Wang, Yonghao Long, Siu Hin Fan, and [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/).

A NeRF-based framework for Stereo Endoscopic Surgery Scene Reconstruction (EndoNeRF).

**[\[Paper\]]() [\[Website\]]() [\[Sample Dataset\]]()** (coming soon)

## Demo
https://user-images.githubusercontent.com/6317569/173825509-41513eb4-3496-4b8b-814b-73f7e960d31a.mp4


## Setup

We recommend using Miniconda to set up an environment:

```bash
cd EndoNeRF
conda create -n endonerf python=3.6
conda activate endonerf
pip install -r requirements.txt
cd torchsearchsorted
pip install .
cd ..
```

We managed to test our code on Ubuntu 18.04 with Python 3.6 and CUDA 10.2.

## Dataset

To test our method on your own data, prepare a data directory organized in the following structure:

```
+ data1
    |
    |+ depth/           # depth maps
    |+ masks/           # binary tool masks
    |+ images/          # rgb images
    |+ pose_bounds.npy  # camera poses & intrinsics in LLFF format
```

In our experiments, stereo depth maps are obtained by [STTR-Light](https://github.com/mli0603/stereo-transformer) and tool masks are extracted manually. Alternatively, you can use segmentation networks, e.g., [MF-TAPNet](https://github.com/YuemingJin/MF-TAPNet), to extract tool masks. The `pose_bounds.npy` file saves camera poses and intrinsics in [LLFF format](https://github.com/Fyusion/LLFF#using-your-own-poses-without-running-colmap). In our single-viewpoint setting, we set all camera poses to identity matrices to avoid interference of ill-calibrated poses.

## Training

Type the command below to train the model:

```bash
export CUDA_VISIBLE_DEVICES=0   # Specify GPU id
python run_endonerf.py --config configs/{your_config_file}.txt
```

We put an example of the config file in `configs/example.txt`. The log files and output will be saved to `logs/{expname}`, where `expname` is specified in the config file.

## Reconstruction

After training, type the command below to reconstruct point clouds from the optimized model:

```bash
python endo_pc_reconstruction.py --config_file configs/{your_config_file}.txt --n_frames {num_of_frames} --depth_smoother --depth_smoother_d 28
```

The reconstructed point clouds will be saved to `logs/{expname}/reconstructed_pcds_{epoch}`. For more options of this reconstruction script, type `python endo_pc_reconstruction.py -h`.

We also build a visualizer to play point cloud animations. To display reconstructed point clouds, type the command as follows.

```bash
python vis_pc.py --pc_dir logs/{expname}/reconstructed_pcds_{epoch}
```

Type `python vis_pc.py -h` for more options of the visualizer.

## Acknowledgement

- Our code is based on [D-NeRF](https://github.com/albertpumarola/D-NeRF) and [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch).
- Evaluation code is borrowed from [this repo](https://github.com/peihaowang/nerf-pytorch) by [@peihaowang](https://github.com/peihaowang/).
