# FastEndoNeRF-Sim

Official code for "Efficient EndoNeRF Reconstruction and Its
Application for Data-driven Surgical Simulation" by *Yuehao Wang, Bingchen Gong, Yonghao Long, Siu Hin Fan, Qi Dou*.

![Teaser](https://github.com/med-air/FastEndoNeRF-Sim/assets/6317569/617dce15-e4a6-482e-a55f-1579143f3605)


## Setup

### 1. Create a conda environment

```bash
conda create --name fastendonerf python=3.8
conda activate fastendonerf
```

### 2. PyTorch Installation

Install PyTorch based on your environment. Our test environment is Ubuntu 18.04 LTS with PyTorch 1.11.0 + CUDA 11.3. Thereby our installation command is:

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```


## Datasets

We test our method on the [EndoNeRF datasets](https://med-air.github.io/EndoNeRF/) but we replace the stereo depths with new depth maps estimated by [unimatch](https://haofeixu.github.io/unimatch/).

We provide a unimatch script (`scripts/unimatch_stereo_depth.py`) that conducts stereo depth estimation on EndoNeRF data. To use it, you need to clone the [unimatch repo](https://github.com/autonomousvision/unimatch) and download the pretrains. Then, copy the script to its root directory. Below is given a usage example of the script:

```bash
python unimatch_stereo_depth.py \
    --mask_dir /path-to-data/cutting_tissue_twice/gt_masks \
    --inference_dir_left /path-to-data/cutting_tissue_twice/images \
    --inference_dir_right /path-to-data/cutting_tissue_twice/images_right \
    --inference_size 512 640 \
    --output_path /path-to-data/cutting_tissue_twice/depth_unimatch \
    --resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth \
    --padding_factor 32 \
    --upsample_factor 4 \
    --num_scales 2 \
    --attn_type self_swin2d_cross_swin1d \
    --attn_splits_list 2 8 \
    --corr_radius_list -1 4 \
    --prop_radius_list -1 1 \
    --reg_refine \
    --num_reg_refine 3
```

After that, you can replace the original `depth/` folder with `depth_unimatch/` to enable new depth maps in the training pipeline.


## Training

Type the command below to launch the training.

```bash
python run.py --config configs/endo/{config_name}.py  --recon_pc --render_video
```

We provide an example configuration file in `configs/endo/example.py`.

The training procedure is much faster than the original EndoNeRF. Below is a comparison within the first 4min of training.


https://github.com/yuehaowang/FastEndoNeRF-Sim/assets/6317569/6d4aef1d-be79-4d67-9489-ef00a7eed94d



### 3D Reconstruction

After training, reconstructed point clouds will be saved in `logs/endo/{expname}/recon_fine_last/`. You can use Meshlab or the [EndoNeRF visualizer](https://github.com/med-air/EndoNeRF/blob/master/vis_pc.py) to display the point clouds.

### Evaluation

To numerically evaluate the rendering results, type the command below:

```bash
python eval_rgb.py --gt_dir {path-to-data}/images --mask_dir {path-to-data}/masks --img_dir ./logs/endo/{expname}/render_video_fine_last/
```

## Soft Tissue Simulation

Type the command below to extract closed meshes from a reconstructed point cloud:

```bash
# --vis flag can enable Open3D visualization
# --output_path specifies the output directory. (default: './').
python scripts/extract_closed_mesh.py --pc_filepath logs/endo/{expname}/recon_fine_last/frame_000000_pc.ply
```

The closed mesh will be saved as `frame_000000_closed_mesh.obj`.

### NVIDIA Omniverse Isaac Sim

The closed mesh asset can be imported to NVIDIA Omniverse to perform FEM simulations.
Please follow [the documentation](https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics/deformable-bodies.html) to create a deformable body in NVIDIA Omniverse.

<img style='max-width: 900px; width: 50%;' src='https://github.com/yuehaowang/FastEndoNeRF-Sim/assets/6317569/61a3f3e2-929a-4066-b2d5-508c04d6bb49' />

### Taichi MPM

To conduct MPM simulations on reconstructed soft tissues, you need to first install Taichi (legacy branch) following the instruction [here](https://github.com/taichi-dev/taichi/blob/5ab90f03ef37701506c7034c3f1955d225b39957/docs/installation.rst). Then, type the command `ti install mpm` to install [Taichi MPM](https://github.com/yuanming-hu/taichi_mpm).

We provide a closed mesh of soft tissue and a surgical instrument in `scripts/simulaton_assets/`. Copy them to `{taichi_repo_dir}/assets/mpm/`. Then, type the command below to start MPM simulation on the reconstructed soft tissue:

```bash
python scripts/taichimpm_soft_tissues_sim.py
```

The simulation results will be saved in `{taichi_repo_dir}/outputs/mpm/`. The output file extension is '.bgeo'. To visualize the simulations, you can render the '.bgeo' file in Houdini. Alternatively, you can use our simple particle renderer. Type the commands below:

```bash
cd tools/taichi_renderer/
# Convert .bgeo to .npz
python gen_particles.py \
    --bgeo_dir {taichi_repo_dir}/outputs/mpm/taichimpm_soft_tissues_sim \
    --output_path ./sim_results/
# Generate color of particles
python gen_particle_skin.py \
    --skin_file {taichi_repo_dir}/assets/mpm/soft_tissues_s2.obj \
    --particle_file ./sim_results/0001.npz \
    --out_file ./particle_skin.npy
# Run rendering script
python render_soft_tissues_s2.py \
    --sim_dir ./sim_results/ \
    --particle_skin ./particle_skin.npy \
    --out_dir ./render_results/
```

> Note: The renderer requires Taichi==0.9.1, which conflicts with the legacy version used for MPM simulation. You may setup another environment separately for the renderer.

The particle visualizations of the MPM simulation:

https://github.com/yuehaowang/FastEndoNeRF-Sim/assets/6317569/1c54e2d8-4a50-4123-bd51-cea29a7e10dc



## Acknowledgements

- Our code is based on [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO) and [TensoRF](https://github.com/apchenstu/TensoRF).
- The unimatch script is adapted from [here](https://github.com/autonomousvision/unimatch/blob/master/evaluate_stereo.py).
- The particle renderer is based on [taichi_elements](https://github.com/taichi-dev/taichi_elements).
- Evaluation code is borrowed from [this repo](https://github.com/peihaowang/nerf-pytorch) by [@peihaowang](https://github.com/peihaowang/).
