# Environment setup

Run the commands from the alignment module directory on a Linux machine with an NVIDIA GPU, CUDA-compatible PyTorch, Conda, Git, and FFmpeg (including ffprobe). Both environment exports record Python 3.10.19. The full YAML and pip exports in this directory are reference snapshots of the original Linux environments.

## Pi3 reconstruction and alignment

```bash
mkdir -p external
git clone https://github.com/yyfz/Pi3.git external/Pi3
git -C external/Pi3 checkout a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99
conda create -n pi3 python=3.10 pip ffmpeg -c conda-forge -y
conda activate pi3
python -m pip install -r external/Pi3/requirements.txt -e external/Pi3
python -m pip install 'opencv-python==4.11.0.86' 'trimesh==4.9.0' 'tqdm==4.67.1' 'PyYAML==6.0.3' scipy
```

## Grounded SAM 2 segmentation

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git external/Grounded-SAM-2
git -C external/Grounded-SAM-2 checkout b7a9c29f196edff0eb54dbe14588d7ae5e3dde28
conda create -n gd-sam2 python=3.10 pip -y
conda activate gd-sam2
python -m pip install 'torch==2.4.0' 'torchvision==0.19.0'
python -m pip install --no-build-isolation -e external/Grounded-SAM-2
python -m pip install 'transformers==4.57.1' 'pycocotools==2.0.10' 'opencv-python==4.12.0.88' PyYAML
(cd external/Grounded-SAM-2/checkpoints && bash download_ckpts.sh)
conda activate pi3
```

The default SAM 2 checkpoint is `external/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt`. Pi3 and Grounding DINO load their weights from Hugging Face on first use. See the [pipeline instructions](../README.md) for the scene layout and commands.
