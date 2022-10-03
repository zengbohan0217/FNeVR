# FNeVR: Neural Volume Rendering for Face Animation

Because this article deals with project confidentiality, only part of the test code is open source  \
[paper](https://arxiv.org/abs/2209.10340)

## Environment configuration

 **The codes are based on python3.8+, CUDA version 11.0+. The specific configuration steps are as follows:**

1. Create conda environment
   
   ```shell
   conda create -n fnerv python=3.8
   conda activate fnerv
   ```

2. Install pytorch
   
   ```shell
   conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
   ```

3. Installation profile
   
   ```shell
   pip install -r requirements.txt
   ```

## Pre-trained checkpoint

Checkpoint can be found under following link: [one-drive](https://1drv.ms/u/s!AraiW_uJqO8vhW-6kP0kWUyd_K7T?e=2fKBcx).

## Image reenactment/reconstruction

To run a reenactment demo, download checkpoint and run the following command:

```shell
python demo.py  --config config/vox_256.yaml --driving_video sup-mat/driving.mp4 --source_image sup-mat/source.png --checkpoint path/to/checkpoint --mode reenactment --relative --adapt_scale
```

To run a reconstruction demo, download checkpoint and run the following command:

```shell
python demo.py  --config config/vox_256.yaml --driving_video sup-mat/driving.mp4 --checkpoint path/to/checkpoint --mode reconstruction
```

The result will be stored in `result.mp4`.
