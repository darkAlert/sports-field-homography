# Sports Field Homography with PyTorch

![chart](https://github.com/darkAlert/sports-field-homography/blob/master/assets/CourtReconstructionChart.png)

*Sports Field Homography* is designed to predict the homography of sports fields such as a basketball court or soccer/football pitch. This repository implements an end-to-end model consisting of the UNET network, the output of which is connected to the Spatial Transformer Network (STN). First, the UNET segments the court area and outputs its mask. The predicted mask is then fed to the STN, which predicts the homography that can be used to reconstruct the court geometry.

## Overview
The model works as follows:
1. A single image (video frame) is fed to the UNET input.
2. UNET is composed of encoder and decoder subnets that downsample and upsample the image, respectively, to output a mask for it.
3. In the backward step (during training), the predicted mask and the ground truth mask are used to calculate the loss. Cross Entropy / Focal loss is used as a loss function for UNET.
4. The predicted mask is blended with the input image and passed to the STN input.
5. STN uses ResNet34 as a backbone and regresses a homography matrix consisting of 9 floating point values.
6. Finally, the predicted homography is used to warp the court template and obtain the resulting court projection image.
7. In the backward step, the resulting court projection image and the ground truth mask are used to calculate the loss. Mean Squared Error / Smooth L1 is used as a loss function for STN and UNET. In addition, the reprojection error of the court points is used as an auxiliary loss function.
8. Knowing the homography, we can map any point of the input image to the coordinates of the court and vice versa.

The model is implemented with PyTorch and Kornia frameworks.


## Installation
Install using pip:
```
# Clone the repository:
git clone https://github.com/darkAlert/sports-field-homography.git
cd sports-field-homography

# Install requirements:
pip3 install virtualenv
virtualenv -p python3 courtvenv
source courtvenv/bin/activate
pip3 install -r requirements.txt
```

### Docker Image
First, clone the repository and build a docker image:
```
# Clone the repository:
git clone https://github.com/darkAlert/sports-field-homography.git
cd sports-field-homography

# Build an image:
docker build -t sports-field:1.0.0 .
```

Then, run inference inside the docker container:
```
docker run --gpus all --rm \
  --shm-size 16G \
  -v $HOST_DATA_DIR:/sports/sports-field/_inference/data \
  -v $HOST_MODELS_DIR:/sports/sports-field/_inference/checkpoints \
  -v $HOME/.aws/credentials:/root/.aws/credentials:ro \
  sports-field:1.0.0 ./scripts/run_inference.sh --game=$GAME --batch=$COURT_BATCH
```
Where 
- `$HOST_DATA_DIR` - path to the directory where the games folders are located (for example, if the target video of the game is in `/path/to/data/my_game/my_game.mp4`, then you need to specify `HOST_DATA_DIR=/path/to/data`);
- `$GAME` - the name of the target game (e.g. `GAME=my_game` without extension `.mp4`); 
- `$HOST_MODELS_DIR` - the directory that contains the trained court model (you can also specify an empty directory, then the model will be downloaded there automatically); 
- `$COURT_BATCH` - the batch size (e.g. `COURT_BATCH=18` for NVIDIA V100 GPU).

Alternatively, you can build a docker container and run the inference with a single script:
```
sh ./scripts/docker_build_and_run.sh --data_dir=/path/to/data --game=my_game --models_dir=/path/to/checkpoints --court_batch=16 
```

Use `nohup` to run the inference in the background:
```
nohup sh ./scripts/docker_build_and_run.sh --data_dir=/path/to/data --game=my_game --models_dir=/path/to/checkpoints --court_batch=16 --to_s3=false &>infer.txt&
```

After processing is complete, the results will be saved in `/path/to/data/my_game/my_game_court.json`.

In addition, you can pass the argument `--to_s3=true` to upload the results to AWS S3.

## Inference
```
# Path to model:
model_path=./checkpoints/NCAA+v6-640x360-aug_unet-resnet34-deconv-mask_ce-l1-rrmse-focal_2/CP_epoch8.pth

# Images (img_dir) or video (video_path) can be used as input data:
#img_dir=/home/ubuntu/data/frames/test_video1/
video_path=/home/ubuntu/data/test_video1/1.mp4

# Destination directory where outputs will be saved:
dst_dir=/home/ubuntu/preds/

# The following output keys can be used in req_outputs:
#   theta       : outputs the predicted homography matrix
#   segm_mask   : outputs the predicted segmentation mask of the court
#   warp_mask   : outputs the court mask obtained by warping using predicted homography
#   poi         : outputs points of interest for the court obtained by warping using predicted homography
#   consistency : outputs a consistency score that shows how accurately homography has been predicted
#   debug       : outputs additional debug information
#req_outputs=warp_mask,theta,consistency,poi,debug
req_outputs=theta,consistency     # use this if you only need homography matrix and confidence score

# Image type of the predicted mask. Can be [bin / gray / rgb]:
mask_type=gray

# File format in which the predicted mask will be saved. Can be [png / pickle]:
mask_save_format=pickle

# Image size of the predicted mask:
out_width=1280
out_height=720

# Batch size:
batchsize=15

python3 predict.py --load ${model_path} \
                     --dst_dir ${dst_dir} \
                     --req_outputs ${req_outputs} \
                     --mask_type ${mask_type} \
                     --mask_save_format ${mask_save_format} \
                     --batchsize ${batchsize} \
                     --out_size ${out_width} ${out_height} \
                     --img_dir ${img_dir}
```

Also you can do inference using bash script:
```
cp /scripts/run_predict.sh.example /scripts/run_predict.sh
# replace paths in /scripts/run_predict.sh with your own
bash /scripts/run_predict.sh
```

## Visualization:
You can visualize predictions with [`scripts/run_viz_preds.sh.example`](https://github.com/darkAlert/sports-field-homography/blob/master/scripts/run_viz_preds.sh.example)

## How to map a point from frame coordinates to court ones:
See [`utils/mapping_example.py`](https://github.com/darkAlert/sports-field-homography/blob/master/utils/mapping_example.py)
