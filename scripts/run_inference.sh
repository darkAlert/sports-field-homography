#!/bin/bash
: '
*****************************************
Does inference for the specified model and video.
If the model or video does not exist then, it will be downloaded.
After processing is complete, the results will be saved to AWS S3 (optional).

Arguments:
  --game       : name (without extension) of the target video to be processed,
  --model      : name of the target model to be used to make predictions (optional),
  --data_dir   : directory where the video is located (optional),
  --dst_dir    : directory where the results will be saved (optional),
  --use_imgs   : if specified, images will be used instead of video (optional, default false)),
  --batch      : input batch size (optional, default 15).
*****************************************
'

# Arguments by default:
MODEL=ncaav8-640x360-aug_unet-resnet34-deconv-img+mask_ce-l1-rrmse-focal_pre
DATA_DIR=$PWD/_inference/data
DST_DIR=$PWD/_inference/results
BATCH=15
USE_IMGS=false
VIZUALIZE=false

# Parse input arguments:
while [ $# -gt 0 ]; do
  case "$1" in
    --game=*)
      GAME="${1#*=}"
      ;;
    --model=*)
      MODEL="${1#*=}"
      ;;
    --data_dir=*)
      DATA_DIR="${1#*=}"
      ;;
    --dst_dir=*)
      DST_DIR="${1#*=}"
      ;;
    --use_imgs=*)
      USE_IMGS="${1#*=}"
      ;;
    --batch=*)
      BATCH="${1#*=}"
      ;;
    --viz*)
      VIZUALIZE=true
      ;;
    *)
      printf "***Error: Unknown argument $1\n"
      exit 1
  esac
  shift
done

if [ -z "${GAME+unset}" ]; then
  printf "***Error: game name not specified! \n"
  exit 1
fi

if [ -z "${MODEL+unset}" ]; then
  printf "***Error: model name not specified! \n"
  exit 1
fi

if [ -z "${DATA_DIR+unset}" ]; then
  printf "***Error: data directory not specified! \n"
  exit 1
fi


# Set arguments to run predict.py:
model_path=$PWD/assets/pretrained/$MODEL/last.pth       # path to model

# Input path:
if [ "$USE_IMGS" = true ]
then
  # images to be used as input:
  input_type=--img_dir
  input_path=$DATA_DIR/$GAME/frames
else
  # video to be used as input:
  input_type=--video_path
  input_path=$DATA_DIR/$GAME/$GAME.mp4
fi

# Output path:
if [ -z "${DST_DIR+unset}" ]; then
  dst_game_dir=$DATA_DIR/$GAME
else
  dst_game_dir=$DST_DIR/$GAME
fi


: '
In req_outputs, the following output keys can be used (comma separated):
  theta       : outputs the predicted homography matrix
  segm_mask   : outputs the predicted segmentation mask of the court
  warp_mask   : outputs the court mask obtained by warping using predicted homography
  poi         : outputs points of interest for the court obtained by warping using predicted homography
  consistency : outputs a consistency score that shows how accurately homography has been predicted
  debug       : outputs additional debug information
'
# Visualize (for debugging) or not:
if [ "$VIZUALIZE" = true ]; then
  req_outputs=theta,consistency,poi,debug
else
  req_outputs=theta,consistency       # use this if you only need homography matrix and confidence score
fi

# Image type of the predicted mask. Can be [bin / gray / rgb]:
mask_type=gray

# File format in which the predicted mask will be saved. Can be [png / pickle]:
mask_save_format=pickle

# Image size of the predicted mask:
out_width=1280
out_height=720


# Run inference:
python3 predict.py --load ${model_path} \
                   --dst_dir ${dst_game_dir} \
                   --req_outputs ${req_outputs} \
                   --mask_type ${mask_type} \
                   --mask_save_format ${mask_save_format} \
                   --batchsize ${BATCH} \
                   --out_size ${out_width} ${out_height} \
                   ${input_type} ${input_path}

if [[ "$?" -ne 0 ]]; then
  printf "An error occurred during inference \n"
  exit 1
fi
printf "Results saved to $dst_game_dir \n"
