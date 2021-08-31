#!/bin/bash
: '
*****************************************
Does inference for the specified model and video.
If the model or video does not exist then, it will be downloaded.
After processing is complete, the results will be saved to AWS S3 (optional).

Argumetns:
  --game       : name (with extension) of the target video to be processed,
  --model      : name of the target model to be used to make predictions (optional),
  --models_dir : directory where the model is located (optional),
  --data_dir   : directory where the video is located and where the results will be saved (optional),
  --use_imgs   : if specified, images will be used instead of video (optional, default false)),
  --batch      : input batch size (optional, default 15),
  --to_s3      : whether to upload the results to AWS S3 (optional, default false).
*****************************************
'
# Arguments by default:
MODEL=NCAA+v7-640x360-aug_unet-resnet34-deconv-mask_ce-l1-rrmse-focal
MODELS_DIR=$PWD/_inference/checkpoints
DATA_DIR=$PWD/_inference/data
BATCH=15
TO_S3=false
USE_IMGS=false

# Parse input arguments:
while [ $# -gt 0 ]; do
  case "$1" in
    --game=*)
      GAME="${1#*=}"
      VIDEO=$GAME.mp4
      ;;
    --model=*)
      MODEL="${1#*=}"
      ;;
    --models_dir=*)
      MODELS_DIR="${1#*=}"
      ;;
    --data_dir=*)
      DATA_DIR="${1#*=}"
      ;;
    --use_imgs=*)
      USE_IMGS="${1#*=}"
      ;;
    --batch=*)
      BATCH="${1#*=}"
      ;;
    --to_s3=*)
      TO_S3="${1#*=}"
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


# Download the model or video if not exist:
if [[ ! -d "$MODELS_DIR/$MODEL" ]]
then
  bash $PWD/scripts/download_s3.sh --file=$MODEL --dst_dir=$MODELS_DIR
fi

if [ "$USE_IMGS" = false ]
then
  if [[ ! -f "$DATA_DIR/$GAME/$VIDEO" ]]
  then
    bash $PWD/scripts/download_video.sh $VIDEO $DATA_DIR/$GAME
  fi
fi


# Set arguments:
model_path=$MODELS_DIR/$MODEL/last.pth       # path to model

if [ "$USE_IMGS" = true ]
then
  # images to be used as input:
  input_path=$DATA_DIR/$GAME
  input_type=--img_dir
  dst_dir=$DATA_DIR/$GAME/preds
else
  # video to be used as input:
  input_path=$DATA_DIR/$GAME/$VIDEO
  input_type=--video_path
  dst_dir=$DATA_DIR/$GAME
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
#req_outputs=warp_mask,theta,consistency,poi,debug
req_outputs=theta,consistency       # use this if you only need homography matrix and confidence score

# Image type of the predicted mask. Can be [bin / gray / rgb]:
mask_type=gray

# File format in which the predicted mask will be saved. Can be [png / pickle]:
mask_save_format=pickle

# Image size of the predicted mask:
out_width=1280
out_height=720


# Run prediction (using video):
python3 predict.py --load ${model_path} \
                   --dst_dir ${dst_dir} \
                   --req_outputs ${req_outputs} \
                   --mask_type ${mask_type} \
                   --mask_save_format ${mask_save_format} \
                   --batchsize ${BATCH} \
                   --out_size ${out_width} ${out_height} \
                   ${input_type} ${input_path}

# Upload the results to AWS S3:
if [ "$TO_S3" = true ] ; then
  printf "Not implemented yet! \n"
  #  bash $PWD/scripts/upload_s3.sh $dst_dir
fi

printf "All done! \n"