#!/bin/bash
: '
*****************************************
Downloads the specified video from api.boostsport.ai.

Argumetns:
  $1 - name of the target video to download,
  $2 - destination directory where the video will be saved (optional, default $PWD/_inference/video).

To install wget:
  apt-get install wget
*****************************************
'
# Arguments by default:
DST_DIR=$PWD/_inference/data
URI=https://api.boostsport.ai/media/processing_videos

# Parse input arguments:
while [ $# -gt 0 ]; do
  case "$1" in
    --video=*)
      VIDEO="${1#*=}"
      ;;
    --dst_dir=*)
      DST_DIR="${1#*=}"
      ;;
    --URI=*)
      URI="${1#*=}"
      ;;
    *)
      printf "***Error: Unknown argument $1\n"
      exit 1
  esac
  shift
done

if [ -z "${VIDEO+unset}" ]; then
  printf "***Error: video_name not specified! \n"
  exit 1
fi

# Download:
echo Downloading $VIDEO...
wget $URI/$VIDEO -O $DST_DIR/$VIDEO