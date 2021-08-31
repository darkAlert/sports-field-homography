#!/bin/bash
: '
*****************************************
Runs a docker image to do inference.

© Boost Technology Inc., 2021
*****************************************
'
# Arguments by default:
HOST_DATA_DIR=/path/to/your/data
HOST_MODELS_DIR=/path/to/your/checkpoints
COURT_IMG=boost-court:1.0.0
COURT_BATCH=16
GAME=game1

# Parse input arguments:
while [ $# -gt 0 ]; do
  case "$1" in
    --data_dir=*)
      HOST_DATA_DIR="${1#*=}"
      ;;
    --models_dir=*)
      HOST_MODELS_DIR="${1#*=}"
      ;;
    --game=*)
      GAME="${1#*=}"
      ;;
    --court_img=*)
      COURT_IMG="${1#*=}"
      ;;
    --court_batch=*)
      COURT_BATCH="${1#*=}"
      ;;
    *)
      printf "***Error: Unknown argument $1\n"
      exit 1
  esac
  shift
done

if [ -z "${HOST_DATA_DIR+unset}" ]; then
  printf "***Error: HOST_DATA_DIR not specified! \n"
  exit 1
fi

if [ -z "${HOST_MODELS_DIR+unset}" ]; then
  printf "***Error: HOST_MODELS_DIR not specified! \n"
  exit 1
fi

if [ -z "${GAME+unset}" ]; then
  printf "***Error: GAME name not specified! \n"
  exit 1
fi

# Make host dirs:
mkdir -p ${HOST_DATA_DIR}
mkdir -p ${HOST_MODELS_DIR}

# Build a docker image if needed:
docker build -t $COURT_IMG .

# Run inference inside docker container:
docker run --gpus all --rm \
  --shm-size 16G \
  -v $HOST_DATA_DIR:/boost/boost-court/_inference/data \
  -v $HOST_MODELS_DIR:/boost/boost-court/_inference/checkpoints \
  -v $HOME/.aws/credentials:/root/.aws/credentials:ro \
  $COURT_IMG ./scripts/run_inference.sh --game=$GAME --batch=$COURT_BATCH
