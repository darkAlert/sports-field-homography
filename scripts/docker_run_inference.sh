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
INFER_ARGS=""

# Parse input arguments:
while [ $# -gt 0 ]; do
  case "$1" in
    --host_data_dir=*)
      HOST_DATA_DIR="${1#*=}"
      ;;
    --host_models_dir=*)
      HOST_MODELS_DIR="${1#*=}"
      ;;
    *)
      INFER_ARGS+=" ${1}"
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

# Run inference inside docker container:
docker run --gpus all --rm \
  --shm-size 16G \
  -v $HOST_DATA_DIR:/boost/boost-court/_inference/data \
  -v $HOST_MODELS_DIR:/boost/boost-court/_inference/checkpoints \
  -v $HOME/.aws/credentials:/root/.aws/credentials:ro \
  $COURT_IMG ./scripts/run_inference.sh ${INFER_ARGS}