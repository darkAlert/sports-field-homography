#!/bin/bash
: '
*****************************************
Downloads data from AWS S3.

Argumetns:
  $1 - name of the target file/directory to download,
  $2 - destination directory where the model will be saved (optional, default $PWD/_inference/checkpoints),
  $3 - path to aws credentials (optional, default $HOME/.aws/credentials),

To install AWS CLI and Unzip:
  apt-get install -y awscli unzip
*****************************************
'
# Arguments by default:
DST_DIR=$PWD/_inference/checkpoints
CREDENTIALS=$HOME/.aws/credentials
S3_URI=s3://boost-cv-models/boost-court-models

# Parse input arguments:
while [ $# -gt 0 ]; do
  case "$1" in
    --file=*)
      FILE="${1#*=}"
      ;;
    --dst_dir=*)
      DST_DIR="${1#*=}"
      ;;
    --credentials=*)
      CREDENTIALS="${1#*=}"
      ;;
    --s3_uri=*)
      S3_URI="${1#*=}"
      ;;
    *)
      printf "***Error: Unknown argument $1\n"
      exit 1
  esac
  shift
done

if [ -z "${FILE+unset}" ]; then
  printf "***Error: filename not specified! \n"
  exit 1
fi

# Download:
echo Downloading $FILE...
AWS_CONFIG_FILE=CREDENTIALS aws s3 cp $S3_URI/$FILE.zip $DST_DIR/$FILE.zip

# Unzip:
yes | unzip $DST_DIR/$FILE.zip -d $DST_DIR
rm $DST_DIR/$FILE.zip

echo Data has been downloaded and unpacked to $DST_DIR/$FILE

