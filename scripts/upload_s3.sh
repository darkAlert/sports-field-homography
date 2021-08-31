#!/bin/bash
: '
*****************************************
Uploads the specified directory to AWS S3.

Argumetns:
  --file        : source directory that will be uploaded to AWS S3,
  --pack        : whether the source directory should be Zip (optional, default true),
  --credentials : path to AWS S3 credentials (optional),
  --s3_uri      : path to aws credentials (optional, default $HOME/.aws/credentials)

To install AWS CLI and Zip:
  apt-get install -y awscli zip
*****************************************
'
# Arguments by default:
PACK=true
CREDENTIALS=$HOME/.aws/credentials
S3_URI=s3://boost-shared/boost-court-results

# Parse input arguments:
while [ $# -gt 0 ]; do
  case "$1" in
    --file=*)
      FILE="${1#*=}"
      ;;
    --pack=*)
      PACK="${1#*=}"
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
  printf "***Error: target file not specified! \n"
  exit 1
fi

# Zip (optional):
if [ "$PACK" = true ] ; then
    printf "Packing $src.zip... \n"
    cur_dir=$PWD
    cd $src
    zip -r $src.zip ./*
    cd $cur_dir
    if ! [ $src == $PWD ]; then
      rm -rf $src;
    fi
    src=$src.zip
fi

# Upload:
AWS_CONFIG_FILE=credentials aws s3 cp $src $s3_uri/
#--recursive

filename="${src##*/}"
echo Data has been uploaded to $s3_uri/$filename