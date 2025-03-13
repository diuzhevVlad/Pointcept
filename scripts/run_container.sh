DATASET_PATH=/home/vladislav/Documents/SemanticKITTI
IMG_TAG=pointcept/pointcept:pytorch2.0.1-cuda11.7-cudnn8-devel

docker run \
  -it \
  --rm \
  --net=host \
  --ipc=host \
  --pid=host \
  --gpus all \
  -e=DISPLAY \
  -v ${DATASET_PATH}:/workspace/Pointcept/data/semantic_kitti/ \
  -v "./$(dirname "$0")/../exp":/workspace/Pointcept/exp \
  ${IMG_TAG} \
  bash