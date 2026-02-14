docker run \
    --gpus all \
    -it --rm --shm-size=100g \
    -v .:/workspace/Pointcept \
    -v /data2/datasets/RSC_estim/dataset:/data \
    pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel \
    bash