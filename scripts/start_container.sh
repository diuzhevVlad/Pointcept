docker run \
    --gpus all \
    -it --rm --shm-size=20g \
    -v .:/workspace/Pointcept \
    -v /media/vladislav/KINGSTON1/dataset:/data \
    pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel \
    bash