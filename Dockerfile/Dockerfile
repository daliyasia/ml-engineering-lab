# docker run -it --net=host --env QT_X11_NO_MITSHM=1 --rm  -e DISPLAY=${DISPLAY} --privileged  --cap-add=SYS_PTRACE --ipc=host --security-opt seccomp=unconfined -v /home/oussamah/OHI/code/deasyml/:/workspace/  -v /home/oussamah/OHI/data/:/data/ --gpus all   --name mlcycle mlcycle
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-venv python3-dev \
        build-essential git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# CUDA 12.6 wheels; CUDA compiler (nvcc) comes from the devel base image.
RUN python -m pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip check

RUN useradd --create-home --shell /bin/bash --no-log-init mlcycle \
    && mkdir -p /workspace /data \
    && chown mlcycle:mlcycle /workspace /data

USER mlcycle
ENV HOME=/home/mlcycle
WORKDIR /workspace
CMD ["bash"]