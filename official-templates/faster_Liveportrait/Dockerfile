# Import necessary base images
# FROM runpod/stable-diffusion:models-1.0.0 as sd-models
# FROM runpod/stable-diffusion-models:2.1 as hf-cache
# FROM nvidia/cuda:12.1.0-base-ubuntu22.04 as runtime
FROM nvidia/cuda:12.2.0-base-ubuntu20.04 as runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set working directory and environment variables
ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

# Set up system
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common openssh-server nginx rsync ffmpeg && \
    apt-get install --yes --no-install-recommends build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.10-dev python3.10-venv python3-libnvinfer-dev -y --no-install-recommends && \
    apt install --yes --no-install-recommends libcudnn8 libcudnn8-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0_1.0-1_amd64.deb && \
    dpkg -i nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0_1.0-1_amd64.deb && \
    cp /var/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0/nv-tensorrt-local-9A1EDFBA-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install tensorrt

RUN wget -O /tmp/cmake.sh https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6-linux-x86_64.sh && \
    sh /tmp/cmake.sh --skip-license --prefix=/usr/local && \
    ln -s /usr/local/bin/cmake /usr/bin/cmake && \
    rm /tmp/cmake.sh

RUN cmake --version

# Set up Python and pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install necessary Python packages
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --upgrade setuptools && \
    pip install --upgrade wheel
RUN pip install --upgrade --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 xformers --index-url https://download.pytorch.org/whl/cu121
RUN pip install --upgrade --no-cache-dir jupyterlab ipywidgets jupyter-archive jupyter_contrib_nbextensions triton gdown gradio opencv-python scipy ffmpeg-python omegaconf torchgeometry insightface mediapipe
RUN pip install --no-cache-dir tensorrt==9.0.1.post11.dev4 tensorrt-bindings==9.0.1.post11.dev4 tensorrt-libs==9.0.1.post11.dev4 --extra-index-url https://pypi.nvidia.com

# Set up Jupyter Notebook
RUN pip install notebook==6.5.5
RUN jupyter contrib nbextension install --user && \
    jupyter nbextension enable --py widgetsnbextension

# Cuda toolkit
RUN cd / && \
    wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run && \
    sh cuda_12.2.0_535.54.03_linux.run --silent --toolkit && \
    rm cuda_12.2.0_535.54.03_linux.run
# RUN export CUDAToolkit_DIR=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDAToolkit_ROOT=/usr/local/cuda
ENV CMAKE_PREFIX_PATH=/usr/local/cuda

# Install LP
RUN cd / && \
    git clone https://github.com/warmshao/FasterLivePortrait.git && \
    cd FasterLivePortrait && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -U "huggingface_hub[cli]" && \
    huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints
RUN pip install nvidia-cublas-cu11==11.11.3.6 nvidia-cuda-nvcc-cu12==12.6.20 nvidia-cuda-nvrtc-cu11==11.8.89 nvidia-cuda-runtime-cu11==11.8.89 nvidia-cuda-runtime-cu12==12.6.37 nvidia-cudnn-cu11==8.9.4.25 nvidia-pyindex==1.0.9
RUN cd / && \
    git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin
COPY CMakeLists.txt /grid-sample3d-trt-plugin/CMakeLists.txt
RUN cd /grid-sample3d-trt-plugin && \
    export PATH=/usr/local/cuda/bin:$PATH && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make
RUN cp /grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so /FasterLivePortrait/scripts/
RUN cp /grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so /FasterLivePortrait/src/models/
RUN cp /grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so /FasterLivePortrait/checkpoints/liveportrait_onnx/
RUN pip uninstall -y fastapi && \
    pip install --no-cache-dir fastapi==0.112.2
COPY engui_app.py /FasterLivePortrait/engui_app.py
COPY run.py /FasterLivePortrait/run.py
COPY checkpoints /FasterLivePortrait/checkpoints


# Create necessary directories and copy necessary files
# RUN set -e && mkdir -p /root/.cache/huggingface && mkdir /comfy-models
# COPY --from=hf-cache /root/.cache/huggingface /root/.cache/huggingface
# COPY --from=sd-models /SDv1-5.ckpt /comfy-models/v1-5-pruned-emaonly.ckpt
# COPY --from=sd-models /SDv2-768.ckpt /comfy-models/SDv2-768.ckpt
# RUN wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -O /comfy-models/sd_xl_base_1.0.safetensors && \
#     wget https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors -O /comfy-models/sd_xl_refiner_1.0.safetensors

# NGINX Proxy
COPY --from=proxy nginx.conf /etc/nginx/nginx.conf
COPY --from=proxy readme.html /usr/share/nginx/html/readme.html

# Copy the README.md
COPY README.md /usr/share/nginx/html/README.md

# Start Scripts
COPY pre_start.sh /pre_start.sh
COPY --from=scripts start.sh /start.sh
RUN chmod +x /start.sh

CMD [ "/start.sh" ]
