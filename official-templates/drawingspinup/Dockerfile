# Import necessary base images
# FROM runpod/stable-diffusion:models-1.0.0 as sd-models
# FROM runpod/stable-diffusion-models:2.1 as hf-cache
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set working directory and environment variables
ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

# Set up system
# RUN apt-get update --yes && \
#     apt-get upgrade --yes && \
#     apt install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common openssh-server nginx rsync ffmpeg && \
#     apt-get install --yes --no-install-recommends build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev unzip && \
#     apt-get install --yes --no-install-recommends libxi6 libxrender-dev libgl-dev libxfixes3 libxxf86vm-dev && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt install python3.10-dev python3.10-venv -y --no-install-recommends && \
#     apt-get autoremove -y && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* && \
#     echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    git wget curl bash libgl1 software-properties-common \
    openssh-server nginx rsync ffmpeg build-essential \
    libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev \
    git-lfs cuda-toolkit-11-8 unzip && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install python3.10-dev python3.10-venv -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Create symlink for CUDA
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so

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
RUN pip install --upgrade --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --upgrade --no-cache-dir jupyterlab ipywidgets jupyter-archive jupyter_contrib_nbextensions triton xformers gdown

# Set up Jupyter Notebook
RUN pip install notebook==6.5.5
RUN jupyter contrib nbextension install --user && \
    jupyter nbextension enable --py widgetsnbextension
# Cuda toolkit
RUN cd / && \
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run && \
    sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit && \
    rm cuda_11.8.0_520.61.05_linux.run

RUN git clone https://github.com/LordLiang/DrawingSpinUp.git && \
    cd DrawingSpinUp && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt
RUN pip install --upgrade --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 xformers --index-url https://download.pytorch.org/whl/cu118
RUN pip install --upgrade --no-cache-dir OpenEXR
# 3090
ENV TCNN_CUDA_ARCHITECTURES=86
RUN pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
RUN git clone https://github.com/cprogrammer1994/python-mesh-raycast && \
    cd python-mesh-raycast && \
    python setup.py develop
RUN cd .. && \
    wget https://download.blender.org/release/Blender3.3/blender-3.3.1-linux-x64.tar.xz && \
    tar -xvf blender-3.3.1-linux-x64.tar.xz
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    ./blender-3.3.1-linux-x64/3.3/python/bin/python3.10 get-pip.py && \
    ./blender-3.3.1-linux-x64/3.3/python/bin/python3.10 -m pip install trimesh
COPY preprocessed.zip /DrawingSpinUp/dataset/AnimatedDrawings/preprocessed.zip
RUN cd /DrawingSpinUp/dataset/AnimatedDrawings && \
    unzip preprocessed.zip && \
    cd ../..


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
