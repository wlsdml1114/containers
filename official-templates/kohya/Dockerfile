# Import necessary base images
# FROM runpod/stable-diffusion:models-1.0.0 as sd-models
# FROM runpod/stable-diffusion-models:2.1 as hf-cache
# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as runtime
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as runtime

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
    apt-get install --yes --no-install-recommends build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev python3-tk && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.10-dev python3.10-venv -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Set up Python and pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN curl -sSL https://github.com/kodxana/RunPod-FilleUploader/raw/main/scripts/installer.sh -o installer.sh && \
    chmod +x installer.sh && \
    ./installer.sh

# Install necessary Python packages
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --upgrade setuptools && \
    pip install --upgrade wheel
RUN pip install --upgrade --no-cache-dir torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124
RUN pip install --upgrade --no-cache-dir jupyterlab ipywidgets jupyter-archive jupyter_contrib_nbextensions triton gdown onnxruntime-gpu==1.18.1

# Set up Jupyter Notebook
RUN pip install notebook==6.5.5
RUN jupyter contrib nbextension install --user && \
    jupyter nbextension enable --py widgetsnbextension

# Install ComfyUI and ComfyUI Manager
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    cd ComfyUI_essentials && \
    pip install -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes && \
    cd ComfyUI_FizzNodes && \
    pip install -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/rgthree/rgthree-comfy && \
    cd rgthree-comfy && \
    pip install -r requirements.txt

# Install kohya
RUN git clone --recursive https://github.com/bmaltais/kohya_ss.git 
# RUN cd kohya_ss && \
#     /kohya_ss/venv/bin/python -m pip install -r /kohya_ss/requirements.txt && \
#     /kohya_ss/venv/bin/python -m pip install -r /kohya_ss/requirements_runpod.txt

# Install LP
# RUN cd / && \
# git clone https://github.com/KwaiVGI/LivePortrait.git && \
# cd LivePortrait && \
# pip install -r requirements.txt && \
# pip install -U "huggingface_hub[cli]" && \
# huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"


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
COPY post_start.sh /post_start.sh
COPY easynegative.safetensors /ComfyUI/models/embeddings/easynegative.safetensors
COPY FastNegativeV2.pt /ComfyUI/models/embeddings/FastNegativeV2.pt
COPY ng_deepnegative_v1_75t.pt /ComfyUI/models/embeddings/ng_deepnegative_v1_75t.pt
COPY kl-f8-anime2.ckpt /ComfyUI/models/vae/kl-f8-anime2.ckpt
COPY animemixv2_v20.safetensors /ComfyUI/models/checkpoints/animemixv2_v20.safetensors
COPY control_v11p_sd15_openpose_fp16.safetensors /ComfyUI/models/controlnet/control_v11p_sd15_openpose_fp16.safetensors
RUN chmod +x /start.sh

CMD [ "/start.sh" ]