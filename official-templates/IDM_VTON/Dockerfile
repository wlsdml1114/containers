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
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common openssh-server nginx rsync ffmpeg && \
    apt-get install --yes --no-install-recommends build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev git-lfs && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.10-dev python3.10-venv -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
# RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
#     apt-get update --yes && \
#     apt-get upgrade --yes && \
#     apt-get install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common openssh-server nginx rsync ffmpeg && \
#     apt-get install --yes --no-install-recommends build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev git-lfs && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get install python3.10-dev python3.10-venv -y --no-install-recommends && \
#     apt-get autoremove -y && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* && \
#     echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

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
RUN pip install --upgrade --no-cache-dir torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu118
RUN pip install --upgrade --no-cache-dir jupyterlab ipywidgets jupyter-archive jupyter_contrib_nbextensions triton gdown

# Set up Jupyter Notebook
RUN pip install notebook==6.5.5
RUN jupyter contrib nbextension install --user && \
    jupyter nbextension enable --py widgetsnbextension
RUN pip install accelerate==0.25.0 torchmetrics==1.2.1 tqdm==4.66.1 transformers==4.36.2 diffusers==0.25.0 einops==0.7.0 bitsandbytes==0.39.0 scipy==1.11.1 opencv-python gradio==4.24.0 fvcore cloudpickle omegaconf pycocotools basicsr av onnxruntime-gpu==1.16.2 fastapi==0.112.4
RUN git clone https://github.com/yisol/IDM-VTON.git && \
    cd IDM-VTON && \
    git clone https://huggingface.co/spaces/yisol/IDM-VTON && \
    mv IDM-VTON/ckpt/densepose/* ./ckpt/densepose/ && \
    mv IDM-VTON/ckpt/humanparsing/* ./ckpt/humanparsing/ && \
    mv IDM-VTON/ckpt/openpose/ckpts/* ./ckpt/openpose/ckpts/ && \
    rm -r IDM-VTON
COPY app.py /IDM-VTON/gradio_demo/app.py
# COPY download_models.py /IDM-VTON/download_models.py
# RUN python /IDM-VTON/download_models.py

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
