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

# Install ComfyUI and ComfyUI Manager
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git && \
    cd ComfyUI-LivePortraitKJ && \
    pip install insightface && \
    pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait.git && \
    cd ComfyUI-AdvancedLivePortrait && \
    pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    cd ComfyUI_essentials && \
    pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && \
    pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/kijai/ComfyUI-MimicMotionWrapper.git && \
    cd ComfyUI-MimicMotionWrapper && \
    pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    cd ComfyUI-Frame-Interpolation && \
    pip install --no-cache-dir -r requirements-no-cupy.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes && \
    cd ComfyUI_FizzNodes && \
    pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/rgthree/rgthree-comfy && \
    cd rgthree-comfy && \
    pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/Gourieff/comfyui-reactor-node && \
    cd comfyui-reactor-node && \
    pip install --no-cache-dir -r requirements.txt
RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/TMElyralab/Comfyui-MusePose.git && \
    cd Comfyui-MusePose && \
    pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -U openmim 
RUN mim install mmengine 
RUN mim install mmcv==2.2.0 mmdet==3.2.0 mmpose==1.3.2
# RUN mim install mmcv==2.1.0
# RUN mim install mmdet==3.2.0
# RUN mim install mmpose==1.2.0
RUN cd /ComfyUI/custom_nodes/Comfyui-MusePose/pretrained_weights && \
    git clone https://huggingface.co/TMElyralab/MusePose && \
    mv ./MusePose/MusePose/* ./MusePose/
RUN cd /ComfyUI/custom_nodes/Comfyui-MusePose/pretrained_weights && \
    mkdir dwpose && \
    mkdir sd-image-variations-diffusers && \
    mkdir image_encoder && \
    mkdir sd-vae-ft-mse
RUN cd /ComfyUI/custom_nodes/Comfyui-MusePose/pretrained_weights/dwpose && \
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth && \
    wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth -O yolox_l_8x8_300e_coco.pth
RUN cd /ComfyUI/custom_nodes/Comfyui-MusePose/pretrained_weights/sd-image-variations-diffusers/ && \
    mkdir unet && \
    cd unet && \
    wget https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/unet/diffusion_pytorch_model.bin && \
    wget https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/unet/config.json
RUN cd /ComfyUI/custom_nodes/Comfyui-MusePose/pretrained_weights/image_encoder/ && \
    wget https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/pytorch_model.bin && \
    wget https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/config.json
RUN cd /ComfyUI/custom_nodes/Comfyui-MusePose/pretrained_weights/sd-vae-ft-mse/ &&\
    git clone https://huggingface.co/stabilityai/sd-vae-ft-mse && \
    mv sd-vae-ft-mse/* ./
# RUN cd /ComfyUI/models && \
#     mkdir mimicmotion && \
#     cd mimicmotion && \
#     wget https://huggingface.co/Kijai/MimicMotion_pruned/resolve/main/MimicMotionMergedUnet_1-0-fp16.safetensors -O MimicMotionMergedUnet_1-0-fp16.safetensors
# RUN pip install -U "huggingface_hub[cli]" && \
#     huggingface-cli login --token $HUGGINGFACE_TOKEN
# RUN cd ../diffusers && \
#     wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/svd_xt_1_1.safetensors -O svd_xt_1_1.safetensors
RUN cd /ComfyUI/ && \
    mkdir pysssss-workflows && \
    cp custom_nodes/ComfyUI-MimicMotionWrapper/examples/*.json pysssss-workflows/ && \
    cp custom_nodes/ComfyUI-LivePortraitKJ/examples/*.json pysssss-workflows/ && \
    cp custom_nodes/ComfyUI-AdvancedLivePortrait/sample/workflows/*.json pysssss-workflows/
RUN cd /ComfyUI/models/ && \
    mkdir insightface && \
    cd insightface && \
    wget https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx

# NGINX Proxy
COPY --from=proxy nginx.conf /etc/nginx/nginx.conf
COPY --from=proxy readme.html /usr/share/nginx/html/readme.html

# Copy the README.md
COPY README.md /usr/share/nginx/html/README.md

# Start Scripts
COPY pre_start.sh /pre_start.sh
COPY --from=scripts start.sh /start.sh
COPY VFI_Faceswap.json /ComfyUI/pysssss-workflows/VFI_Faceswap.json
COPY musepose-workflow-demo.json /ComfyUI/pysssss-workflows/musepose-workflow-demo.json
COPY __init__.py /venv/lib/python3.10/site-packages/mmdet/__init__.py
COPY animemixv2_v20.safetensors /ComfyUI/models/checkpoints/animemixv2_v20.safetensors
COPY klF8Anime2VAE_klF8Anime2VAE.safetensors /ComfyUI/models/vae/klF8Anime2VAE_klF8Anime2VAE.safetensors
COPY motionModel_v01.ckpt /ComfyUI/models/animatediff_models/motionModel_v01.ckpt
RUN chmod +x /start.sh

CMD [ "/start.sh" ]
