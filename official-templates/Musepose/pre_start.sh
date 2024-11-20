#!/bin/bash

export PYTHONUNBUFFERED=1
source /venv/bin/activate
rsync -au --remove-source-files /ComfyUI/ /workspace/ComfyUI/
ln -s /comfy-models/* /workspace/ComfyUI/models/checkpoints/

cd /workspace/ComfyUI/models/diffusers/
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/svd_xt_1_1.safetensors -O svd_xt_1_1.safetensors

cd /workspace/ComfyUI/
python main.py --listen --port 3000 > /workspace/engui_output.log 2>&1 &