#!/bin/bash

export PYTHONUNBUFFERED=1
source /venv/bin/activate
rsync -au --remove-source-files /ComfyUI/ /workspace/ComfyUI/
rsync -au --remove-source-files /kohya_ss/ /workspace/kohya_ss/

cd /workspace/ComfyUI/
python main.py --listen --port 3000 > /workspace/comfy_output.log 2>&1 &

# /workspace/kohya_ss/venv/bin/python -m pip install --upgrade huggingface_hub==0.25.2

cd /workspace/kohya_ss/
./setup.sh -vvv --skip-space-check --runpod -u
./gui.sh --listen=0.0.0.0 --headless > /workspace/kohya_output.log 2>&1 &