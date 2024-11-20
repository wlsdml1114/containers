#!/bin/bash

export PYTHONUNBUFFERED=1
source /venv/bin/activate
rsync -au --remove-source-files /ComfyUI/ /workspace/ComfyUI/

cd /workspace/ComfyUI/
python main.py --listen --port 3000 > /workspace/engui_output.log 2>&1 &