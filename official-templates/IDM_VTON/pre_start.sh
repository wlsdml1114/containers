#!/bin/bash

export PYTHONUNBUFFERED=1
source /venv/bin/activate
rsync -au --remove-source-files /IDM-VTON/ /workspace/IDM-VTON/

cd /workspace/IDM-VTON/
python gradio_demo/app.py > /workspace/output.log 2>&1 &