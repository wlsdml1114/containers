#!/bin/bash

export PYTHONUNBUFFERED=1
source /venv/bin/activate

rsync -au --remove-source-files /LivePortrait/ /workspace/LivePortrait/

cd /workspace/LivePortrait/
python app.py --server-name "0.0.0.0" --server-port 8890 &