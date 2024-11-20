#!/bin/bash

export PYTHONUNBUFFERED=1
source /venv/bin/activate

cd /LivePortrait/src/utils/dependencies/XPose/models/UniPose/ops
python setup.py build install

rsync -au --remove-source-files /LivePortrait/ /workspace/LivePortrait/

cd /workspace/LivePortrait/
python app.py --server-name "0.0.0.0" --server-port 8890 > /workspace/LP_output.log 2>&1 &

cd /workspace/LivePortrait/
python app_animals.py --server-name "0.0.0.0" --server-port 8892 > /workspace/LP_animal_output.log 2>&1 &

cd /workspace/LivePortrait/
python engui_app.py > /workspace/engui_output.log 2>&1 &