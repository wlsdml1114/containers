#!/bin/bash

export PYTHONUNBUFFERED=1

# cd /FasterLivePortrait
# sh scripts/all_onnx2trt.sh

rsync -au --remove-source-files /FasterLivePortrait/ /workspace/FasterLivePortrait/
