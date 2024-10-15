#!/bin/bash

export PYTHONUNBUFFERED=1
source /venv/bin/activate

rsync -au --remove-source-files /DrawingSpinUp/ /workspace/DrawingSpinUp/