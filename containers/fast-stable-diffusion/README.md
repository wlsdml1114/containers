## Fast Stable Diffusion

### General

**Note that this does not work out of the box with encrypted volumes!**

This is a RunPod packaged template for stable diffusion

Runpod does not maintain the code for this repo, we just package it so that it's easier for you to use.

If you need help with settings, etc. You can feel free to ask us, but just keep in mind that we're not experts at stable diffusion! We'll try our best to help, but the RP community or automatic/stable diffusion communities may be better at helping you :)

**Please wait until the GPU/CPU Utilization % is 0 before attempting to connect. You will likely get a 502 error before that as the pod is still getting ready to be used.**

## How to use this template

Start by connecting to jupyter lab. From there you will have the option to run the automatic1111 notebook, which will launch the UI for automatic, or you can directly train dreambooth using one of the dreambooth notebooks.

### Changing launch parameters

There is a "Start Stable-Diffusion" cell in the RNPD-A1111.ipynb notebook. You can feel free to change the launch params by changing this line `!python /workspace/sd/stable-diffusion-webui/webui.py $configf`.

### Using your own models

The best ways to get your models onto your pod is by using [runpodctl](https://github.com/runpod/runpodctl/blob/main/README.md) or by uploading them to google drive or other cloud storage and downloading them to your pod from there. You should put models that you want to use with auto in the /workspace/auto-models directory.

### Uploading to google drive

If you're done with the pod and would like to send things to google drive, you can use [this colab](https://colab.research.google.com/drive/1ot8pODgystx1D6_zvsALDSvjACBF1cj6) to do it using runpodctl. You run the runpodctl either in a web terminal (found in the pod connect menu), or in a terminal on the desktop

## Template Requirements

| Port | Type (HTTP/TCP) | Function                |
|------|-----------------|-------------------------|
| 22   | TCP             | SSH                     |
| 3001 | HTTP            | Stable Diffusion Web UI |
| 8888 | HTTP            | Jupyter Lab             |
