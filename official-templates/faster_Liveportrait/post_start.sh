export PATH=/root/miniconda3/bin:$PATH
export PATH=$PATH:/usr/local/cuda/bin
export PATH=$PATH:/usr/local/cuda/include
export PATH=/opt/cmake-3.30.0-linux-x86_64/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=/opt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH

# ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.550.54.14 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
# ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.550.54.14 /usr/lib/x86_64-linux-gnu/libcuda.so.1
nvidia_version=$(ls /usr/lib/x86_64-linux-gnu | grep -E 'libnvidia-ml\.so\.[0-9]+\.[0-9]+' | sed -n 's/.*\.so\.\(.*\)/\1/p')
ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.$nvidia_version /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.$nvidia_version /usr/lib/x86_64-linux-gnu/libcuda.so.1

# cd /workspace/FasterLivePortrait/
# sh scripts/all_onnx2trt.sh
cd /workspace/FasterLivePortrait/
sh scripts/all_onnx2trt.sh
python engui_app.py > /workspace/engui_output.log 2>&1 &
python main.py > /workspace/output.log 2>&1 &

mkdir -p ~/.ssh
cd $_
chmod 700 ~/.ssh
echo $PUBLIC_KEY >> authorized_keys
chmod 700 authorized_keys
service ssh start

runpod-uploader

