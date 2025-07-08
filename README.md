# cosyvocie2_trt_vllm

docker build -t cosyvoice-image-vllm docker/.
docker run --gpus=all -it --name cosyvoice_trt_vllm -v /root/cosyvoice:/workspace/local -p 7868:7868 e1f1e8556863

git clone https://github.com/shivammehta25/Matcha-TTS.git

pip install numpy==1.26.4
pip install triton==2.3.1
pip install protobuf==4.25.0