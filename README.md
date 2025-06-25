# cosyvocie2_trt_vllm

docker build -t cosyvoice-image-vllm docker/.
docker run --gpus=all -it --name cosyvoice_trt_vllm_test -v C:/Users/Administrator/Desktop/cosyvocie2_trt_vllm:/workspace/local -p 7860:7860 cosyvoice-image-vllm:latest

git clone https://github.com/shivammehta25/Matcha-TTS.git