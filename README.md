# cosyvocie2_trt_vllm

docker build -t cosyvoice-image-vllm docker/.
docker run --gpus=all -it --name cosyvoice_test -p 7860:7860 cosyvoice-image-vllm:latest