# cosyvocie2_trt_vllm

docker build -t cosyvoice-image-vllm docker/.
docker run --gpus=all -it --name cosyvoice_trt_vllm1 -v /path/to/your/local/asset:/workspace/cosyvocie2_trt_vllm/asset -p 7860:7860 cosyvoice-image-vllm:latest
