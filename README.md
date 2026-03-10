# Gemma-Optimization-Benchmark

## Post Training Quantization - ModelOpt

container used - nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5.post1

run ModelOpt PTQ for gemma-3-4b-it - ```python3 quantize_model.py```

diff hf folder and ModelOpt generated folder, copy missing files into ModelOpt generated folder

## Deploy vLLM server

container used - nvcr.io/nvidia/vllm:26.02-py3

run vllm server - ```python3 -m vllm.entrypoints.openai.api_server --model /workspace/gemma-3-4b-it-fp8/ --served-model-name gemma3-4b-it-fp8 --max-model-len 6000 --max-num-batched-tokens 5000 --kv-cache-dtype fp8 --enable-chunked-prefill --max-num-seqs 256```

## Benchmark Server using aiperf

container used - nvcr.io/nvidia/tritonserver:25.05-py3-sdk

install aiperf -  ```pip install aiperf```

[!NOTE](If facing blinker related issues during pip follow this - apt uninstall blinker and then pip install aiperf)

run benchmark - ```bash aiperf_benchmark.sh```
