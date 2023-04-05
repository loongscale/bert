docker run --gpus '"device=1"' -p 8502:8500 -p 8503:8501 --mount type=bind,source=$(pwd)/multiModel/,target=/models/multiModel \
--mount type=bind,source=$(pwd)/models.config,target=/models/models.config \
 -t tensorflow/serving:1.14.0-gpu --per_process_gpu_memory_fraction=0.2  --model_config_file=/models/models.config&



