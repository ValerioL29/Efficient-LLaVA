nsys profile \
    --trace=cuda,cudnn,cublas,osrt,nvtx --gpu-metrics-device=all \
    --force-overwrite=true -o llava-7b-v16-batch_1 \
python eval.py