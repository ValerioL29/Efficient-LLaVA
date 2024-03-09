from llava.mm_utils import get_model_name_from_path
from llava.model.utils import log
from llava.eval.run_llava_vllm import eval_model


# Define  model input
model_path = "liuhaotian/llava-v1.6-vicuna-7b" # "liuhaotian/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"    
# Define benchmark config
bench_config = {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "use_flash_attn": False, # Quadro RTX 6000 doesn't support FA2 as it's Turing arch 
    "load_4bit": False,
    "load_8bit": True,
    "batch": 10,
}


# Start Benchmark
log.info(f"Starting benchmark for model: {model_path}")
# Show benchmark config
log.info(f"Benchmark config: {bench_config}")
# Run benchmark
args = type('Args', (), bench_config)()
_ = eval_model(args)
# End Benchmark
log.info(f"Ending benchmark for model: {model_path}")
