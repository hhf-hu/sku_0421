#torchrun
torchrun --nproc_per_node=2 pre-clip_pe.py
#deepspeed
# conda env config vars set CUDA_HOME=/usr/local/cuda
deepspeed --num_gpus 8  deepspeed_character_clip_df5b.py
