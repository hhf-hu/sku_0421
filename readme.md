# Tradamark Similar Training

## install
```
# conda env config vars set CUDA_HOME=/usr/local/cuda
conda create -n tmk python=3.12
pip install -r requirements.txt
```
## downloads models
```
# download apple/DFN5B-CLIP-ViT-H-14-378
modelscope download --model apple/DFN5B-CLIP-ViT-H-14-378  --local_dir ./models/apple-DFN5B-CLIP-ViT-H-14-378

# download google/siglip2-giant-opt-patch16-384
modelscope download --model google/siglip2-giant-opt-patch16-384  --local_dir ./models/google-siglip2-giant-opt-patch16-384
```

## run
### torchrun
```
torchrun --nproc_per_node=8 pre-clip_pe.py
```

### deepspeed
```
# conda env config vars set CUDA_HOME=/usr/local/cuda
deepspeed --num_gpus 8  deepspeed_character_clip_df5b.py
```


## fix bug 

> ValueError: Sequence length must be less than max_position_embeddings (got `sequence length`: 77 and max_position_embeddings: 0
```
vi /***/anaconda3/envs/tmk/lib/python3.12/site-packages/transformers/models/clip/modeling_clip.py
```

Add on the 237 line:
``` python
if max_position_embedding == 0:
    print("fix position embedding")
    max_position_embedding = 77
```
