export MASTER_PORT=29501
export PYTHONPATH=./WSI_LLaVA
export CUDA_VISIBLE_DEVICES=0
./miniconda3/envs/llava/bin/python ./WSI_LLaVA/llava/eval/model_vqa.py \
    --model-path "" \
    --image-folder  \
    --question-file .jsonl \
    --answers-file .jsonl \
    --conv-mode llava_v1 \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0.2 \
    # --top_p 0.9 \
    # --num_beams 4 

