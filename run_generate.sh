export NCCL_SHM_DISABLE=1
python generate.py \
--exp_name generate \
--exp_id train \
--model_path $1 \
--src_lang zh \
--tgt_lang en \
--output_path /data/paranmt \
--src_path /data/ldc-zhen-codes/train.en-zh.zh.pth \
--ref_path /data/ldc-zhen-codes/train.en-zh.en.pth