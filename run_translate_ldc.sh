export NCCL_SHM_DISABLE=1
python translate_ldc.py \
--exp_name translate_ldc \
--exp_id 001 \
--model_path /dumped/ldc_zhen/periodic-5.pth \
--src_lang zh \
--tgt_lang en \