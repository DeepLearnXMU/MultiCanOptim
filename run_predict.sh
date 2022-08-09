export NCCL_SHM_DISABLE=1
python predict.py \
--exp_name predict \
--exp_id $1 \
--src_lang en \
--tgt_lang zh \
--model_path $2 \
--output_path  /data/diversity-nmt/paranmt/$3 \
--src_path /data/diversity-nmt/paranmt/$4 \
--ref_path /data/diversity-nmt/paranmt/$5