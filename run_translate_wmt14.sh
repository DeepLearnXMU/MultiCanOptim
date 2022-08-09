export NCCL_SHM_DISABLE=1
python translate_wmt14.py \
--exp_name translate_wmt14 \
--exp_id 001 \
--model_path /dumped/wmt14_ende/periodic-17.pth \
--src_lang de \
--tgt_lang en \
--src_path /data/wmt14-ende-codes/train.de-en.de.pth \
--ref_path /data/wmt14-ende-codes/train.de-en.en.pth \
--output_path /dumped/translate_wmt14 \