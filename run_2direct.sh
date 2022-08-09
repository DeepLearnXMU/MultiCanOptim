#nvidia-smi topo -m
export NGPU=4
export NCCL_SHM_DISABLE=1
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
--exp_name ldc_zhen_pt \
--dump_path /dumped \
--data_path /data/ldc-zhen-codes/ \
--lgs 'en-zh' \
--mt_steps 'en-zh,zh-en' \
--encoder_only false \
--emb_dim 512 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 8192 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.001 \
--epoch_size 200000 \
--eval_bleu true \
--stopping_criterion 'valid_zh-en_mt_bleu,10' \
--validation_metrics 'valid_zh-en_mt_bleu' \
--beam_size 5