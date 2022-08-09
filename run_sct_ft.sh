#nvidia-smi topo -m
export NGPU=2
export NCCL_SHM_DISABLE=1
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
--exp_name ldc_zhen_ft_hct \
--dump_path /dumped \
--data_path /data/paranmt \
--lgs 'en-zh' \
--mt_steps 'zh-en' \
--encoder_only false \
--emb_dim 512 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 4096 \
--batch_size 32 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 500 \
--eval_bleu true \
--stopping_criterion 'valid_zh-en_mt_bleu,10' \
--validation_metrics 'valid_zh-en_mt_bleu' \
--beam_size 5 \
--save_periodic 1 \
--accumulate_gradients 2 \
--bleu_npy '/data/paranmt/bleu.npy' \
--dbleu_npy '/data/paranmt/dbleu.npy' \
--reload_model /dumped/ldc_zhen_pt/best-ckpt.pth,/dumped/ldc_zhen_pt/best-ckpt.pth