SRC_CODES="/home/lilai.lh/data/user-driven-nmt/wmt17/bpe-codes.src"
TGT_CODES="/home/lilai.lh/data/user-driven-nmt/wmt17/bpe-codes.tgt"
DATA_ROOT="/home/lilai.lh/data/diversity-nmt"

for i in 'nist02' 'nist03' 'nist04' 'nist05' 'nist06' 'nist08' 
do
    python /home/lilai.lh/pnmt/user-driven-nmt/OpenNMT-py-cache/tools/apply_bpe.py -c $SRC_CODES < $DATA_ROOT/ldc/$i.src > $DATA_ROOT/ldc-zhen-codes/$i.src.bpe
    for j in 'ref0' 'ref1' 'ref2' 'ref3'
    do
        python /home/lilai.lh/pnmt/user-driven-nmt/OpenNMT-py-cache/tools/apply_bpe.py -c $TGT_CODES < $DATA_ROOT/ldc/$i.tok.$j > $DATA_ROOT/ldc-zhen-codes/$i.$j.bpe
    done
done