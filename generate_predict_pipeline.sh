#!/bin/bash -e
set -e
GENERATOR_PATH="multi-candidate-optim"
DATA_PATH="data/ldc-zhen-codes"
PARANMT_PATH="data/paranmt" # generated data
CKPT="" # replace with best ckpt
BEST_BLEU=40.0 # replace with ckpt performance
BEST_DBLEU=60.0 # replace with ckpt performance

#  1. Preprocess
cd $PARANMT_PATH
for file in 'src.bpe' 'tgt.bpe' 'hyp.K0.bpe' 'hyp.K1.bpe' 'hyp.K2.bpe'
do
    python $GENERATOR_PATH/preprocess.py $DATA_PATH/vocab.zh-en $file
    echo ".pth file in $PARANMT_PATH/$file.pth"
done


# 2. Generate diverse translations
cd $GENERATOR_PATH
bash run_generate.sh $CKPT


# 3. Back-translate and predict BLEU score (reconstruct-BLEU)
for file in 'tgt.bpe.pth' 'hyp.K0.bpe.pth' 'hyp.K1.bpe.pth' 'hyp.K2.bpe.pth'
do 
    echo "----------------submit new task----------------"
    bash run_predict.sh $file $CKPT $file.predict.bleu $file src.bpe.pth
    sleep 10
done


# 4. Add divBLEU to the last col
python add_div_score.py
cat *.predict.bleu.div_bleu > all.predict


# 5. Filter data based on BLEU and reconstruct-BLEU. 
# The following mySQL queries are runned on Open Data Processing Service, which can also be transferred to local database. 
cd $PARANMT_PATH
odpscmd # start odps
CREATE TABLE IF NOT EXISTS odps.ldc_all_predict(bleu double, equal boolean, zh string, en string, dbleu double);
tunnel upload -fd '|||' -s true all.predict odps.ldc_all_predict;
CREATE TABLE IF NOT EXISTS odps.ldc_uniq_predict as select distinct * from odps.ldc_all_predict;

CREATE TABLE odps.ldc_paranmt as
select zh, en, bleu, dbleu from odps.ldc_uniq_predict
where
(dbleu<$BEST_DBLEU and bleu>$BEST_BLEU);

tunnel download -fd '|||' odps.ldc_paranmt odps.ldc_paranmt;
QUIT  # end odps


# 6. extract data from odps table for finetuning
awk -F "\\\\|\\\\|\\\\|" '{print $1}' odps.ldc_paranmt > ldc_paranmt.zh
awk -F "\\\\|\\\\|\\\\|" '{print $2}' odps.ldc_paranmt > ldc_paranmt.en
awk -F "\\\\|\\\\|\\\\|" '{print $3}' odps.ldc_paranmt > ldc_paranmt.bleu
awk -F "\\\\|\\\\|\\\\|" '{print $4}' odps.ldc_paranmt > ldc_paranmt.dbleu
python txt2npy.py ldc_paranmt.bleu bleu.npy
python txt2npy.py ldc_paranmt.dbleu dbleu.npy

for file in 'ldc_paranmt.zh' 'ldc_paranmt.en'
do
    python $GENERATOR_PATH/preprocess.py $DATA_PATH/vocab.zh-en $file
    echo ".pth file in $PARANMT_PATH/$file.pth"
done
cp ldc_paranmt.zh.pth train.en-zh.zh.pth
cp ldc_paranmt.en.pth train.en-zh.en.pth
cp $DATA_PATH/valid.en-zh.*.pth .
cp $DATA_PATH/test.en-zh.*.pth .