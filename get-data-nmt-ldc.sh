# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e


#
# Data preprocessing configuration
#
CODES=60000     # number of BPE codes
N_THREADS=16    # number of threads in data preprocessing


#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --src)
    SRC="$2"; shift 2;;
  --tgt)
    TGT="$2"; shift 2;;
  --reload_codes)
    RELOAD_CODES="$2"; shift 2;;
  --reload_vocab)
    RELOAD_VOCAB="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"


#
# Check parameters
#
if [ "$RELOAD_CODES" != "" ] && [ ! -f "$RELOAD_CODES" ]; then echo "cannot locate BPE codes"; exit; fi
if [ "$RELOAD_VOCAB" != "" ] && [ ! -f "$RELOAD_VOCAB" ]; then echo "cannot locate vocabulary"; exit; fi
if [ "$RELOAD_CODES" == "" -a "$RELOAD_VOCAB" != "" -o "$RELOAD_CODES" != "" -a "$RELOAD_VOCAB" == "" ]; then echo "BPE codes should be provided if and only if vocabulary is also provided"; exit; fi


#
# Initialize tools and data paths
#

# main paths
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
PROC_PATH=$PWD/data/ldc-zhen-codes

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $PROC_PATH

# fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# Sennrich's WMT16 scripts for Romanian preprocessing
WMT16_SCRIPTS=$TOOLS_PATH/wmt16-scripts
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py

# BPE / vocab files
#BPE_CODES=$PROC_PATH/codes
SRC_VOCAB=$PROC_PATH/vocab.zh
TGT_VOCAB=$PROC_PATH/vocab.en
FULL_VOCAB=$PROC_PATH/vocab.en-zh

# train / valid / test parallel BPE data
PARA_SRC_TRAIN_BPE=$PROC_PATH/train.zh.bpe
PARA_TGT_TRAIN_BPE=$PROC_PATH/train.en.bpe
PARA_SRC_VALID_BPE=$PROC_PATH/nist02.src.bpe


# install tools
./install-tools.sh

# extract source and target vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $PARA_SRC_TRAIN_BPE > $SRC_VOCAB
  $FASTBPE getvocab $PARA_TGT_TRAIN_BPE > $TGT_VOCAB
fi
echo "zh vocab in: $SRC_VOCAB"
echo "en vocab in: $TGT_VOCAB"

# reload full vocabulary
cd $MAIN_PATH
if [ ! -f "$FULL_VOCAB" ] && [ -f "$RELOAD_VOCAB" ]; then
  echo "Reloading vocabulary from $RELOAD_VOCAB ..."
  cp $RELOAD_VOCAB $FULL_VOCAB
fi

# extract full vocabulary
if ! [[ -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $PARA_SRC_TRAIN_BPE $PARA_TGT_TRAIN_BPE > $FULL_VOCAB
fi
echo "Full vocab in: $FULL_VOCAB"

# binarize data
if ! [[ -f "$PARA_SRC_TRAIN_BPE.pth" ]]; then
  echo "Binarizing zh data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_TRAIN_BPE
fi
if ! [[ -f "$PARA_TGT_TRAIN_BPE.pth" ]]; then
  echo "Binarizing en data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_TRAIN_BPE
fi
echo "zh binarized data in: $PARA_SRC_TRAIN_BPE.pth"
echo "en binarized data in: $PARA_TGT_TRAIN_BPE.pth"

echo "Binarizing valid/test data..."
$MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_VALID_BPE

for j in 'ref0' 'ref1' 'ref2' 'ref3'
do
    PARA_TGT_VALID_BPE=$PROC_PATH/nist02.$j.bpe
    $MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_VALID_BPE
done

for i in 'nist03' 'nist04' 'nist05' 'nist06' 'nist08' 
do
    PARA_SRC_TEST_BPE=$PROC_PATH/$i.src.bpe
    $MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_SRC_TEST_BPE
    for j in 'ref0' 'ref1' 'ref2' 'ref3'
    do
        PARA_TGT_TEST_BPE=$PROC_PATH/$i.$j.bpe
        $MAIN_PATH/preprocess.py $FULL_VOCAB $PARA_TGT_TEST_BPE
    done
done

#
# Summary
#
echo ""
echo "===== Data summary"
echo "Parallel training data:"
echo "    zh: $PARA_SRC_TRAIN_BPE.pth"
echo "    en: $PARA_TGT_TRAIN_BPE.pth"
echo "Parallel validation data:"
echo "    zh: $PARA_SRC_VALID_BPE.pth"
echo "    en: $PARA_TGT_VALID_BPE.pth"
echo "Parallel test data:"
echo "    zh: $PARA_SRC_TEST_BPE.pth"
echo "    en: $PARA_TGT_TEST_BPE.pth"
echo ""
