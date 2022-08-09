# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | \
#     python translate.py --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#

import os
import io
import sys
import argparse
import torch
import numpy as np

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp, restore_segmentation
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel
from src.evaluation.evaluator import convert_to_text
from src.data.loader import set_dico_parameters
from src.data.dataset import Dataset, ParallelDataset
from src.utils import to_cuda


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")
    parser.add_argument("--tokens_per_batch", type=int, default=6250, help="")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    return parser


def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # Load a binarized dataset.  
    def load_binarized(data_path):
        assert data_path.endswith('.pth')
        assert os.path.isfile(data_path), data_path
        logger.info("Loading data from %s ..." % data_path)
        data = torch.load(data_path)

        # process_binarized
        dico = data['dico']
        assert ((data['sentences'].dtype == np.uint16) and (len(dico) < 1 << 16) or
                (data['sentences'].dtype == np.int32) and (1 << 16 <= len(dico) < 1 << 31))
        logger.info("%i words (%i unique) in %i sentences. %i unknown words (%i unique) covering %.2f%% of the data." % (
            len(data['sentences']) - len(data['positions']),
            len(dico), len(data['positions']),
            sum(data['unk_words'].values()), len(data['unk_words']),
            100. * sum(data['unk_words'].values()) / (len(data['sentences']) - len(data['positions']))
        ))
        if (data['sentences'].dtype == np.int32) and (len(dico) < 1 << 16):
            logger.info("Less than 65536 words. Moving data from int32 to uint16 ...")
            data['sentences'] = data['sentences'].astype(np.uint16)
        
        # update dictionary parameters
        set_dico_parameters(params, data, data['dico']) # will params be updated?
        return data

    for i in ['nist02', 'nist03', 'nist04', 'nist05', 'nist06', 'nist08']:
        src_path = '/home/lilai.lh/data/ldc-zhen-codes/{}.src.bpe.pth'.format(i)
        last_lang1_txt = []
        for j in ['ref0', 'ref1', 'ref2', 'ref3']:
            
            ref_path = '/home/lilai.lh/data/ldc-zhen-codes/{}.{}.bpe.pth'.format(i, j)
            lang2_path = '/home/lilai.lh/data/ldc-zhen-codes/{}.{}.txt'.format(i, j)

            src_data = load_binarized(src_path)
            tgt_data = load_binarized(ref_path)
            
            dataset = ParallelDataset(
                src_data['sentences'], src_data['positions'],
                tgt_data['sentences'], tgt_data['positions'],
                params
            )
            iterator = dataset.get_iterator(
                            shuffle=False,
                            group_by_size=False,
                            n_sentences=-1,)

            lang1_txt = []
            lang2_txt = []
            for batch in iterator:
                '''
                if len(lang2_txt)>200:
                    break
                '''
                (x1, len1), (x2, len2) = batch

                # make src text
                lang1_txt.extend(convert_to_text(x1, len1, tgt_data['dico'], params))
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                
                # make ref text
                lang2_txt.extend(convert_to_text(x2, len2, tgt_data['dico'], params))
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]
            
            if j != 'ref0':
                print("old {}, new {}, lang2 {}".format(len(last_lang1_txt), len(lang1_txt), len(lang2_txt)))
                assert lang1_txt == last_lang1_txt
            last_lang1_txt = lang1_txt

            with open(lang2_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lang2_txt) + '\n') 
            restore_segmentation(lang2_path)



if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # translate
    with torch.no_grad():
        main(params)
