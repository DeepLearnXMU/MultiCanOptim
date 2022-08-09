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
import subprocess

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp, restore_segmentation
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel
from src.evaluation.evaluator import convert_to_text, convert_to_diverse_text, eval_moses_bleu, eval_diverse_bleu
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
    parser.add_argument("--dump_path", type=str, default="/dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")
    parser.add_argument("--tokens_per_batch", type=int, default=6250, help="")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")

    # model / output / reference / src paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--output_path", type=str, default="", help="Output path")
    parser.add_argument("--src_path", type=str, default="", help="Source path (codes+zh.pth)")
    parser.add_argument("--ref_path", type=str, default="", help="Reference path (en.pth)")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    return parser

'''
def restore_segmentation(path):
    """
    Take a file segmented with BPE and restore it to its original segmentation.
    """
    assert os.path.isfile(path)
    newpath = path.replace(".bpe", ".tok")
    restore_cmd = "sed -i -r 's/(@@ )|(@@ ?$)//g' %s > %s"
    subprocess.Popen(restore_cmd % (path, newpath), shell=True).wait()
    return newpath
'''


def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

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

    src_data = load_binarized(params.src_path)
    tgt_data = load_binarized(params.ref_path)

    #dataset = Dataset(data['sentences'], data['positions'], params)
    dataset = ParallelDataset(
        src_data['sentences'], src_data['positions'],
        tgt_data['sentences'], tgt_data['positions'],
        params
    )

    # build dictionary / build encoder / build decoder / reload weights
    encoder = TransformerModel(model_params, src_data['dico'], is_encoder=True, with_output=True).cuda().eval()
    decoder = TransformerModel(model_params, tgt_data['dico'], is_encoder=False, with_output=True).cuda().eval()
    encoder.load_state_dict({k.replace('module.',''):v for k,v in reloaded['encoder'].items()})
    decoder.load_state_dict({k.replace('module.',''):v for k,v in reloaded['decoder'].items()})
    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]
    
    iterator = dataset.get_iterator(
                    shuffle=False,
                    group_by_size=True,
                    n_sentences=-1,)

    K=3
    for batch in iterator:

        hypothesis_ks = []
        hypothesis_ks_tok = []
        for i in range(K):
            hypothesis_ks.append([])
            hypothesis_ks_tok.append([])

        lang1_txt = []
        lang2_txt = []
        lang2_tok_txt = []

        if len(lang2_txt)>200:
            break

        (x1, len1), (x2, len2) = batch

        # training data has codes in src
        x1 = torch.cat((x1[0,:].unsqueeze(0),x1[2:,:])) #remove the codes in src
        len1-=1

        langs1 = x1.clone().fill_(params.src_id)
        langs2 = x2.clone().fill_(params.tgt_id)
        
        # make src text
        # filter need <codes> en -> zh, so lang1(zh) don't need codes, we have already remove it above
        lang1_txt.extend(convert_to_text(x1, len1, src_data['dico'], params))
        lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
        
        # make ref text
        # for train data we use "convert_to_diverse_text", otherwise "convert_to_text"
        # because training data has codes in the beginning, while we do not consider it when eval BLEU
        lang2_txt.extend(convert_to_text(x2, len2, tgt_data['dico'], params)) 
        lang2_tok_txt.extend(convert_to_diverse_text(x2, len2, tgt_data['dico'], params)) 
        
        lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]
        lang2_tok_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_tok_txt]

        # cuda
        x1, len1, langs1, x2, len2, langs2 = to_cuda(x1, len1, langs1, x2, len2, langs2)

        # encode source sentence
        enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        enc1 = enc1.transpose(0, 1)

        max_len = int(1.5 * len1.max().item() + 10)
            
        generated_ks, lengths_ks = decoder.generate_diverse_beam(
            K, enc1, len1, params.tgt_id, beam_size=5,
            length_penalty=1,
            early_stopping=False,
            max_len=max_len)
                   
        assert len(generated_ks)==K
        assert len(lengths_ks)==K

        for i in range(K):
            hypothesis_ks[i].extend(convert_to_text(generated_ks[i], lengths_ks[i], tgt_data['dico'], params)) # keep the tgt codes
            hypothesis_ks_tok[i].extend(convert_to_diverse_text(generated_ks[i], lengths_ks[i], tgt_data['dico'], params)) # remove the codes to calculate BLEU
    
        lang1_path = os.path.join(params.output_path, 'src.bpe')
        with open(lang1_path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lang1_txt) + '\n') 

        lang2_path = os.path.join(params.output_path, 'tgt.bpe')
        lang2_tok_path = os.path.join(params.output_path, 'tgt.tok') # for BLEU

        with open(lang2_path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lang2_txt) + '\n') 

        with open(lang2_tok_path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lang2_tok_txt) + '\n') 
       
        for i in range(K):
            hyp_name = 'hyp.K{}.bpe'.format(i)
            hyp_path = os.path.join(params.output_path, hyp_name)
            hyp_tok_path = hyp_path.replace(".bpe", ".tok")
            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis_ks[i]) + '\n')
            with open(hyp_tok_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis_ks_tok[i]) + '\n')

    # write and eval_bleu
    best_bleu = 0.0
    hyp_tok_paths = []
    # we don't need to restore bpe in lang1 because filter needs it
    restore_segmentation(lang2_tok_path)
    
    for i in range(K):
        hyp_tok_path = os.path.join(params.output_path, 'hyp.K{}.tok'.format(i))
        hyp_tok_paths.append(hyp_tok_path)
        restore_segmentation(hyp_tok_path)

        # evaluate best BLEU and diverse BLEU
        bleu = eval_moses_bleu(lang2_tok_path, hyp_tok_path)
        if bleu > best_bleu:
            best_bleu = bleu
    sys.stderr.write("best BLEU %s %s : %f\n" % (hyp_tok_path, lang2_tok_path, best_bleu))

    div_bleu = eval_diverse_bleu(hyp_tok_paths)
    sys.stderr.write("div_BLEU %s %s : %f\n" % (hyp_tok_path, lang2_tok_path, div_bleu))



if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.model_path)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang
    assert params.output_path and not os.path.isfile(params.output_path)

    # translate
    with torch.no_grad():
        main(params)
