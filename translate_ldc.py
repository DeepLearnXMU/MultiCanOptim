# For NIST dataset only
# Modified from generate.py
# Translate diverse sentences from the input stream (appended with codes at beginning) and then evaluate.
#
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.

import os
import io
import sys
import argparse
import torch
import numpy as np

from src.utils import AttrDict
from src.utils import initialize_exp, restore_segmentation
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel
from src.evaluation.evaluator import convert_to_text, convert_to_diverse_text, eval_mbleu, eval_diverse_bleu
from src.data.loader import set_dico_parameters
from src.data.dataset import Dataset, ParallelDataset
from src.utils import to_cuda
from logging import getLogger

logger = getLogger()

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
    parser.add_argument("--dataset", type=str, default="ldc-zhen-codes/", help="Dataset path")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    parser.add_argument("--k", type=int, default=3, help="Generate top k")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")

    return parser


def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    assert os.path.isdir(params.dump_path)

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]

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

    root_path = "/data/"
    data_root = root_path+"ldc"
    encoder = None
    decoder = None
    for nist in ["nist02", "nist03", "nist04", "nist05", "nist06", "nist08"]:        
        src_path = root_path + params.dataset + nist + ".src.bpe.pth"
        ref_path = root_path + params.dataset + nist + ".ref0.bpe.pth" # will not be directly used
        src_data = load_binarized(src_path)
        tgt_data = load_binarized(ref_path)

        #dataset = Dataset(data['sentences'], data['positions'], params)
        dataset = ParallelDataset(
            src_data['sentences'], src_data['positions'],
            tgt_data['sentences'], tgt_data['positions'],
            params
        )

        if encoder is None:
            # build dictionary / build encoder / build decoder / reload weights
            encoder = TransformerModel(model_params, src_data['dico'], is_encoder=True, with_output=True).cuda().eval()
            decoder = TransformerModel(model_params, tgt_data['dico'], is_encoder=False, with_output=True).cuda().eval()
            encoder.load_state_dict({k.replace('module.',''):v for k,v in reloaded['encoder'].items()})
            decoder.load_state_dict({k.replace('module.',''):v for k,v in reloaded['decoder'].items()})

        iterator = dataset.get_iterator(
                        shuffle=False,
                        group_by_size=False,
                        n_sentences=-1,)

        K = params.k
        hypothesis_ks = []
        for i in range(K):
            hypothesis_ks.append([])

        lang2_txt = []
        mean_conf = torch.zeros(K)
        for batch in iterator:
            if len(lang2_txt)>200:
                break

            (x1, len1), (x2, len2) = batch
            # if train
            # x1 = torch.cat((x1[0,:].unsqueeze(0),x1[2:,:])) #remove the codes in src
            # len1-=1

            langs1 = x1.clone().fill_(params.src_id)
            langs2 = x2.clone().fill_(params.tgt_id)

            # cuda
            x1, len1, langs1, x2, len2, langs2 = to_cuda(x1, len1, langs1, x2, len2, langs2)

            # encode source sentence
            enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)

            max_len = int(1.5 * len1.max().item() + 10)
                
            generated_ks, lengths_ks, conf_ks = decoder.generate_diverse_beam(
                K, enc1, len1, params.tgt_id, beam_size=params.beam_size,
                length_penalty=1,
                early_stopping=False,
                max_len=max_len)     
                    
            assert len(generated_ks)==K
            assert len(lengths_ks)==K

            mean_conf = (mean_conf + conf_ks.cpu()) / 2.0

            for i in range(K):
                hypothesis_ks[i].extend(convert_to_diverse_text(generated_ks[i], lengths_ks[i], tgt_data['dico'], params))
                #hypothesis_ks[i].extend(convert_to_text(generated_ks[i], lengths_ks[i], tgt_data['dico'], params))
        '''
        hyp_bin = "{0}_hyp_{1}-{2}.K0.pth".format(nist, params.src_lang, params.tgt_lang)
        hyp_bin_path = os.path.join(params.dump_path, params.exp_name, params.exp_id, hyp_bin)
        torch.save(generated_ks[0], hyp_bin_path, pickle_protocol=4)
        print("binarized file save in %s" % hyp_bin_path)
        '''
        
        # write and eval_bleu
        hyp_paths = []

        for i in range(K):
            hyp_name = "{0}_hyp_{1}-{2}.K{3}.txt".format(nist, params.src_lang, params.tgt_lang, i)
            hyp_path = os.path.join(params.dump_path, params.exp_name, params.exp_id, hyp_name)
            hyp_paths.append(hyp_path)
            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis_ks[i]) + '\n')
            restore_segmentation(hyp_path)

        bleus = eval_mbleu(['%s/%s.tok.ref0'%(data_root, nist), '%s/%s.tok.ref1'%(data_root, nist), '%s/%s.tok.ref2'%(data_root, nist), '%s/%s.tok.ref3'%(data_root, nist)], hyp_paths)

        logger.info("%s mBLEU (max): %f\n" % (nist,  max(bleus)))
        logger.info("%s mBLEU (avg): %f\n" % (nist, sum(bleus) / len(bleus)))
        for i, b in enumerate(bleus):
            logger.info("%s top %i confidence/ BLEU: %f/ %f\n" % (nist, i+1, mean_conf[i], b))

        div_bleu = eval_diverse_bleu(hyp_paths)
        logger.info("%s div_BLEU: %f\n" % (nist, div_bleu))




if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.model_path)
    assert os.path.isdir(params.dump_path)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang

    # translate
    with torch.no_grad():
        main(params)
