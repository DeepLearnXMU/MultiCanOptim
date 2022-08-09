
# Translate sentences and predict codes from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
# 
# input: codes+en (get from generator), zh(ground truth)
# output: gt code | right flag (0/1) | bleu | zh | en
# Usage:
#     python predict.py --exp_name predict \
#     --src_lang en --tgt_lang zh \
#     --model_path trained_model.pth 
#     --output_path output \
#     --src_path xxx \
#     --ref_path xxx \
#

import os
import io
import sys
import argparse
import torch
import numpy as np
import torch.nn.functional as F

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp, restore_segmentation
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel
from src.evaluation.evaluator import convert_to_text, convert_to_diverse_text, eval_moses_bleu, eval_diverse_bleu
from src.data.loader import set_dico_parameters
from src.data.dataset import Dataset, ParallelDataset
from src.utils import to_cuda
from sacrebleu import sentence_bleu

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")
    parser.add_argument("--tokens_per_batch", type=int, default=6250, help="")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")

    # model / output / reference / src paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--output_path", type=str, default="", help="Output path")
    parser.add_argument("--src_path", type=str, default="", help="Source path (codes+en.pth)")
    parser.add_argument("--ref_path", type=str, default="", help="Reference path (zh.pth)")

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
                    group_by_size=False,
                    n_sentences=-1,)

    hypothesis = []
    hypothesis_codes = []
    gt_codes = []
    equal = [] # wheather predict codes equals the gt
    lang1_txt = []
    lang2_txt = []

    for batch in iterator:

        # generate batch
        (x1, len1), (x2, len2) = batch
        
        x1_codes = x1[1,:]
        x1 = torch.cat((x1[0,:].unsqueeze(0),x1[2:,:])) # remove the codes in src
        len1-=1

        langs1 = x1.clone().fill_(params.src_id)

        # make src text (pesudo en)
        lang1_txt.extend(convert_to_text(x1, len1, tgt_data['dico'], params))
        lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
        
        # make ref text (gt zh)
        lang2_txt.extend(convert_to_text(x2, len2, tgt_data['dico'], params))
        lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

        # cuda
        x1, len1, langs1, x2, len2, x1_codes = to_cuda(x1, len1, langs1, x2, len2, x1_codes)

        # encode source sentence
        enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        
        # predict codes
        e1 = torch.mean(enc1, dim=0).type_as(enc1)
        scores = encoder.pred_layer.get_scores(e1)      # (bs, n_words)

        # make predict mask
        mc = torch.topk(scores, 1)[1].squeeze(1).unsqueeze(0)
        pred_mask = torch.ones_like(mc).transpose(0, 1)

        # visualize codes and gt_codes
        
        codes = scores.max(1)[1]

        '''
        hypothesis_codes.extend(convert_to_text(codes.unsqueeze(0), pred_mask, src_data['dico'], params, eos=False))
        print("hypothesis_codes", hypothesis_codes)
        '''
        gt_codes.extend(convert_to_text(x1_codes.unsqueeze(0), pred_mask, src_data['dico'], params, eos=False))
        # print("gt_codes", gt_codes)

        equal.extend(codes == x1_codes)
        # print("equal", equal)
              
        # generate translation - translate / convert to text
        enc1 = enc1.transpose(0, 1)
        max_len = int(1.5 * len1.max().item() + 10)
        generated, lengths = decoder.generate_beam(
            enc1, len1, params.tgt_id, beam_size=5,
            length_penalty=1,
            early_stopping=False,
            max_len=max_len)
        hypothesis.extend(convert_to_text(generated, lengths, tgt_data['dico'], params))
        if len(hypothesis)%100000 == 0:
            print("{} sents proceed.", len(hypothesis))
    
    bleu = []
    hypothesis = [x.replace('<unk>', '<<unk>>') for x in hypothesis]
    # evaluate BLEU score
    for (hyp, ref) in zip(hypothesis, lang2_txt):
        bleu.append(sentence_bleu(hyp,[ref]).score)
    assert len(bleu)==len(equal)==len(lang2_txt)==len(lang1_txt)==len(gt_codes)

    lines = []
    for i, j, u, v, w in zip(bleu, equal, lang2_txt, lang1_txt):
        lines.append("|||".join([str(i), str(int(j)), u, w+" "+v]))
    
    # write to output file 
    output_path = params.output_path

    # export sentences to hypothesis file / restore BPE segmentation
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    # restore_segmentation(output_path)

    print("Successfully saved in %s" % output_path)



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
