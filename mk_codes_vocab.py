'''
get the idxs list for codes using xlm-vocab
python mk_codes_vocab.py vocab.txt out.txt

voc_path: /home/lilai.lh/data/ldc-zhen-codes/vocab.zh-en
out_path: /home/lilai.lh/data/ldc-zhen-codes/vocab.codes
'''

from src.data.dictionary import Dictionary
import os
import sys
import pickle


if __name__ == '__main__':

    voc_path = sys.argv[1]
    out_path = sys.argv[2]
    assert os.path.isfile(voc_path)
    dico = Dictionary.read_vocab(voc_path)
    codes = []
    for line in open(voc_path, "r"):
        if line.startswith("<c"):
            line = line.split(" ")[0]
            word_id = dico.index(line, no_unk=False)
            codes.append(word_id)
    f = open(out_path, "wb")
    pickle.dump(codes, f)

