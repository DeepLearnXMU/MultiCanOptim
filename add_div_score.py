from sacrebleu import sentence_bleu
import os

DATA_DIR = "/data/diversity-nmt/paranmt/"
for file_name in ["hyp.K0.bpe.pth.predict.bleu", "hyp.K1.bpe.pth.predict.bleu", "hyp.K2.bpe.pth.predict.bleu"]:
    FILE_DIR = DATA_DIR + file_name
    OUT_DIR = DATA_DIR + file_name + ".div_bleu"
    assert os.path.isfile(FILE_DIR)
    assert not os.path.isfile(OUT_DIR)
    out_txt = []
    out2_txt = []
    with open(FILE_DIR) as f1, open(DATA_DIR+"tgt.bpe.pth.predict.bleu") as f2:
        for (l1, l2) in zip (f1, f2):
            l1 = l1.strip()
            l2 = l2.strip()
            hyp = l1.split("|||")[-1]
            ref = l2.split("|||")[-1]
            hyp_c = hyp.split(" ")[0]
            ref_c = ref.split(" ")[0]
            h = " ".join(hyp.split(" ")[1:])
            r = " ".join(ref.split(" ")[1:])
            bleu = 100.0 if hyp_c == ref_c else sentence_bleu(h,[r]).score
            newline1 = l1+"|||"+str(bleu)
            newline2 = l2+"|||0.0"
            out_txt.append(newline1)
            if len(out_txt)%100000 == 0:
                print(len(out_txt))
            if "K2" in FILE_DIR:
                out2_txt.append(newline2)
    with open(OUT_DIR, "w") as f:
        f.write('\n'.join(out_txt) + '\n')
    print("successfully saved in %s"%OUT_DIR)

with open(DATA_DIR+"tgt.bpe.pth.predict.div_bleu", "w") as f:
        f.write('\n'.join(out2_txt) + '\n')
print("successfully saved in %s"%(DATA_DIR+"tgt.bpe.pth.predict.div_bleu"))
