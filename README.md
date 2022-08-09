# Multi-Candidate-Optimization
The source code of "Bridging the Gap between Training and Inference: Multi-Candidate Optimization for Diverse Neural Machine Translation (Findings of NAACL 2022)".
This project is based on [XLM](https://github.com/facebookresearch/XLM), please follow the original repository to install the dependencies and preprocess data.


## Pretrain a diverse NMT model (Tree2Code)
We provide the procedure of training [Tree2Code](https://github.com/zomux/tree2code) on NIST ZH-EN as example. You can switch to different language pairs by modifying `--src_lang`, `--tgt_lang` and data file in the following scripts.

First of all, please follow the instructions of Tree2Code to export syntactic codes into dataset (**both source and target sides need to be exported**), and put the processed data in `/data/ldc-zhen-codes`. Then make vocab and binarize data:
```
bash get-data-nmt-ldc.sh
```
Next, train a diverse NMT model using:
```
bash run_2direct.sh
```
This will train a bidirectional NMT model (for back-translation), checkpoints will be saved under `--dump_path`.


## Finetune with multiple candidates
The following script will: 1) generate candidate translations of training set using the pretrained model; 2) filter candidate translations according to BLEU and dBLEU; 3) output high-quality candidate translations (*for HCT*) and their scores (*for SCT*) in `/data/paranmt`:
```
bash generate_predict_pipeline.sh # need to change global variables at the beginning
```


### Hard constrained training (HCT)
Hard constrained training optimize model  with high-quality candidate translations. Please refer to the following script:
```
bash run_hct_ft.sh # remember to change model path
```

### Soft constrained training (SCT)
Soft constrained training optimize model with BLEU and divBLEU scores of candidate translations. Please refer to the following script:
```
bash run_sct_ft.sh # remember to change model path
```

## Citation
If you would like to use this project, please cite from below:
```
@inproceedings{lin-etal-2022-bridging,
    title = "Bridging the Gap between Training and Inference: Multi-Candidate Optimization for Diverse Neural Machine Translation",
    author = "Lin, Huan  and
      Yang, Baosong  and
      Yao, Liang  and
      Liu, Dayiheng  and
      Zhang, Haibo  and
      Xie, Jun  and
      Zhang, Min  and
      Su, Jinsong",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.200",
    pages = "2622--2632"
}
```



