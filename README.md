# Sequential2.0: A Self Supervised Speech Translation Model Based on Deberta and Squeezeformer Using Pseudo Languages

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors:**: FelHong Liu, RongCai Zhao

Paper link:&#x20;

## Model Checkpoints

### Pre-trained Models

| Model                             | Pre-training updates | Dataset                           | Link                                                                  |
| --------------------------------- | -------------------- | --------------------------------- | --------------------------------------------------------------------- |
| Sequtial2.0 (from HuBERT-base)    | 400K + 25K           | LibriSpeech 960h                  | [Download](https://1drv.ms/u/s!Alpovoi4BA48nBJLGH1UhvRid2iT?e=VdQCzb) |
| Sequtial2.0  (from HuBERT-base)   | 400K + 100K          | LibriSpeech 960h                  | [Download](https://1drv.ms/u/s!Alpovoi4BA48nBEWvwodv4EuL4c9?e=AVRuO2) |
| Sequtial2.0  (from fat\_*en\_zh*) | 400K + 25K           | AiShell 10Kh                      | [Download](https://1drv.ms/u/s!Alpovoi4BA48nAw3MAAvrfCLL_aK?e=9NIAzU) |
| Sequtial2.0 (from fat\_*en\_zh*)  | 400K + 25K           | ted\_*en\_*zh 2kh -> AiShell 10Kh | [Download](https://1drv.ms/u/s!Alpovoi4BA48nAt27GsNejkumbVb?e=sjPO6X) |

### Fine-tuned Models

| Model                            | Pre-training updates | Finetuning split   | Link                                                                                                                     |
| -------------------------------- | -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Sequtial2.0 (from HuBERT-base)   | 400K + 25K           | LibriSpeech 10h    | [Download](https://1drv.ms/u/s!Alpovoi4BA48nA5k25pJzyftOg_C?e=wvhcRz)                                                    |
| Sequtial2.0 (from HuBERT-base)   | 400K + 100K          | LibriSpeech 100h   | [Download](https://public-dataset-model-store.awsdev.asapp.com/fwu/wav2seq/public/ft/wav2seq-hubert-base-100k-ft-10h.pt) |
| Sequtial2.0 (from fat\_*en\_zh*) | 400K + 25K           | ted\_*en\_*zh 10h  | [Download](https://1drv.ms/u/s!Alpovoi4BA48nA_h3BC4CC3mngMo?e=rNt2ph)                                                    |
| Sequtial2.0 (from fat\_*en\_zh*) | 400K + 100K          | ted\_*en\_*zh 100h | [Download](https://1drv.ms/u/s!Alpovoi4BA48nA3w4o0rs139MAHr?e=W1ptmw)                                                    |

### Pre-trained k-means Models for Psuedo Characters

| Number of Clusters | Link                                                                  |
| ------------------ | --------------------------------------------------------------------- |
| 25                 | [Download](https://1drv.ms/u/s!Alpovoi4BA48nAlhjr47-UiyiS_j?e=Oy3o2v) |
| 100                | [Download](https://1drv.ms/u/s!Alpovoi4BA48nAod4VYjFgGnmxMq?e=7F7Kpm) |
| 500                | [Download](https://1drv.ms/u/s!Alpovoi4BA48nAN7Nf9xSsCQegZn?e=RKxeMd) |

### Pre-trained BPE model for Psuedo Subwords

| Number of Clusters | Number of Subwords | Link                                                                  |
| ------------------ | ------------------ | --------------------------------------------------------------------- |
| 25                 | 1000               | [Download](https://1drv.ms/u/s!Alpovoi4BA48nAh86a6s4sO94p2d?e=zsadLb) |
| 25                 | 3000               | [Download](https://1drv.ms/u/s!Alpovoi4BA48nAc47HCqyMVtAFou?e=nerLEq) |
| 25                 | 10000              | [Download](https://1drv.ms/u/s!Alpovoi4BA48nAaJ42UrKcn1Ozjr?e=nN1ER6) |
| 25                 | 30000              | [Download](https://1drv.ms/u/s!Alpovoi4BA48nAXYCTpNFIgfj63j?e=R1D53d) |
| 100                | 3000               | [Download](https://1drv.ms/u/s!Alpovoi4BA48nARGoNP9FFBEvASk?e=nfW9NL) |
| 100                | 10000              | [Download](https://1drv.ms/u/s!Alpovoi4BA48nAKhchtBFCBf_5Kb?e=mnHo7W) |
| 100                | 30000              | [Download](https://1drv.ms/u/s!Alpovoi4BA48nAGIxXzEy1XAPlRj?e=srM62J) |
| 500                | 3000               | [Download](https://1drv.ms/u/s!Alpovoi4BA48nADinAKHZfTk3AXp?e=6o9iFT) |
| 500                | 10000              | [Download](https://1drv.ms/u/s!Alpovoi4BA48m34kYuJCnl_m4Uld?e=KXatUo) |
| 500                | 30000              | [Download](https://1drv.ms/u/s!Alpovoi4BA48m3_A8gbG-KyEac0D?e=ysr5Jh) |

## Usage

### Dependency

    torch==1.9.0+cu111
    torchaudio==0.9.0
    tqdm==4.62.3
    hydra-core==1.0.7
    omegaconf==2.0.6
    einops==0.3.0
    fire==0.4.0
    fairseq==1.0.0a0+bba000d
    paddlepaddle==2.4.1
    paddlespeech==1.4.1

### Installation

    git clone git@github.com:961241279/Sequential2.0.git
    cd wav2seq
    pip install -e .

### Download the manifests generated by Paddle

1.  Please download the files from: [manifests](https://1drv.ms/f/s!Alpovoi4BA48m315rl76xHetZ4hi?e=FrgUsH)
2.  unzipped and put these files under "data/"

### Creatining Psuedo Subword Tokens

1.  Create wav2vec style manifest files
    Please set `LIBRISPEECH_PATH` to your librispeech folder which contains three subfolders `train-clean-100`, `train-clean-360`, `train-other-500`.

```sh
#librispeech:
mkdir -p manifest/librispeech/train-960
python -m examples.wav2vec.wav2vec_manifest LIBRISPEECH_PATH  --dest manifest/librispeech/train-960 --ext flac --valid-percent 0.01 --path-must-contain train
#aishell:
python utils/aishell.py --tgt-dir=YOUR_DATASET_DIR --src-dir=manifest/aishell
#ted_en_zh:
python utils/ted_en_zh.py --tgt-dir=YOUR_DATASET_DIR --src-dir=
manifest/ted_en_zh
```

1.  Train k-means model and get cluster indices.
    Please make sure that you have download pre-trained hubert-base checkpoint at `HUBERT_PATH`.
    Notably, this step requires a GPU for feature extraction and 64GB main memory for k-means training.
    Extracting HuBERT features takes about 15 minutes, training k-means may take about an hour, dumping the cluster ids of the whole Librispeech 960h data takes more than two hours.

```sh
HUBERT_PATH="save/pretrained/hubert_base_ls960.pt"
FAT_PATH="save/pretrained/fat_en_zh.pdparams"
mkdir -p save/pretrained
if ! [ -f $HUBERT_PATH ]; then
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -O  $HUBERT_PATH
fi
if ! [ -f $FAT_PATH ]; then
    wget 
https://paddlespeech.bj.bcebos.com/s2t/ted_en_zh/st1/paddle.98.pdparams --no-check-certificate -O $FAT_PATH
fi
bash scripts/pl/extract-features.sh $HUBERT_PATH 9 2 2 500 False
bash scripts/pl/extract-features.sh $FAT_PATH 9 2 2 500 True
```

where 9, 2, 2, 500 means that we use the 9-th layer of HuBERT, kernel size 2 and stride size 2 for average pooling, and 500 custers in k-means.

1.  Training BPE model and create pseudo subword tokens.

```sh
bash scripts/pl/create-pseudo-language.sh labels/hubert_base-l9-k2s2-fp16-ls0.1/c500 30000
bash scripts/pl/create-pseudo-language.sh labels/fat-l9-k2s2-fp16-ls0.1/c500 30000
```

### Pre-training Sequntial2.0

```sh
bash scripts/sequntial2.0-pt.sh wav2seq-hubert-base-ls960
bash scripts/sequntial2.0-pt.sh wav2seq-fat-base-ls960
```

### Fine-tuning Sequntial2.0 on LibriSpeech

To fine-tune a pretrained checkpoint on librispeech with 10h data. Please use this command.

```sh
bash scripts/sequntial2.0-ft-ls.sh $pretrained_ckpt ft-ls-10h
```

where `$pretrained_ckpt` is your pretrained checkpoint.

With 100h supervised data, please use this command.

```sh
bash scripts/sequntial2.0-ft-ls.sh $pretrained_ckpt ft-ls-100h
```

Please make sure that your manifest files are stored in `manifest/librispeech`.
We provide our manifest [here](https://public-dataset-model-store.awsdev.asapp.com/fwu/wav2seq/public/manifest/librispeech-ft.tar.gz) for reproducibility. Please make sure that you change the first line of all tsv files so that the path of the data is set correctly.
We use a pretrained subword tokenizer [link](https://public-dataset-model-store.awsdev.asapp.com/fwu/wav2seq/public/tokenizer/ls_text_bpe_unigram1000.tar.gz) to convert LibriSpeech transcripts into subword tokens.
