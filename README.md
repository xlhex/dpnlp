# Differentially Private Representation for NLP: Formal Guarantee and An Empirical Study on Privacy and Fairness

## Descriptions
This repo contains source code and pre-processed corpora for Differentially Private Representation for NLP: Formal Guarantee and An Empirical Study on Privacy and Fairness (accepted to Findings of EMNLP 2020) ([paper](https://arxiv.org/abs/2010.01285))


## Dependencies
* python3
* pytorch>=1.4
* transformers==3.0.2
* cuda 10.0

## Usage
```shell
git clone https://github.com/xlhex/dpnlp.git
```

## Training and evaluation
```shell
export GLUE_DIR=data/
export TASK_NAME=ag # ag: AG news, bl: Blog post tp: TP-US

EPSILON=0.5 # {0.05, 0.1, 0.5, 1} refer to table 2 in our paper
LAPLACE=1 # using Laplace mechanism (1) and Gaussian mechanism (0)
SEED=1

python run_dp.py \
    --model_type dpbert \
    --model_name_or_path bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 64 \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --epsilon $EPSILON \
    --laplace $LAPLACE \
    --seed $SEED \
    --output_dir ${TASK_NAME}/
```
