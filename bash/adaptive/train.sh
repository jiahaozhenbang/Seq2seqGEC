####################
# Train Baseline
####################

SEED=43
FAIRSEQ_CLI_PATH=src/fairseq-0.10.2/fairseq_cli
MODEL_DIR_STAGE1=outputs/baseline/stage1_large_bs
MODEL_DIR_STAGE2=outputs/adaptive/stage2_less_bs

PROCESSED_DIR_STAGE1=preprocess/1BW
PROCESSED_DIR_STAGE2=preprocess/stage2_en
FAIRSEQ_PATH=src/fairseq-0.10.2/fairseq
BART_PATH=/home/ljh/model/bart-large-fairseq/bart.large/model.pt  # You need to first download BART from https://huggingface.co/facebook/bart-large

mkdir -p $MODEL_DIR_STAGE2

mkdir -p $MODEL_DIR_STAGE2/src

cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE2/src

cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE2/src

cp bash/adaptive/train.sh $MODEL_DIR_STAGE2


# Transformer-base-setting stage 1

CUDA_VISIBLE_DEVICES=1 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE2/bin \
    --save-dir $MODEL_DIR_STAGE2 \
    --finetune-from-model $MODEL_DIR_STAGE1/checkpoint_best.pt \
    --arch bart_large \
    --task translation \
    --max-tokens 10240 \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --update-freq 2 \
    --lr 3e-05 \
    --warmup-updates 2000 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.3 \
    --lr-scheduler polynomial_decay \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 30 \
    --patience 3 \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 30 \
    --min_len 3 \
    --max_len 64 \
    --use-quality \
    --correct_probs_path preprocess/stage2_en/train.correct_probs.npz \
    --entropy_path preprocess/stage2_en/train.entropy.npz \
    --seed $SEED >${MODEL_DIR_STAGE2}/nohup.log 2>&1 &
