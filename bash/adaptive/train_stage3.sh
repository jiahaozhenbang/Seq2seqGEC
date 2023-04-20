####################
# Train Baseline
####################

SEED=43
FAIRSEQ_CLI_PATH=src/fairseq-0.10.2/fairseq_cli
MODEL_DIR_STAGE1=outputs/baseline/stage1_large_bs_wo_smooth
MODEL_DIR_STAGE2=outputs/adaptive/stage2_w_tn_wo_smooth_stage1_large_bs_wo_smooth
MODEL_DIR_STAGE3=outputs/adaptive/stage3_wo_smooth_basedon_7_stage2_w_tn_wo_smooth_stage1_large_bs_wo_smooth

PROCESSED_DIR_STAGE1=preprocess/1BW
PROCESSED_DIR_STAGE2=preprocess/stage2_en
PROCESSED_DIR_STAGE3=preprocess/wi_locness
FAIRSEQ_PATH=src/fairseq-0.10.2/fairseq
BART_PATH=/home/ljh/model/bart-large-fairseq/bart.large/model.pt  # You need to first download BART from https://huggingface.co/facebook/bart-large

mkdir -p $MODEL_DIR_STAGE3

mkdir -p $MODEL_DIR_STAGE3/src

cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE3/src

cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE3/src

cp bash/adaptive/train_stage3.sh $MODEL_DIR_STAGE3

# Transformer-base-setting stage 1

CUDA_VISIBLE_DEVICES=5 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE3/bin \
    --save-dir $MODEL_DIR_STAGE3 \
    --finetune-from-model $MODEL_DIR_STAGE2/checkpoint7.pt \
    --arch bart_large \
    --task translation \
    --max-tokens 20480 \
    --optimizer adam \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --update-freq 2 \
    --lr 1e-05 \
    --weight-decay 0.01 \
    -s src \
    -t tgt \
    --dropout 0.3 \
    --lr-scheduler polynomial_decay \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.0 \
    --max-epoch 10 \
    --patience 3 \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --fp16 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --keep-last-epochs 10 \
    --seed $SEED >${MODEL_DIR_STAGE3}/nohup.log 2>&1 &
