MODEL_DIR_STAGE2=outputs/baseline/stage2_large_bs_origin_no_len_limit
PROCESSED_DIR_STAGE2=preprocess/stage2_en

CUDA_VISIBLE_DEVICES=-1 python bash/adaptive/tmp.py $PROCESSED_DIR_STAGE2/bin \
    --valid-subset train\
    --path $MODEL_DIR_STAGE2/checkpoint8.pt \
    --task translation \
    --max-tokens 20480 \
    -s src \
    -t tgt \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --log-format tqdm \
    --skip-invalid-size-inputs-valid-test \
    --use-quality \
    --correct_probs_path preprocess/stage2_en/train.correct_probs.npz \
    --entropy_path preprocess/stage2_en/train.entropy.npz 