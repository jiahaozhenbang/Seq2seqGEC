# MODEL_DIR_STAGE2=outputs/baseline/stage2_large_bs
# PROCESSED_DIR_STAGE2=preprocess/stage2_en

# CUDA_VISIBLE_DEVICES=0 python src/generate_quality.py $PROCESSED_DIR_STAGE2/bin \
#     --valid-subset train\
#     --path $MODEL_DIR_STAGE2/checkpoint8.pt \
#     --task translation \
#     --max-tokens 20480 \
#     -s src \
#     -t tgt \
#     --criterion label_smoothed_cross_entropy \
#     --label-smoothing 0.1 \
#     --log-format tqdm \
#     --skip-invalid-size-inputs-valid-test \
#     --entropy_output_file $PROCESSED_DIR_STAGE2/train.entropy   \
#     --correct_probs_output_file $PROCESSED_DIR_STAGE2/train.correct_probs  

# MODEL_DIR_STAGE2=outputs/baseline/stage2plus_w_tn
# PROCESSED_DIR_STAGE2=preprocess/stage2_en

# CUDA_VISIBLE_DEVICES=0 python src/generate_quality.py $PROCESSED_DIR_STAGE2/bin \
#     --valid-subset train\
#     --path $MODEL_DIR_STAGE2/checkpoint2.pt \
#     --task translation \
#     --max-tokens 2048 \
#     -s src \
#     -t tgt \
#     --criterion label_smoothed_cross_entropy \
#     --label-smoothing 0.1 \
#     --log-format tqdm \
#     --skip-invalid-size-inputs-valid-test \
#     --entropy_output_file $PROCESSED_DIR_STAGE2/train0411.entropy   \
#     --correct_probs_output_file $PROCESSED_DIR_STAGE2/train0411.correct_probs  



# MODEL_DIR_STAGE2=outputs/baseline/stage3_wo_smooth_stage2_wo_smooth_stage1_large_bs_wo_smooth
# PROCESSED_DIR_STAGE2=preprocess/stage2_en

# CUDA_VISIBLE_DEVICES=2 python src/generate_quality.py $PROCESSED_DIR_STAGE2/bin \
#     --valid-subset train\
#     --path $MODEL_DIR_STAGE2/checkpoint1.pt \
#     --task translation \
#     --max-tokens 20480 \
#     -s src \
#     -t tgt \
#     --criterion label_smoothed_cross_entropy \
#     --label-smoothing 0.1 \
#     --log-format tqdm \
#     --skip-invalid-size-inputs-valid-test \
#     --entropy_output_file $PROCESSED_DIR_STAGE2/train0413.entropy   \
#     --correct_probs_output_file $PROCESSED_DIR_STAGE2/train0413.correct_probs  

# MODEL_DIR_STAGE2=outputs/baseline/stage3_wo_smooth_stage2_wo_smooth_stage1_large_bs_wo_smooth
# PROCESSED_DIR_STAGE3=preprocess/wi_locness

# CUDA_VISIBLE_DEVICES=3 python src/generate_quality.py $PROCESSED_DIR_STAGE3/bin \
#     --valid-subset train\
#     --path $MODEL_DIR_STAGE2/checkpoint1.pt \
#     --task translation \
#     --max-tokens 20480 \
#     -s src \
#     -t tgt \
#     --criterion label_smoothed_cross_entropy \
#     --label-smoothing 0.1 \
#     --log-format tqdm \
#     --skip-invalid-size-inputs-valid-test \
#     --entropy_output_file $PROCESSED_DIR_STAGE3/train0413.entropy   \
#     --correct_probs_output_file $PROCESSED_DIR_STAGE3/train0413.correct_probs  

MODEL_DIR_STAGE2=outputs/baseline/stage3_wo_smooth_stage2_w_tn_wo_smooth_stage1_large_bs_wo_smooth
PROCESSED_DIR_STAGE2=preprocess/stage2_en

CUDA_VISIBLE_DEVICES=1 python src/generate_quality.py $PROCESSED_DIR_STAGE2/bin \
    --valid-subset train\
    --path $MODEL_DIR_STAGE2/checkpoint3.pt \
    --task translation \
    --max-tokens 20480 \
    -s src \
    -t tgt \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --log-format tqdm \
    --skip-invalid-size-inputs-valid-test \
    --entropy_output_file $PROCESSED_DIR_STAGE2/train0414.entropy   \
    --correct_probs_output_file $PROCESSED_DIR_STAGE2/train0414.correct_probs  