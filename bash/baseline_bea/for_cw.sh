CUDA_DEVICE=6
BEAM=12
N_BEST=1
FAIRSEQ_DIR=src/fairseq-0.10.2/fairseq_cli
MODEL_DIR=outputs/adaptive/stage2_w_tn_wo_smooth_stage1_large_bs_wo_smooth
CP_NAME=checkpoint4

PROCESSED_DIR=preprocess/wi_locness

OUTPUT_DIR=$MODEL_DIR/results/for_cw

mkdir -p $OUTPUT_DIR



echo "Generating BEA19..."
SECONDS=0
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u ${FAIRSEQ_DIR}/interactive.py $PROCESSED_DIR/bin \
    --task translation \
    --path ${MODEL_DIR}/${CP_NAME}.pt \
    --beam ${BEAM} \
    --nbest ${N_BEST} \
    -s src \
    -t tgt \
    --bpe gpt2 \
    --buffer-size 5000 \
    --batch-size 256 \
    --num-workers 12 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    --output_file $OUTPUT_DIR/BEA19.out.nbest \
    < /home/ljh/GEC/Seq2seqGEC/outputs/adaptive/stage2_w_tn_wo_smooth_stage1_large_bs_wo_smooth/results/for_cw/test.txt
echo "Generating Finish!"
