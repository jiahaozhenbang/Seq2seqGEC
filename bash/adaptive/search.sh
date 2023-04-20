CUDA_DEVICE=1
BEAM=12
N_BEST=1
FAIRSEQ_DIR=src/fairseq-0.10.2/fairseq_cli
MODEL_DIR=outputs/adaptive/stage3_wo_smooth_basedon_7_stage2_w_tn_wo_smooth_stage1_large_bs_wo_smooth
for N in 1 2 3 4 5 # 6 7 8 # 9 10 # 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
do

CP_NAME=checkpoint${N}
echo ${CP_NAME}.pt

PROCESSED_DIR=preprocess/wi_locness

OUTPUT_DIR=$MODEL_DIR/results/${CP_NAME}

CoNLL14_INPUT_FILE=data/conll14_test/test.src
BEA19_INPUT_FILE=data/bea19_test/test.src
BEA19_DEV_FILE=data/bea19_dev/valid.src

mkdir -p $OUTPUT_DIR
cp $CoNLL14_INPUT_FILE $OUTPUT_DIR/CoNLL14.src
cp $BEA19_INPUT_FILE $OUTPUT_DIR/BEA19.src
cp $BEA19_DEV_FILE $OUTPUT_DIR/BEA19.dev.src


echo "Generating BEA19-dev..."
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
    --output_file $OUTPUT_DIR/BEA19.dev.out.nbest \
    < $OUTPUT_DIR/BEA19.dev.src

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

cat $OUTPUT_DIR/BEA19.dev.out.nbest | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 > $OUTPUT_DIR/BEA19.dev.out
sed -i '$d' $OUTPUT_DIR/BEA19.dev.out


reference_m2=/home/ljh/GEC/Data/wi+locness/m2/ABCN.dev.gold.bea19.m2
corrected_m2=$OUTPUT_DIR/BEA19.dev_without_postprocess.m2
errant_parallel -orig $BEA19_DEV_FILE -cor $OUTPUT_DIR/BEA19.dev.out -out $corrected_m2

errant_compare -hyp $corrected_m2 -ref $reference_m2
done

# nohup bash bash/adaptive/search.sh 2>&1 >outputs/adaptive/stage3_wo_smooth_basedon_7_stage2_w_tn_wo_smooth_stage1_large_bs_wo_smooth/tmp.log &