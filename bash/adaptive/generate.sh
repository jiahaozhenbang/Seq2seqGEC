CUDA_DEVICE=3
BEAM=12
N_BEST=1
FAIRSEQ_DIR=src/fairseq-0.10.2/fairseq_cli
MODEL_DIR=outputs/adaptive/stage2plus_wo_tn_constant3_mean_0411_wo_smooth_wo_smooth_seed_2023
CP_NAME=checkpoint3

PROCESSED_DIR=preprocess/wi_locness

OUTPUT_DIR=$MODEL_DIR/results/${CP_NAME}

CoNLL14_INPUT_FILE=data/conll14_test/test.src
BEA19_INPUT_FILE=data/bea19_test/test.src
BEA19_DEV_FILE=data/bea19_dev/valid.src

mkdir -p $OUTPUT_DIR
cp $CoNLL14_INPUT_FILE $OUTPUT_DIR/CoNLL14.src
cp $BEA19_INPUT_FILE $OUTPUT_DIR/BEA19.src
cp $BEA19_DEV_FILE $OUTPUT_DIR/BEA19.dev.src


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
    < $OUTPUT_DIR/BEA19.src

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

cat $OUTPUT_DIR/BEA19.out.nbest | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 > $OUTPUT_DIR/BEA19.out
sed -i '$d' $OUTPUT_DIR/BEA19.out

zip -j $OUTPUT_DIR/bea19_without_postprocess.zip $OUTPUT_DIR/BEA19.out

# echo "Generating BEA19-dev..."
# SECONDS=0
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u ${FAIRSEQ_DIR}/interactive.py $PROCESSED_DIR/bin \
#     --task translation \
#     --path ${MODEL_DIR}/${CP_NAME}.pt \
#     --beam ${BEAM} \
#     --nbest ${N_BEST} \
#     -s src \
#     -t tgt \
#     --bpe gpt2 \
#     --buffer-size 5000 \
#     --batch-size 256 \
#     --num-workers 12 \
#     --log-format tqdm \
#     --remove-bpe \
#     --fp16 \
#     --output_file $OUTPUT_DIR/BEA19.dev.out.nbest \
#     < $OUTPUT_DIR/BEA19.dev.src

# echo "Generating Finish!"
# duration=$SECONDS
# echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

# cat $OUTPUT_DIR/BEA19.dev.out.nbest | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 > $OUTPUT_DIR/BEA19.dev.out
# sed -i '$d' $OUTPUT_DIR/BEA19.dev.out

# reference_m2=/home/ljh/GEC/Data/wi+locness/m2/ABCN.dev.gold.bea19.m2

# corrected_m2=$OUTPUT_DIR/BEA19.dev_without_postprocess.m2
# errant_parallel -orig $BEA19_DEV_FILE -cor $OUTPUT_DIR/BEA19.dev.out -out $corrected_m2

# errant_compare -hyp $corrected_m2 -ref $reference_m2

echo "Generating CoNLL14..."
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
    --output_file $OUTPUT_DIR/CoNLL14.out.nbest \
    < $OUTPUT_DIR/CoNLL14.src

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

cat $OUTPUT_DIR/CoNLL14.out.nbest | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 > $OUTPUT_DIR/CoNLL14.out
sed -i '$d' $OUTPUT_DIR/CoNLL14.out
python utils/post_process_english.py $OUTPUT_DIR/CoNLL14.src $OUTPUT_DIR/CoNLL14.out $OUTPUT_DIR/CoNLL14.out.post_processed

python /home/ljh/GEC/m2scorer/python3/m2scorer/scripts/m2scorer.py $OUTPUT_DIR/CoNLL14.out.post_processed  /home/ljh/GEC/Data/conll14st-test-data/noalt/official-2014.combined.m2
python /home/ljh/GEC/m2scorer/python3/m2scorer/scripts/m2scorer.py $OUTPUT_DIR/CoNLL14.out  /home/ljh/GEC/Data/conll14st-test-data/noalt/official-2014.combined.m2

# python /home/ljh/GEC/m2scorer/python3/m2scorer/scripts/m2scorer.py output/baseline/selfdata_2022/stage1/results/CoNLL14.out.post_processed  /home/ljh/GEC/Data/conll14st-test-data/noalt/official-2014.combined.m2
