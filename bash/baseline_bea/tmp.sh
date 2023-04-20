

CUDA_DEVICE=5
BEAM=12
N_BEST=1
FAIRSEQ_DIR=src/fairseq-0.10.2/fairseq_cli
MODEL_DIR=outputs/adaptive/stage2_adjust_entropy
CP_NAME=checkpoint6

PROCESSED_DIR=preprocess/wi_locness

OUTPUT_DIR=$MODEL_DIR/results/${CP_NAME}

CoNLL14_INPUT_FILE=data/conll14_test/test.src
BEA19_INPUT_FILE=data/bea19_test/test.src
BEA19_DEV_FILE=data/bea19_dev/valid.src

python utils/post_process_english.py $OUTPUT_DIR/BEA19.dev.src $OUTPUT_DIR/BEA19.dev.out $OUTPUT_DIR/BEA19.dev.out.post_processed


corrected_m2=$OUTPUT_DIR/BEA19.dev.m2
reference_m2=/home/ljh/GEC/Data/wi+locness/m2/ABCN.dev.gold.bea19.m2

errant_parallel -orig $BEA19_DEV_FILE -cor $OUTPUT_DIR/BEA19.dev.out.post_processed -out $corrected_m2

errant_compare -hyp $corrected_m2 -ref $reference_m2