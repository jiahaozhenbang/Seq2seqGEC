# ####################
# # Preprocess 1BW
# ####################

# FAIRSEQ_DIR=src/fairseq-0.10.2/fairseq_cli
# PROCESSED_DIR=preprocess/1BW
# mkdir -p $PROCESSED_DIR

# WORKER_NUM=32

# # File path
# TRAIN_SRC_FILE=/home/ljh/GEC/gector/data/new_1bw/train_source
# TRAIN_TGT_FILE=/home/ljh/GEC/gector/data/new_1bw/train_target
# VALID_SRC_FILE=data/bea19_dev/valid.src
# VALID_TGT_FILE=data/bea19_dev/valid.tgt


# # apply bpe
# if [ ! -f $PROCESSED_DIR/train.bpe.src ]; then
#   echo "Apply BPE..."
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $TRAIN_SRC_FILE \
#             --outputs $PROCESSED_DIR/train.bpe.src \
#             --workers 32 \
#             --keep-empty;
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $TRAIN_TGT_FILE \
#             --outputs $PROCESSED_DIR/train.bpe.tgt \
#             --workers 32 \
#             --keep-empty;
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $VALID_SRC_FILE \
#             --outputs $PROCESSED_DIR/valid.bpe.src \
#             --workers 32 \
#             --keep-empty;
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $VALID_TGT_FILE \
#             --outputs $PROCESSED_DIR/valid.bpe.tgt \
#             --workers 32 \
#             --keep-empty;
# fi



# # fairseq preprocess
# mkdir -p $PROCESSED_DIR
# cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
# cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
# cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
# cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt


# echo "Preprocess..."
# rm -rf $PROCESSED_DIR/bin
# mkdir -p $PROCESSED_DIR/bin

# python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
#        --task translation \
#        --trainpref $PROCESSED_DIR/train.bpe \
#        --validpref $PROCESSED_DIR/valid.bpe \
#        --destdir $PROCESSED_DIR/bin \
#        --workers $WORKER_NUM \
#        --srcdict /home/ljh/model/bart-large-fairseq/dict.txt \
#        --tgtdict /home/ljh/model/bart-large-fairseq/dict.txt 

# echo "Finished!"


# ################################
# # Preprocess stage2 (clang8 + FCE + NUCLE + WI)
# ################################


# FAIRSEQ_DIR=src/fairseq-0.10.2/fairseq_cli
# PROCESSED_DIR=preprocess/stage2_en
# mkdir -p $PROCESSED_DIR

# WORKER_NUM=32

# # File path

# TRAIN_SRC_FILE=preprocess/stage2_en/train.src
# TRAIN_TGT_FILE=preprocess/stage2_en/train.tgt
# VALID_SRC_FILE=data/bea19_dev/valid.src
# VALID_TGT_FILE=data/bea19_dev/valid.tgt

# cat /home/ljh/GEC/QualityGec/data/clang8_train/clang8.src /home/ljh/GEC/QualityGec/data/stage2_en/train.src > $TRAIN_SRC_FILE
# cat /home/ljh/GEC/QualityGec/data/clang8_train/clang8.tgt /home/ljh/GEC/QualityGec/data/stage2_en/train.tgt > $TRAIN_TGT_FILE


# # apply bpe
# if [ ! -f $PROCESSED_DIR/train.bpe.src ]; then
#   echo "Apply BPE..."
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $TRAIN_SRC_FILE \
#             --outputs $PROCESSED_DIR/train.bpe.src \
#             --workers 32 \
#             --keep-empty;
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $TRAIN_TGT_FILE \
#             --outputs $PROCESSED_DIR/train.bpe.tgt \
#             --workers 32 \
#             --keep-empty;
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $VALID_SRC_FILE \
#             --outputs $PROCESSED_DIR/valid.bpe.src \
#             --workers 32 \
#             --keep-empty;
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $VALID_TGT_FILE \
#             --outputs $PROCESSED_DIR/valid.bpe.tgt \
#             --workers 32 \
#             --keep-empty;
# fi



# # fairseq preprocess
# mkdir -p $PROCESSED_DIR
# cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
# cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
# cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
# cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt


# echo "Preprocess..."
# rm -rf $PROCESSED_DIR/bin
# mkdir -p $PROCESSED_DIR/bin

# python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
#        --task translation \
#        --trainpref $PROCESSED_DIR/train.bpe \
#        --validpref $PROCESSED_DIR/valid.bpe \
#        --destdir $PROCESSED_DIR/bin \
#        --workers $WORKER_NUM \
#        --srcdict /home/ljh/model/bart-large-fairseq/dict.txt \
#        --tgtdict /home/ljh/model/bart-large-fairseq/dict.txt 

# echo "Finished!"


# #######################
# # Preprocess Wi+Locness
# #######################

# FAIRSEQ_DIR=src/fairseq-0.10.2/fairseq_cli
# PROCESSED_DIR=preprocess/wi_locness
# mkdir -p $PROCESSED_DIR

# WORKER_NUM=32

# # File path
# TRAIN_SRC_FILE=/home/ljh/GEC/QualityGec/data/wi_locness_train/train.src
# TRAIN_TGT_FILE=/home/ljh/GEC/QualityGec/data/wi_locness_train/train.tgt
# VALID_SRC_FILE=data/bea19_dev/valid.src
# VALID_TGT_FILE=data/bea19_dev/valid.tgt


# # apply bpe
# if [ ! -f $PROCESSED_DIR/train.bpe.src ]; then
#   echo "Apply BPE..."
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $TRAIN_SRC_FILE \
#             --outputs $PROCESSED_DIR/train.bpe.src \
#             --workers 32 \
#             --keep-empty;
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $TRAIN_TGT_FILE \
#             --outputs $PROCESSED_DIR/train.bpe.tgt \
#             --workers 32 \
#             --keep-empty;
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $VALID_SRC_FILE \
#             --outputs $PROCESSED_DIR/valid.bpe.src \
#             --workers 32 \
#             --keep-empty;
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $VALID_TGT_FILE \
#             --outputs $PROCESSED_DIR/valid.bpe.tgt \
#             --workers 32 \
#             --keep-empty;
# fi



# # fairseq preprocess
# mkdir -p $PROCESSED_DIR
# cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
# cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
# cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
# cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt


# echo "Preprocess..."
# rm -rf $PROCESSED_DIR/bin
# mkdir -p $PROCESSED_DIR/bin

# python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
#        --task translation \
#        --trainpref $PROCESSED_DIR/train.bpe \
#        --validpref $PROCESSED_DIR/valid.bpe \
#        --destdir $PROCESSED_DIR/bin \
#        --workers $WORKER_NUM \
#        --srcdict /home/ljh/model/bart-large-fairseq/dict.txt \
#        --tgtdict /home/ljh/model/bart-large-fairseq/dict.txt 

# echo "Finished!"


# #######################
# # Preprocess CoNLL-14-Test
# #######################

# FAIRSEQ_DIR=src/fairseq-0.10.2/fairseq_cli
# PROCESSED_DIR=preprocess/conll14_test
# mkdir -p $PROCESSED_DIR
# WORKER_NUM=32

# # File path
# TEST_SRC_FILE=data/conll14_test/test.src
# TEST_TGT_FILE=data/conll14_test/test.tgt

# # apply bpe
# if [ ! -f  $PROCESSED_DIR/test.bpe.src ]; then
#   echo "Apply BPE..."
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $TEST_SRC_FILE \
#             --outputs $PROCESSED_DIR/test.bpe.src \
#             --workers 32 \
#             --keep-empty;
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $TEST_TGT_FILE \
#             --outputs $PROCESSED_DIR/test.bpe.tgt \
#             --workers 32 \
#             --keep-empty;
# fi


# # fairseq preprocess

# cp $TEST_SRC_FILE $PROCESSED_DIR/test.src


# echo "Preprocess..."
# mkdir -p $PROCESSED_DIR/bin

# python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
#        --only-source \
#        --task translation \
#        --testpref $PROCESSED_DIR/test.bpe \
#        --destdir $PROCESSED_DIR/bin \
#        --workers $WORKER_NUM \
#        --srcdict /home/ljh/model/bart-large-fairseq/dict.txt \
#        --tgtdict /home/ljh/model/bart-large-fairseq/dict.txt 

# echo "Finished!"

# #######################
# # Preprocess BEA-19-Test
# #######################

# FAIRSEQ_DIR=src/fairseq-0.10.2/fairseq_cli
# PROCESSED_DIR=preprocess/bea19_test
# mkdir -p $PROCESSED_DIR
# WORKER_NUM=32

# # File path
# TEST_SRC_FILE=data/bea19_test/test.src

# # apply bpe
# if [ ! -f  $PROCESSED_DIR/test.bpe.src ]; then
#   echo "Apply BPE..."
#   python src/fairseq-0.10.2/examples/roberta/multiprocessing_bpe_encoder.py \
#             --encoder-json /home/ljh/model/bart-large-fairseq/encoder.json \
#             --vocab-bpe /home/ljh/model/bart-large-fairseq/vocab.bpe \
#             --inputs $TEST_SRC_FILE \
#             --outputs $PROCESSED_DIR/test.bpe.src \
#             --workers 32 \
#             --keep-empty;
# fi


# # fairseq preprocess

# cp $TEST_SRC_FILE $PROCESSED_DIR/test.src


# echo "Preprocess..."
# mkdir -p $PROCESSED_DIR/bin

# python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
#        --only-source \
#        --task translation \
#        --testpref $PROCESSED_DIR/test.bpe \
#        --destdir $PROCESSED_DIR/bin \
#        --workers $WORKER_NUM \
#        --srcdict /home/ljh/model/bart-large-fairseq/dict.txt \
#        --tgtdict /home/ljh/model/bart-large-fairseq/dict.txt 

# echo "Finished!"