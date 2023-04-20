

# from fairseq import (
#     checkpoint_utils,
#     distributed_utils,
#     options,
#     quantization_utils,
#     tasks,
#     utils,
# )
# parser = options.get_training_parser()
# print(options.parse_args_and_arch(parser, modify_parser=None))

# import numpy as np
# tgt_file = 'preprocess/stage2_en/train.bpe.tgt'
# with open(tgt_file, 'r') as f:
#     tgt = f.readlines()
# correct_probs = 'preprocess/stage2_en/train.correct_probs.npz'
# correct_probs_dict = np.load(correct_probs)
# correct_probs = dict(correct_probs_dict)

# print(len(correct_probs['lengths']))


# for i in range(10):
#     print(correct_probs['data'][i][:correct_probs['lengths'][i]])
#     print(correct_probs['lengths'][i])


# i = 2388458
# print(correct_probs['lengths'][i])
# print(tgt[i].split().__len__())

# i = 2388459
# print(correct_probs['lengths'][i])
# print(tgt[i].split().__len__())

# i = 2388460
# print(correct_probs['lengths'][i])
# print(tgt[i].split().__len__())

# index_c = 0
# index_t = 0
# patience = 10 
# while True:
#     if tgt[index_t].split().__len__() > 1024 or index_t == 1363024:
#         index_t += 1
#         continue
#     if tgt[index_t].split().__len__() + 1 != correct_probs['lengths'][index_c]:
#         print(index_c, index_t)
#         print(tgt[index_t].split().__len__(), correct_probs['lengths'][index_c])
#         patience -= 1
#         if patience < 0:
#             exit()
#     index_c += 1
#     index_t += 1

a='/home/ljh/GEC/Seq2seqGEC/preprocess/stage2_en/train.bpe.src'
b='/home/ljh/GEC/Seq2seqGEC/preprocess/stage2_en/train.bpe.tgt'
with open(a, 'r') as f:
    data_a = f.readlines()
with open(b, 'r') as f:
    data_b = f.readlines()
cnt=0
for _a, _b in zip(data_a, data_b):
    if _a==_b:
        cnt += 1
print(cnt)
print(data_a[1363024])
print(data_b[1363024])
print(data_a[1363024] == data_b[1363024])