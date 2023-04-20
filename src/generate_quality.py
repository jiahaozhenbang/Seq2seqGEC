#!/usr/bin/env python3 -u
#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from itertools import chain

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar
import numpy as np
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(args, override_args=None):
    
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(model_args)

    # Build criterion
    criterion = task.build_criterion(model_args)
    criterion.eval()


    correct_probs = []
    entropy = []
    target_lens = []
    ids = []
    for subset in args.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        # print(dataset.ordered_indices()[:10])
        def get_quality(sample, model):
            # print(sample.keys())
            # print(sample['id'].shape, sample['target'].shape)
            # print(sample['id'])
            # print(sample["net_input"]['src_lengths'])
            # print(sample['target'][0])

            # print(sample["net_input"].keys())
            # for key in sample["net_input"]:
            #     print(type(sample["net_input"][key]))
            #     if hasattr(sample["net_input"][key], 'shape'):
            #         print(sample["net_input"][key].shape)
            #         print(sample["net_input"][key][0])
            # exit()
            net_output = model(**sample["net_input"])
            probs = model.get_normalized_probs(net_output, log_probs=False)
            target = model.get_targets(sample, net_output)
            
            assert target.dim() == probs.dim() - 1
            target_lens = target.ne(criterion.padding_idx).sum(dim=-1).cpu().tolist()
            correct_probs = correct_probs = torch.gather(probs, dim=2, index=target.unsqueeze(-1)).squeeze(-1).cpu().tolist()
            entropy = (torch.sum(- probs * torch.log2(probs), dim=-1) / torch.log2(torch.tensor(probs.shape[-1]))).cpu().tolist()
            ids = sample['id'].cpu().tolist()

            sample_size = (
                sample["ntokens"]
            )
            logging_output = {
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            return correct_probs, entropy, target_lens, ids, logging_output

        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            correct_probs_per_batch, entropy_per_batch, target_lens_per_batch, ids_per_batch, log_output_per_batch = get_quality(sample, model)
            correct_probs.extend(correct_probs_per_batch)
            entropy.extend(entropy_per_batch)
            target_lens.extend(target_lens_per_batch)
            ids.extend(ids_per_batch)
            # _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
            progress.log(log_output_per_batch, step=i)
            log_outputs.append(log_output_per_batch)

        if args.distributed_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=getattr(args, "all_gather_list_size", 16384),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        progress.print(log_output, tag=subset, step=i)

    def save(origin_array_list, target_lens, file):
        lengths = target_lens
        return_array = np.asarray(np.zeros((len(lengths), max(lengths))))
        for i, _len in tqdm(enumerate(lengths)):
            slices = tuple([i, slice(0, _len)])
            return_array[slices] = origin_array_list[i][:_len]
        np.savez(file, data=return_array, lengths=np.array(lengths))
    
    assert len(correct_probs) == len(entropy) and len(entropy) == len(target_lens)
    recovery_ids = np.argsort(ids)
    correct_probs = [correct_probs[id] for id in recovery_ids]
    entropy = [entropy[id] for id in recovery_ids]
    target_lens = [target_lens[id] for id in recovery_ids]
    print('## begin save correct_probs')
    save(correct_probs, target_lens, args.correct_probs_output_file)
    
    print('## begin save entropy')
    save(entropy, target_lens, args.entropy_output_file)
    

def cli_main():

    parser = options.get_validation_parser()
    parser.add_argument('--correct_probs_output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--entropy_output_file',
                        help='Path to the output file',
                        required=True)
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_parser.add_argument('--correct_probs_output_file',
                        help='Path to the output file',
                        required=True)
    override_parser.add_argument('--entropy_output_file',
                        help='Path to the output file',
                        required=True)
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)


    distributed_utils.call_main(args, main, override_args=override_args)


if __name__ == "__main__":
    cli_main()
