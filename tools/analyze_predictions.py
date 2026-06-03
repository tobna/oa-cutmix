"""Evaluate a model on its test/val set and store per-sample prediction
results: dataset index, ground-truth label, predicted label, probability
on the correct label, and whether the prediction is correct.

Usage (single GPU)::

    python tools/analyze_predictions.py \
        configs/my_config.py work_dirs/exp/latest.pth \
        --output work_dirs/exp/predictions.npz

The output ``.npz`` file contains:

- ``indices``     : int array (N,) – dataset indices in evaluation order
- ``gt_labels``   : int array (N,) – ground-truth class indices
- ``pred_labels`` : int array (N,) – argmax-predicted class indices
- ``probs``       : float array (N, C) – softmax probabilities
- ``correct_prob``: float array (N,) – softmax prob assigned to the gt label
- ``is_correct``  : bool  array (N,) – True where prediction matches gt

A short summary (top-1 accuracy, mean correct/wrong prob) is printed and
written alongside the ``.npz`` as ``predictions_summary.txt``.
"""
import argparse
import importlib
import os
import os.path as osp
import time

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from openmixup.datasets import build_dataloader, build_dataset
from openmixup.models import build_model
from openmixup.utils import (
    get_root_logger,
    setup_multi_processes,
    traverse_replace,
)


# ---------------------------------------------------------------------------
# Core collection
# ---------------------------------------------------------------------------

def collect_predictions(model, data_loader, head_key=None):
    """Run inference and collect per-sample results.

    Args:
        model: Wrapped model (e.g. MMDataParallel).
        data_loader: DataLoader (shuffle=False).
        head_key (str | None): Which output head to use.  If None the first
            key returned by the model is used.

    Returns:
        indices     (np.ndarray, int64,   N)
        gt_labels   (np.ndarray, int64,   N)
        all_logits  (np.ndarray, float32, N x C)
    """
    model.eval()

    use_cuda = next(model.parameters()).is_cuda
    # CPU: log every 100 images; CUDA: log every 100 batches
    log_interval_batches = 100
    log_interval_images = 100

    all_indices = []
    all_gt_labels = []
    all_logits = []
    chosen_key = head_key  # resolved on first batch
    images_done = 0
    next_image_log = log_interval_images

    total = len(data_loader.dataset)

    for batch_idx, data in enumerate(data_loader):
        # Grab index and label from the batch *before* passing to model,
        # because some wrappers (MMDataParallel) may not forward non-tensor
        # keys back to the caller.
        indices = data['idx'].numpy().astype(np.int64)
        gt_labels = data['gt_label'].numpy().astype(np.int64)
        batch_size = len(indices)

        with torch.no_grad():
            result = model(mode='test', **data)
        # result: dict{head_key: Tensor (B, C)}

        if chosen_key is None:
            chosen_key = list(result.keys())[0]
        logits = result[chosen_key].cpu().numpy().astype(np.float32)

        all_indices.append(indices)
        all_gt_labels.append(gt_labels)
        all_logits.append(logits)

        images_done += batch_size

        if use_cuda:
            if (batch_idx + 1) % log_interval_batches == 0:
                print(
                    f'[CUDA] batch {batch_idx + 1}/{len(data_loader)}'
                    f'  ({images_done}/{total} images)')
        else:
            if images_done >= next_image_log:
                print(
                    f'[CPU]  {images_done}/{total} images processed')
                next_image_log = (
                    (images_done // log_interval_images) + 1
                ) * log_interval_images

    print(f'Done. {images_done}/{total} images processed.')

    indices = np.concatenate(all_indices, axis=0)
    gt_labels = np.concatenate(all_gt_labels, axis=0)
    logits = np.concatenate(all_logits, axis=0)

    return indices, gt_labels, logits, chosen_key


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def build_results(indices, gt_labels, logits):
    """Convert raw logits to the full per-sample result dict."""
    probs = softmax(logits)                               # (N, C)
    pred_labels = probs.argmax(axis=1).astype(np.int64)  # (N,)
    # probability the model assigned to the *correct* class
    correct_prob = probs[np.arange(len(gt_labels)), gt_labels].astype(
        np.float32)
    is_correct = (pred_labels == gt_labels)               # (N,) bool

    return dict(
        indices=indices,
        gt_labels=gt_labels,
        pred_labels=pred_labels,
        probs=probs,
        correct_prob=correct_prob,
        is_correct=is_correct,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a model and store per-sample prediction results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='path to output .npz file '
             '(default: <work_dir>/predictions.npz)')
    parser.add_argument(
        '--head-key',
        type=str,
        default=None,
        help='which output head to use (default: first head)')
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='override work_dir from config')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='device to run inference on, e.g. "cuda:0" or "cpu"')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options in key=value format')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    setup_multi_processes(cfg)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Resolve work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join(
            './work_dirs', work_type,
            osp.splitext(osp.basename(args.config))[0])
    device = torch.device(args.device)
    cfg.gpu_ids = [0] if device.type == 'cuda' else []
    cfg.model.pretrained = None

    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(
        cfg.work_dir, 'analyze_predictions_{}.log'.format(timestamp))
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level)

    # Build dataset and dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # Build model and load weights
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if device.type == 'cuda':
        model = MMDataParallel(model, device_ids=[device.index or 0])
    else:
        model = model.to(device)

    logger.info('Running inference …')
    indices, gt_labels, logits, used_key = collect_predictions(
        model, data_loader, head_key=args.head_key)
    logger.info('Used head key: %s', used_key)

    results = build_results(indices, gt_labels, logits)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.output is not None:
        out_path = args.output
    else:
        out_path = osp.join(cfg.work_dir, 'predictions.npz')

    np.savez(out_path, **results)
    logger.info('Saved per-sample results to %s', out_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n = len(indices)
    n_correct = results['is_correct'].sum()
    top1 = 100.0 * n_correct / n
    mean_prob_correct = results['correct_prob'][results['is_correct']].mean()
    mean_prob_wrong = results['correct_prob'][~results['is_correct']].mean()

    summary_lines = [
        'Prediction analysis summary',
        '===========================',
        f'  Samples evaluated : {n}',
        f'  Correct           : {n_correct}  ({top1:.2f}%)',
        f'  Wrong             : {n - n_correct}  ({100 - top1:.2f}%)',
        f'  Mean p(gt) correct: {mean_prob_correct:.4f}',
        f'  Mean p(gt) wrong  : {mean_prob_wrong:.4f}',
        f'  Head used         : {used_key}',
        f'  Output file       : {out_path}',
        '',
        'Array shapes in the .npz:',
        f'  indices      : {results["indices"].shape}  int64',
        f'  gt_labels    : {results["gt_labels"].shape}  int64',
        f'  pred_labels  : {results["pred_labels"].shape}  int64',
        f'  probs        : {results["probs"].shape}  float32',
        f'  correct_prob : {results["correct_prob"].shape}  float32',
        f'  is_correct   : {results["is_correct"].shape}  bool',
    ]
    summary = '\n'.join(summary_lines)
    logger.info('\n' + summary)

    summary_path = osp.splitext(out_path)[0] + '_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary + '\n')
    logger.info('Summary written to %s', summary_path)


if __name__ == '__main__':
    main()
