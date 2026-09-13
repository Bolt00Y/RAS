#!/usr/bin/env python3
"""Static FLOPs worksheet for cvr_mature_rankmixer_fst_v1 (no TF/Flood needed).

Usage: python3 <this_file> --batch-size 2048 --details

Scope: one forward pass of the dense prediction path. Matrix products use
1 MAC = 2 FLOPs, excluding bias. Sparse lookup/pooling, loss, gradients,
optimizer, metrics, memory traffic, and communication are excluded.

The augmented estimate is a separate, illustrative scalar convention:
- add/subtract/multiply/divide count as 1;
- sigmoid/tanh/sqrt/rsqrt each count as 1 symbolic operation, NOT a measured
  hardware cost; x**3 is modeled as two multiplications;
- inference BN is a precomputed affine transform (2 ops/output), kept separate
  from Dense; BN constants and GELU constants are precomputed;
- means over n values use n-1 additions and one division;
- LayerNorm follows the source literally with both (x-mean) expressions,
  8n+2 ops/vector; RMSNorm uses 4n+2 ops/vector;
- retain multiplication by residual_scale=1; exclude comparisons (ReLU/clip),
  reshapes, transposes, concatenations, and splits.
Do not interpret the augmented estimate as a TF profiler result or exact
instruction count. Fusion and actual nonlinear/custom kernels change it.
"""

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/models/rankmixer/cvr_mature_rankmixer_fst_v1.py'
CONFIG = ROOT / 'bash/set-cvr-mature-rankmixer-fst-v1-args.txt'


def source_spec():
    """Read field literals and token allocations without importing the model."""
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    fields = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == '_ids'):
            fields[node.targets[0].id] = ast.literal_eval(node.value.args[0]).split()
    model = next(n for n in tree.body
                 if isinstance(n, ast.ClassDef) and n.name == 'MLPModel')
    attrs = {n.targets[0].id: n.value for n in model.body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    groups = []
    for kind in ('_USER_GROUPS', '_ITEM_GROUPS'):
        part = []
        for group in attrs[kind].elts:
            name, ids, tokens = group.elts
            part.append((ast.literal_eval(name), len(fields[ids.id]),
                         ast.literal_eval(tokens)))
        groups.append(part)
    match = re.search(r"--model_args='([^']+)'", CONFIG.read_text(encoding='utf-8'))
    if match is None:
        raise ValueError('Cannot locate the model_args JSON in the config.')
    cfg = json.loads(match.group(1))
    frozen = dict(embedding_size=17, mixup_token_num=32, mixup_token_dim=256,
                  mlp_mixer_layers=3, mixer_expand_ratio=3.5,
                  global_token_hidden_dim=512, creative_hidden_dim=256,
                  creative_output_dim=32, cvr_layers=[256, 128],
                  use_senet=True, use_senet_bn=True, batch_norm=True)
    if any(cfg.get(key) != value for key, value in frozen.items()):
        raise ValueError('Architecture changed: update this worksheet first.')
    expected_params = ast.literal_eval(attrs['_EXPECTED_DENSE_TRAINABLE_PARAMS'])
    return groups[0], groups[1], len(fields['_CREATIVE_IDS']), expected_params


def calculate():
    user_groups, item_groups, creative_fields, expected_params = source_spec()
    E, T, D, M, L = 17, 32, 256, 896, 3
    U = sum(n for _, n, _ in user_groups) * E
    I = sum(n for _, n, _ in item_groups) * E
    C = creative_fields * E
    if (U, I, C, sum(t for _, _, t in user_groups + item_groups)) != (
            6545, 14195, 238, 31):
        raise ValueError('Feature routing changed: update this worksheet first.')
    rows = []

    def dense(module, name, inputs, outputs, repetitions=1):
        macs = repetitions * inputs * outputs
        rows.append(dict(module=module, name=name, inputs=inputs, outputs=outputs,
                         repetitions=repetitions, macs=macs, flops=2 * macs,
                         bias=repetitions * outputs))

    for name, a, h, b in (
            ('user', U, 256, U), ('item', U + I, 128, I),
            ('creative', C, 128, C)):
        dense('senet', name + '.squeeze', a, h)
        dense('senet', name + '.excitation', h, b)
    for name, fields, tokens in user_groups + item_groups:
        dense('local_tokens', name, fields * E, tokens * D)
    dense('global_token', 'hidden', U + I, 512)
    dense('global_token', 'projection', 512, D)
    for layer in range(1, L + 1):
        dense('mixer', f'block{layer}.gate', D, M, T)
        dense('mixer', f'block{layer}.value', D, M, T)
        dense('mixer', f'block{layer}.down', M, D, T)
    dense('creative_bypass', 'hidden', C, 256)
    dense('creative_bypass', 'projection', 256, 32)
    dense('task_head', 'hidden1', D + 32, 256)
    dense('task_head', 'hidden2', 256, 128)
    dense('task_head', 'output', 128, 1)

    modules = defaultdict(lambda: dict(macs=0, flops=0, bias=0))
    for row in rows:
        for key in ('macs', 'flops', 'bias'):
            modules[row['module']][key] += row[key]
    main = sum(row['flops'] for row in rows)
    bias = sum(row['bias'] for row in rows)

    # Each dense kernel is used once per sample in this specific model.
    # Norm parameters and trainable Swish beta are counted independently.
    other_params = (2 * (U + I + C) + 2 * 512 + 2 * 31 * D
                    + 2 * (U + I) + 2 * D
                    + L * (2 * D + M + D) + 2 * D
                    + 3 * (256 + 32) + 2 * (256 + 128))
    params = main // 2 + bias + other_params
    if params != expected_params:
        raise ValueError(f'Parameter cross-check failed: {params} != {expected_params}')

    def ln(n):
        return 8 * n + 2

    def rms(n):
        return 4 * n + 2

    extra = {
        'dense_bias_adds': bias,
        'input_bn_affine': 2 * (U + I + C),
        'senet_bn_affine': 2 * 512,
        'senet_sigmoid_and_multiply': 2 * (U + I + C),
        'local_gelu_and_bn': 11 * 31 * D,
        'global_ln_and_gelu': ln(U + I) + 9 * 512 + ln(D),
        'mixer_ln_rms_activations_residuals': L * (
            T * ln(D) + 3 * T * M + T * rms(M) + T * rms(D) + 2 * T * D),
        'mixer_final_ln': T * ln(D),
        'mean_pool': T * D,
        'creative_bn_and_swish': 5 * (256 + 32),
        'head_bn_and_gelu': 11 * (256 + 128),
        'final_sigmoid': 1,
    }
    return rows, modules, main, extra, params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--details', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    rows, modules, total, extra, params = calculate()
    print(f'Source: {SOURCE}')
    print('Convention: matrix products only, 1 MAC = 2 FLOPs; forward pass.')
    print('The following module and operator rows are PER SAMPLE.')
    print(f'{"Module":20s} {"MACs":>15s} {"FLOPs":>15s} {"Share":>9s}')
    for name, values in modules.items():
        print(f'{name:20s} {values["macs"]:15,d} {values["flops"]:15,d}'
              f' {values["flops"] / total:8.2%}')
    if args.details:
        print('\nPer-matrix ledger: repetitions * input_width * output_width * 2')
        for row in rows:
            print(f'{row["module"]}.{row["name"]}: '
                  f'{row["repetitions"]} * {row["inputs"]} * {row["outputs"]} * 2'
                  f' = {row["flops"]:,} FLOPs')
        print('\nIllustrative additional scalar-operation ledger (see docstring):')
        for name, value in extra.items():
            print(f'{name}: {value:,}')
    print(f'\nDense trainable parameter cross-check: {params:,} (PASS)')
    print(f'Matrix FLOPs/sample: {total:,} = {total / 1e6:.6f} MFLOPs'
          f' = {total / 1e9:.9f} GFLOPs')
    batch = args.batch_size
    print(f'Matrix FLOPs/batch (B={batch}): {total * batch:,}'
          f' = {total * batch / 1e9:.9f} GFLOPs')
    print(f'Forward+backward matrix estimate/batch (~3x): {3 * total * batch:,}'
          f' = {3 * total * batch / 1e12:.12f} TFLOPs; excludes optimizer, etc.')
    print(f'Illustrative augmented dense inference estimate/sample: '
          f'{total + sum(extra.values()):,} operations'
          f' ({(total + sum(extra.values())) / 1e6:.6f} M); NOT measured FLOPs.')


if __name__ == '__main__':
    main()
