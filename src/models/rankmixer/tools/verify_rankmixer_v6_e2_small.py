#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict FP32 checks for the only optimized component in v6-E2-Small.

The reference executes the ORIGINAL v6-E2 tokenization methods at D=256,
bypassing its constructor's D=512 architecture guard. The candidate executes
the Small methods. Both use real TensorFlow ops and the same checkpoint.
Model source is read with AST so this isolated check needs neither HDFS nor a
running Flood PS. It does not mock TensorFlow kernels or measure full training.

Run in the existing server environment; do not install/upgrade dependencies:
    python src/models/rankmixer/tools/verify_rankmixer_v6_e2_small.py
    python src/models/rankmixer/tools/verify_rankmixer_v6_e2_small.py \
        --benchmark-steps 30 --benchmark-batch-size 2048 --output /tmp/small.json

Native Adam is used ONLY as a deterministic update/slot-restore probe here.
The production model retains FloodAdam, Flood BN, data loading and PS updates.
"""
import argparse
import ast
from collections import Counter
import json
import logging
from pathlib import Path
import platform
import tempfile
import time


ROOT = Path(__file__).resolve().parents[4]
REFERENCE_PATH = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2.py'
SMALL_PATH = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small.py'
MODEL_BASE_PATH = ROOT / 'src/models/model_base.py'
KERNEL_METHODS = {
    'get_init', '_build_semantic_feature_groups',
    '_calculate_dense_trainable_params', '_rm_rms_norm', '_rm_norm',
    '_project_token_family', '_semantic_tokenize',
}


def tensorflow_v1():
    import tensorflow as tensorflow
    if int(tensorflow.__version__.split('.')[0]) >= 2:
        tf = tensorflow.compat.v1
        tf.disable_v2_behavior()
        return tf, tensorflow.__version__
    return tensorflow, tensorflow.__version__


def model_class_ast(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name in ('MLPModel', 'ModelBase'))


def load_kernel_model(path, tf=None, np=None, optimized=True, partitions=1):
    """Load source methods unchanged; no imports or constructor are executed."""
    import math
    source_class = model_class_ast(path)
    methods = [node for node in source_class.body
               if isinstance(node, ast.FunctionDef) and node.name in KERNEL_METHODS]
    if {node.name for node in methods} != KERNEL_METHODS:
        raise ValueError('Missing source methods in {}'.format(path))
    assignments = [node for node in source_class.body if isinstance(node, ast.Assign)]
    activation = next(node for node in model_class_ast(MODEL_BASE_PATH).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'get_act_func')
    kernel_class = ast.ClassDef(name='KernelModel', bases=[], keywords=[],
                               body=assignments + methods + [activation], decorator_list=[])
    module = ast.Module(body=[kernel_class])
    # Python 3.6/3.7 (older TF servers) do not have Module.type_ignores.
    if 'type_ignores' in ast.Module._fields:
        module.type_ignores = []
    ast.fix_missing_locations(module)
    namespace = {'tf': tf, 'np': np, 'math': math, 'logging': logging}
    exec(compile(module, str(path), 'exec'), namespace)
    model = namespace['KernelModel']()
    model.embedding_size = 17
    model.senet_hidden_size = 128
    model.rm_hidden_dim = 256
    model.rm_token_num = 32
    model.rm_local_token_num = 31
    model.rm_layer_num = 2
    model.rm_swiglu_hidden_dim = 704
    model.rm_rms_epsilon = 1e-6
    model.rm_token_proj_act = 'gelu_2'
    model.cvr_layers = [2048, 2048, 256]
    model.rm_optimize_tokenize = optimized
    model.rm_semantic_feature_groups = model._build_semantic_feature_groups()
    model.partitioner = (tf.min_max_variable_partitioner(
        max_partitions=partitions, min_slice_size=1024000) if tf is not None else None)
    return model


class TokenizationGraph:
    def __init__(self, tf, np, path, config, partitions, optimized=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(20260902)
            self.model = load_kernel_model(path, tf, np, optimized, partitions)
            self.inputs = tf.placeholder(tf.float32, [None, 20978], name='embeddings')
            self.cotangent = tf.placeholder(tf.float32, [None, 31, 256], name='cotangent')
            columns = tf.split(self.inputs, [17] * 1234, axis=1)
            bucket_field_maps = {}
            offset = 0
            for bucket in self.model._BUCKET_NAMES:
                field_ids = [field for _, fields in self.model.rm_semantic_feature_groups[bucket]
                             for field in fields]
                bucket_field_maps[bucket] = dict(zip(field_ids, columns[offset:offset + len(field_ids)]))
                offset += len(field_ids)
            if offset != 1234:
                raise AssertionError('Field coverage changed')
            start = len(self.graph.get_operations())
            with tf.variable_scope('Cvr-task-part', reuse=tf.AUTO_REUSE,
                                   partitioner=self.model.partitioner):
                self.output = self.model._semantic_tokenize(bucket_field_maps, export=False)
            self.forward_ops = Counter(op.type for op in self.graph.get_operations()[start:])
            if self.output.shape.as_list() != [None, 31, 256]:
                raise AssertionError('Token output shape changed')
            self.variables = tf.trainable_variables()
            # A random cotangent checks a dense vector-Jacobian product, avoiding
            # cancellation that an unweighted sum of normalized outputs can hide.
            self.loss = tf.reduce_sum(self.output * self.cotangent) / tf.cast(
                tf.shape(self.inputs)[0], tf.float32)
            start = len(self.graph.get_operations())
            gradients = tf.gradients(self.loss, [self.inputs] + self.variables)
            if any(gradient is None for gradient in gradients):
                raise AssertionError('Disconnected tokenization gradient')
            gradients = [tf.convert_to_tensor(gradient) for gradient in gradients]
            self.backward_ops = Counter(op.type for op in self.graph.get_operations()[start:])
            self.checks = {'tokens': self.output, 'probe_loss': self.loss,
                           'input_gradient': gradients[0]}
            self.checks.update(('gradient/' + variable.op.name, gradient)
                               for variable, gradient in zip(self.variables, gradients[1:]))
            self.optimizer = tf.train.AdamOptimizer(2e-5, beta1=0.9, beta2=0.999, epsilon=1e-8)
            self.train_op = self.optimizer.apply_gradients(zip(gradients[1:], self.variables))
            self.state = {variable.op.name: variable for variable in tf.global_variables()}
            self.manifest = {name: variable.shape.as_list() for name, variable in self.state.items()}
            self.saver = tf.train.Saver(var_list=self.state)
            self.initializer = tf.global_variables_initializer()
        self.session = tf.Session(graph=self.graph, config=config)

    def feed(self, data, cotangent):
        return {self.inputs: data, self.cotangent: cotangent}

    def close(self):
        self.session.close()


def assert_exact(np, reference, actual, label):
    if reference.keys() != actual.keys():
        raise AssertionError('{}: tensor names differ'.format(label))
    for name in reference:
        left, right = reference[name], actual[name]
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise AssertionError('{} / {}: non-finite result'.format(label, name))
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            error = float(np.max(np.abs(left - right))) if left.shape == right.shape else None
            raise AssertionError('{} / {}: strict comparison failed; max_abs_error={}'.format(
                label, name, error))


def make_batch(np, rng, batch_size, zeros=False):
    data = rng.normal(0, 0.5, [batch_size, 20978]).astype(np.float32)
    if zeros:
        data.fill(0)
    cotangent = rng.normal(0, 0.02, [batch_size, 31, 256]).astype(np.float32)
    return data, cotangent


def verify_partition(tf, np, config, partitions, optimized, benchmark_steps=0,
                     benchmark_batch_size=2048, warmup_steps=5):
    reference = TokenizationGraph(tf, np, REFERENCE_PATH, config, partitions)
    candidate = None
    try:
        candidate = TokenizationGraph(tf, np, SMALL_PATH, config, partitions, optimized)
        if reference.manifest != candidate.manifest:
            raise AssertionError('Checkpoint variable/Adam slot names or shapes changed')
        count = sum(int(np.prod(variable.shape.as_list())) for variable in candidate.variables)
        if count != 5386240:
            raise AssertionError('Unexpected tokenization parameter count: {}'.format(count))
        if candidate.model._calculate_dense_trainable_params() != 102356069:
            raise AssertionError('Unexpected full Small parameter count')
        if optimized:
            if candidate.forward_ops['Transpose'] != reference.forward_ops['Transpose']:
                raise AssertionError('Projection layout changed')
            if candidate.backward_ops['StridedSliceGrad'] != 0:
                raise AssertionError('Full-family slice gradients remain')
        reference.session.run(reference.initializer)
        rng = np.random.RandomState(20260902)
        batches = [make_batch(np, rng, size) for size in (1, 7, 17)]
        batches.append(make_batch(np, rng, 3, zeros=True))
        with tempfile.TemporaryDirectory(prefix='rankmixer-small-check-') as directory:
            checkpoint = str(Path(directory) / 'tokenization')
            reference.saver.save(reference.session, checkpoint)
            # No candidate initialization: every parameter and Adam state must
            # be restored from the reference checkpoint by the original name.
            candidate.saver.restore(candidate.session, checkpoint)
            for index, (data, cotangent) in enumerate(batches):
                expected = reference.session.run(reference.checks, reference.feed(data, cotangent))
                actual = candidate.session.run(candidate.checks, candidate.feed(data, cotangent))
                assert_exact(np, expected, actual, 'batch {}'.format(index))
            for data, cotangent in batches[:3]:
                reference.session.run(reference.train_op, reference.feed(data, cotangent))
                candidate.session.run(candidate.train_op, candidate.feed(data, cotangent))
                assert_exact(np, reference.session.run(reference.state),
                             candidate.session.run(candidate.state), 'Adam update/state')
            # Check a trained checkpoint, including nonzero first/second moments.
            reference.saver.save(reference.session, checkpoint)
            candidate.saver.restore(candidate.session, checkpoint)
            data, cotangent = batches[1]
            assert_exact(np, reference.session.run(reference.checks, reference.feed(data, cotangent)),
                         candidate.session.run(candidate.checks, candidate.feed(data, cotangent)),
                         'trained checkpoint restore')
        report = {
            'partitions': partitions, 'optimized': optimized, 'strict_equal': True,
            'max_abs_error': 0.0, 'batch_sizes': [1, 7, 17, 3], 'zero_input_checked': True,
            'adam_steps_checked': 3, 'checkpoint_and_slots_restored': True,
            'tokenization_parameters': count, 'full_dense_parameters': 102356069,
            'reference_forward_transposes': reference.forward_ops['Transpose'],
            'candidate_forward_transposes': candidate.forward_ops['Transpose'],
            'reference_slice_gradients': reference.backward_ops['StridedSliceGrad'],
            'candidate_slice_gradients': candidate.backward_ops['StridedSliceGrad'],
        }
        if benchmark_steps:
            data, cotangent = make_batch(np, rng, benchmark_batch_size)
            elapsed = {'reference': [], 'candidate': []}
            for index in range(warmup_steps + benchmark_steps):
                for name, fixture in (('reference', reference), ('candidate', candidate)):
                    start = time.perf_counter()
                    fixture.session.run(fixture.train_op, fixture.feed(data, cotangent))
                    duration = time.perf_counter() - start
                    if index >= warmup_steps:
                        elapsed[name].append(duration)
            assert_exact(np, reference.session.run(reference.state),
                         candidate.session.run(candidate.state), 'benchmark update/state')
            report['benchmark'] = {
                'scope': 'tokenization + backward + native Adam + feed copies; no Flood/PS/I/O',
                'batch_size': benchmark_batch_size, 'steps': benchmark_steps,
                'warmup_steps': warmup_steps,
                'mean_ms': {name: 1000 * float(np.mean(values)) for name, values in elapsed.items()},
                'median_ms': {name: 1000 * float(np.median(values)) for name, values in elapsed.items()},
            }
        return report
    finally:
        reference.close()
        if candidate is not None:
            candidate.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--partitions', type=int, nargs='+', default=[1, 3])
    parser.add_argument('--reference-mode', action='store_true', help='Check Small with optimization disabled')
    parser.add_argument('--intra-op-threads', type=int, default=32)
    parser.add_argument('--inter-op-threads', type=int, default=8)
    parser.add_argument('--benchmark-steps', type=int, default=0)
    parser.add_argument('--benchmark-batch-size', type=int, default=2048)
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if any(value < 1 for value in args.partitions) or args.benchmark_batch_size < 1:
        parser.error('partitions and benchmark batch size must be positive')
    if min(args.intra_op_threads, args.inter_op_threads, args.benchmark_steps, args.warmup_steps) < 0:
        parser.error('thread counts and step counts must be nonnegative')
    import numpy as np
    tf, version = tensorflow_v1()
    config = tf.ConfigProto(intra_op_parallelism_threads=args.intra_op_threads,
                            inter_op_parallelism_threads=args.inter_op_threads,
                            device_count={'GPU': 0})
    result = {
        'tensorflow': version, 'platform': platform.platform(), 'python': platform.python_version(),
        'dtype': 'float32', 'comparison': 'exact (no atol/rtol)',
        'reference': 'original v6-E2 tokenization at D=256, same weights and inputs',
        'scope': 'isolated CPU TensorFlow tokenization; not full Flood training or AUC validation',
        'intra_op_threads': args.intra_op_threads, 'inter_op_threads': args.inter_op_threads,
        'checks': [],
    }
    for partitions in args.partitions:
        result['checks'].append(verify_partition(
            tf, np, config, partitions, not args.reference_mode,
            args.benchmark_steps, args.benchmark_batch_size, args.warmup_steps))
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + '\n', encoding='utf-8')
    print(output)
    return result


if __name__ == '__main__':
    main()
