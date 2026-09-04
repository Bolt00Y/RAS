"""Architecture contracts for the v6-E2-Small-1 terminal ablation.

The production model remains a standalone TensorFlow 1.x/Flood module.  These
tests inspect its source without importing Flood and, when TensorFlow is
available, compare the new final LayerNorm numerically with mature_v1.
"""

import ast
import copy
import importlib.util
import json
import logging
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[4]
SMALL = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small.py'
VARIANT = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small_1.py'
MATURE = ROOT / 'src/models/rankmixer/cvr_senet_mature_rankmixer_v1.py'
VERIFY = ROOT / 'src/models/rankmixer/tools/verify_rankmixer_v6_e2_small.py'
ARGS = ROOT / 'bash/set-rankmixer-v6-e2-small-1-args.txt'
INTRO = ROOT / 'introduce/rankmixer_v6_e2_small_1_introduction.md'


def model_ast(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'MLPModel')


def method_map(model):
    return {node.name: node for node in model.body
            if isinstance(node, ast.FunctionDef)}


def class_value(model, name):
    return ast.literal_eval(next(
        node.value for node in model.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets)
    ))


def parse_args(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    model_args_line = next(line for line in lines
                           if line.startswith("--model_args='"))
    model_args = json.loads(model_args_line[len("--model_args='"):-1])
    outer_args = [line for line in lines[1:]
                  if not line.startswith('--model_args=')]
    return lines[0], model_args, outer_args


class CosmeticNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace(
                'RankMixer v6-E2-Small-1', 'RankMixer v6-E2-Small')
        return node

    def visit_Str(self, node):
        node.s = node.s.replace(
            'RankMixer v6-E2-Small-1', 'RankMixer v6-E2-Small')
        return node


def normalized(node):
    return ast.dump(
        CosmeticNormalizer().visit(copy.deepcopy(node)),
        include_attributes=False,
    )


def module_node(body):
    module = ast.Module(body=body)
    if 'type_ignores' in ast.Module._fields:
        module.type_ignores = []
    return ast.fix_missing_locations(module)


def literal_equals(node, expected):
    try:
        return ast.literal_eval(node) == expected
    except (TypeError, ValueError):
        return False


class Small1ArchitectureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small = model_ast(SMALL)
        cls.variant = model_ast(VARIANT)
        cls.small_methods = method_map(cls.small)
        cls.variant_methods = method_map(cls.variant)

    def test_standalone_source_and_server_imports(self):
        def imports(path):
            return [ast.dump(node, include_attributes=False)
                    for node in ast.parse(
                        path.read_text(encoding='utf-8')).body
                    if isinstance(node, (ast.Import, ast.ImportFrom))]

        self.assertEqual(imports(SMALL), imports(VARIANT))
        source = VARIANT.read_text(encoding='utf-8')
        self.assertIn('由 cvr_bn_rankmixer_v6_e2_small.py 发展的末端消融版本',
                      source.splitlines()[1])
        self.assertNotIn('from .cvr_bn_rankmixer', source)
        self.assertEqual([ast.dump(base) for base in self.variant.bases],
                         ["Name(id='ModelBase', ctx=Load())"])

    def test_only_terminal_architecture_methods_change(self):
        added = set(self.variant_methods) - set(self.small_methods)
        self.assertEqual(added, {'_rm_final_layer_norm', '_mature_batch_norm'})
        changed = {'__init__', '_calculate_dense_trainable_params',
                   '_task_head', 'model_fn'}
        unchanged = set(self.small_methods) - changed
        self.assertGreater(len(unchanged), 35)
        for name in sorted(unchanged):
            self.assertEqual(
                normalized(self.small_methods[name]),
                normalized(self.variant_methods[name]),
                name,
            )
        self.assertTrue({
            '_build_semantic_feature_groups', '_validate_semantic_feature_groups',
            'senet_layer', '_semantic_tokenize', '_build_global_token',
            '_rm_block', '_rm_per_token_swiglu', 'build_loss_op',
            'build_optimizer_op', 'get_dataset', 'train', 'test', '_build_export',
        }.issubset(unchanged))

    def test_feature_and_token_abi_is_unchanged(self):
        for name in ('_BUCKET_NAMES', '_EXPECTED_FIELD_COUNTS',
                     '_GROUP_VERSION', '_GROUP_CHECKSUMS', '_GROUP_SIZES'):
            self.assertEqual(class_value(self.small, name),
                             class_value(self.variant, name), name)
        self.assertNotIn('_creative_converter', self.variant_methods)

    def test_model_uses_final_layer_norm_then_mean_pool(self):
        method = self.variant_methods['model_fn']
        assignments = {
            target.id: node.value
            for node in ast.walk(method) if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        final_tokens = assignments['final_tokens']
        self.assertEqual(final_tokens.func.attr, '_rm_final_layer_norm')
        self.assertEqual(final_tokens.args[0].id, 'hidden_tokens')

        context = assignments['context']
        self.assertEqual(context.func.attr, 'reduce_mean')
        self.assertEqual(context.args[0].id, 'final_tokens')
        self.assertEqual(ast.literal_eval(next(
            keyword.value for keyword in context.keywords
            if keyword.arg == 'axis')),
            1)
        method_dump = ast.dump(method)
        self.assertNotIn('rm_pure_flatten', method_dump)
        self.assertNotIn('rm_final_rms_norm', method_dump)
        self.assertNotIn('creative_context', method_dump)

    def test_final_layer_norm_matches_mature_formula(self):
        method = self.variant_methods['_rm_final_layer_norm']
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        variables = [node for node in calls
                     if isinstance(node.func, ast.Attribute)
                     and node.func.attr == 'get_variable']
        names = [ast.literal_eval(node.args[0]) for node in variables]
        self.assertEqual(names, ['gamma', 'beta'])
        self.assertEqual(len([node for node in calls
                              if isinstance(node.func, ast.Attribute)
                              and node.func.attr == 'reduce_mean']), 2)
        self.assertEqual(len([node for node in calls
                              if isinstance(node.func, ast.Attribute)
                              and node.func.attr == 'square']), 1)
        self.assertIn("attr='rm_final_ln_epsilon'", ast.dump(method))
        self.assertNotIn("attr='_rm_norm'", ast.dump(method))

        mature_method = method_map(model_ast(MATURE))['_layer_norm']
        for required in ('gamma', 'beta', 'reduce_mean', 'square', 'sqrt'):
            self.assertIn(required, ast.dump(mature_method))
            self.assertIn(required, ast.dump(method))

    def test_task_head_has_mature_order_and_regularization(self):
        method = self.variant_methods['_task_head']
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        dense_calls = [node for node in calls
                       if isinstance(node.func, ast.Attribute)
                       and node.func.attr == 'fully_connected']
        self.assertEqual(len(dense_calls), 2)
        for dense_call in dense_calls:
            keywords = {keyword.arg: keyword.value
                        for keyword in dense_call.keywords}
            self.assertIn('weights_regularizer', keywords)
            regularizer = keywords['weights_regularizer']
            self.assertEqual(regularizer.func.attr, 'l2_regularizer')
            self.assertEqual(ast.dump(regularizer.args[0]),
                             "Attribute(value=Name(id='self', ctx=Load()), attr='l2_deep', ctx=Load())")

        hidden_dense = next(node for node in dense_calls
                            if any(keyword.arg == 'num_outputs'
                                   and isinstance(keyword.value, ast.Name)
                                   for keyword in node.keywords))
        hidden_keywords = {keyword.arg: keyword.value
                           for keyword in hidden_dense.keywords}
        self.assertIsNone(ast.literal_eval(hidden_keywords['activation_fn']))

        output_dense = next(node for node in dense_calls
                            if any(keyword.arg == 'num_outputs'
                                   and literal_equals(keyword.value, 1)
                                   for keyword in node.keywords))
        output_keywords = {keyword.arg: keyword.value
                           for keyword in output_dense.keywords}
        self.assertEqual(output_keywords['activation_fn'].attr, 'identity')

        self.assertEqual(len([node for node in calls
                              if isinstance(node.func, ast.Attribute)
                              and node.func.attr == '_mature_batch_norm']), 1)
        activation = next(node for node in calls
                          if isinstance(node.func, ast.Attribute)
                          and node.func.attr == 'get_act_func')
        self.assertEqual(activation.args[0].attr, 'mlp_act_type')
        self.assertEqual({keyword.arg for keyword in activation.keywords},
                         {'is_train', 'name'})

        bn_method = self.variant_methods['_mature_batch_norm']
        bn_call = next(node for node in ast.walk(bn_method)
                       if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute)
                       and node.func.attr == 'batch_norm_layer_v2')
        self.assertEqual({keyword.arg for keyword in bn_call.keywords}, {
            'x', 'train_phase', 'scope_bn', 'batch_norm_decay',
            'use_riemann_bn', 'renorm', 'renorm_decay', 'export',
        })

    def test_exact_parameter_budget(self):
        spec = importlib.util.spec_from_file_location('verify_small', VERIFY)
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        model = verifier.load_kernel_model(VARIANT)
        model.cvr_layers = [256, 128]
        self.assertEqual(model._EXPECTED_DENSE_TRAINABLE_PARAMS, 80938853)
        self.assertEqual(model._calculate_dense_trainable_params(), 80938853)
        task_head = (
            256 * 256 + 256 + 2 * 256
            + 256 * 128 + 128 + 2 * 128
            + 128 + 1
        )
        self.assertEqual(task_head, 99585)

    def test_args_only_change_the_declared_terminal_contract(self):
        _, old, old_outer = parse_args(
            ROOT / 'bash/set-rankmixer-v6-e2-small-args.txt')
        entry, new, new_outer = parse_args(ARGS)
        self.assertEqual(
            entry,
            'models.rankmixer.cvr_bn_rankmixer_v6_e2_small_1.MLPModel',
        )
        self.assertEqual(old_outer, new_outer)
        self.assertIn('--ignore_dense_checkpoint=True', new_outer)

        self.assertEqual(old.pop('rm_readout_type'), 'pure_flat')
        self.assertEqual(new.pop('rm_readout_type'), 'mean_pool')
        self.assertEqual(old.pop('cvr_layers'), [2048, 2048, 256])
        self.assertEqual(new.pop('cvr_layers'), [256, 128])
        self.assertEqual(new.pop('rm_final_norm_type'), 'layer_norm')
        self.assertEqual(new.pop('rm_final_ln_epsilon'), 1e-8)

        for name, expected in (
                ('batch_norm_decay', 0.9),
                ('embed_use_renorm', False),
                ('embed_renorm_decay', 0.99),
                ('l2_deep', 1e-6)):
            self.assertEqual(new.pop(name), expected)

        for name in ('skip_tensors', 'warm_up_tensors'):
            self.assertEqual(
                new.pop(name),
                old.pop(name).replace(
                    'rm_final_rms_norm', 'rm_final_layer_norm'),
            )
        self.assertEqual(old, new)

    def test_introduction_tracks_code_and_args(self):
        document = INTRO.read_text(encoding='utf-8')
        self.assertIn('80,938,853', document)
        self.assertIn('256 → 128 → 1', document)
        self.assertIn('rm_final_layer_norm', document)
        self.assertIn('set-rankmixer-v6-e2-small-1-args.txt', document)


@unittest.skipUnless(importlib.util.find_spec('tensorflow') is not None,
                     'Real TensorFlow runtime is required for numerical checks')
class Small1NumericalTest(unittest.TestCase):
    def test_final_layer_norm_and_pool_match_mature_v1(self):
        import numpy as np
        import tensorflow as tensorflow

        tf = (tensorflow if int(tensorflow.__version__.split('.')[0]) == 1
              else tensorflow.compat.v1)
        if int(tensorflow.__version__.split('.')[0]) >= 2:
            tf.disable_v2_behavior()

        variant_method = method_map(model_ast(VARIANT))['_rm_final_layer_norm']
        mature_method = method_map(model_ast(MATURE))['_layer_norm']
        candidate_class = ast.ClassDef(
            name='Candidate',
            bases=[],
            keywords=[],
            body=[variant_method],
            decorator_list=[],
        )
        reference_class = ast.ClassDef(
            name='Reference',
            bases=[],
            keywords=[],
            body=[mature_method],
            decorator_list=[],
        )
        namespace = {'tf': tf, 'math': math, 'logging': logging}
        exec(compile(module_node([candidate_class, reference_class]),
                     str(VARIANT), 'exec'), namespace)

        graph = tf.Graph()
        with graph.as_default():
            inputs = tf.placeholder(tf.float32, [None, 32, 256])
            candidate = namespace['Candidate']()
            candidate.rm_token_num = 32
            candidate.rm_hidden_dim = 256
            candidate.rm_final_ln_epsilon = 1e-8
            candidate.partitioner = None
            actual = candidate._rm_final_layer_norm(inputs)
            expected = namespace['Reference']._layer_norm(
                inputs,
                epsilon=1e-8,
                scope='mature_reference',
            )
            actual_pool = tf.reduce_mean(actual, axis=1)
            expected_pool = tf.reduce_mean(expected, axis=1)
            variables = tf.trainable_variables()
            gamma = np.linspace(0.7, 1.3, 256).astype(np.float32)
            beta = np.linspace(-0.2, 0.2, 256).astype(np.float32)
            assign = [tf.assign(variable, beta if variable.op.name.endswith('/beta')
                                else gamma)
                      for variable in variables]

        rng = np.random.RandomState(20260904)
        config = tf.ConfigProto(intra_op_parallelism_threads=4,
                                inter_op_parallelism_threads=2,
                                device_count={'GPU': 0})
        with tf.Session(graph=graph, config=config) as session:
            session.run(assign)
            for data in (
                    rng.normal(size=(1, 32, 256)).astype(np.float32),
                    rng.normal(size=(7, 32, 256)).astype(np.float32),
                    np.ones((3, 32, 256), np.float32)):
                left, right, left_pool, right_pool = session.run(
                    [expected, actual, expected_pool, actual_pool],
                    {inputs: data},
                )
                np.testing.assert_array_equal(left, right)
                np.testing.assert_array_equal(left_pool, right_pool)
                self.assertTrue(np.all(np.isfinite(right)))


if __name__ == '__main__':
    unittest.main()
