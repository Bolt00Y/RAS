"""Offline regression checks for the original Small model's 3-block variant.

Run with unittest; TensorFlow, Flood and training data are not required.
The source constructor and parameter accounting execute unchanged with only
the external feature builder and configuration lookup replaced by fixtures.
"""

import ast
import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[4]
SMALL = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small.py'
SMALL_3 = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small_3.py'
ARGS = ROOT / 'bash/set-rankmixer-v6-e2-small-args.txt'
ARGS_3 = ROOT / 'bash/set-rankmixer-v6-e2-small-3-args.txt'


def model_ast(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'MLPModel')


def method_map(model):
    return {node.name: node for node in model.body
            if isinstance(node, ast.FunctionDef)}


def class_values(model):
    return {target.id: ast.literal_eval(node.value)
            for node in model.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)}


def parse_args(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    model_line = next(line for line in lines
                      if line.startswith("--model_args='"))
    return (lines[0], json.loads(model_line[len("--model_args='"):-1]),
            [line for line in lines[1:]
             if not line.startswith('--model_args=')])


class CosmeticNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace(
                'RankMixer v6-E2-Small-3', 'RankMixer v6-E2-Small')
        return node

    def visit_Str(self, node):
        node.s = node.s.replace(
            'RankMixer v6-E2-Small-3', 'RankMixer v6-E2-Small')
        return node


class DepthNormalizer(CosmeticNormalizer):
    """Allow only the constructor's depth default and required depth to differ."""

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        target = node.targets[0]
        if (isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == 'self'
                and target.attr == 'rm_layer_num'
                and isinstance(node.value, ast.Call)
                and node.value.args):
            get_call = node.value.args[0]
            if (isinstance(get_call, ast.Call)
                    and isinstance(get_call.func, ast.Attribute)
                    and get_call.func.attr == 'get'
                    and len(get_call.args) == 2
                    and ast.literal_eval(get_call.args[0]) == 'rm_layer_num'):
                get_call.args[1] = ast.Constant(value=2)
        if (isinstance(target, ast.Name)
                and target.id == 'required_architecture'
                and isinstance(node.value, ast.Dict)):
            for key, value in zip(node.value.keys, node.value.values):
                if ast.literal_eval(key) == 'rm_layer_num':
                    value.elts[1] = ast.Constant(value=2)
        return node


def normalized(node, constructor=False):
    normalizer = DepthNormalizer() if constructor else CosmeticNormalizer()
    return ast.dump(normalizer.visit(copy.deepcopy(node)),
                    include_attributes=False)


def load_offline_model_class(path):
    """Execute the actual constructor, guards, counter and stack without imports."""
    model = copy.deepcopy(model_ast(path))
    keep_methods = {
        '__init__', '_build_semantic_feature_groups',
        '_validate_semantic_feature_groups',
        '_calculate_dense_trainable_params', '_rm_stack',
    }
    model.body = [node for node in model.body
                  if isinstance(node, ast.Assign)
                  or (isinstance(node, ast.FunctionDef)
                      and node.name in keep_methods)]
    module = ast.Module(body=[model])
    if 'type_ignores' in ast.Module._fields:
        module.type_ignores = []
    namespace = {
        'ModelBase': object, 'hashlib': hashlib, 'logging': logging,
        'FeatureColumnBuilder': lambda **kwargs: None,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    model_class = namespace['MLPModel']
    groups = model_class._build_semantic_feature_groups()
    feature_maps = {
        bucket + '_fea_map': {field: None for _, fields in bucket_groups
                             for field in fields}
        for bucket, bucket_groups in groups.items()
    }
    feature_maps.update({name + '_fea_map': {}
                         for name in ('coupon', 'dense', 'seq', 'gattr', 'din')})
    config_module = SimpleNamespace(
        FeatureConfig=lambda: SimpleNamespace(**feature_maps))
    namespace['locate'] = lambda _: config_module
    return model_class


class Small3DepthAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small = model_ast(SMALL)
        cls.small_3 = model_ast(SMALL_3)
        cls.methods = method_map(cls.small)
        cls.methods_3 = method_map(cls.small_3)
        cls.offline_small = load_offline_model_class(SMALL)
        cls.offline_small_3 = load_offline_model_class(SMALL_3)

    def build_model(self, variant=True, **overrides):
        _, args, _ = parse_args(ARGS_3 if variant else ARGS)
        # Omit the bash depth so this also exercises the production default.
        args.pop('rm_layer_num')
        args.update(tf_config={'task': {'index': 0},
                               'cluster': {'ps': ['ps'], 'worker': ['worker']}},
                    log_gflags=False)
        args.update(overrides)
        model_class = self.offline_small_3 if variant else self.offline_small
        return model_class(**args)

    def test_standalone_imports_and_base_class_are_preserved(self):
        def imports(path):
            return [ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(node, (ast.Import, ast.ImportFrom))]

        self.assertEqual(imports(SMALL), imports(SMALL_3))
        self.assertEqual([ast.dump(base) for base in self.small_3.bases],
                         ["Name(id='ModelBase', ctx=Load())"])

    def test_all_algorithm_methods_preserve_small_except_depth(self):
        self.assertEqual(set(self.methods), set(self.methods_3))
        # Covers final RMSNorm, PureFlat, original task head, training, export,
        # tokenization, feature maps and checkpoint hooks as well as each block.
        for name in sorted(self.methods):
            with self.subTest(method=name):
                self.assertEqual(
                    normalized(self.methods[name], constructor=name == '__init__'),
                    normalized(self.methods_3[name], constructor=name == '__init__'),
                )

    def test_class_contract_only_changes_parameter_total(self):
        original = class_values(self.small)
        variant = class_values(self.small_3)
        self.assertEqual(original.pop('_EXPECTED_DENSE_TRAINABLE_PARAMS'), 102356069)
        self.assertEqual(variant.pop('_EXPECTED_DENSE_TRAINABLE_PARAMS'), 137081957)
        self.assertEqual(original, variant)

    def test_constructor_defaults_and_guards(self):
        model = self.build_model()
        self.assertEqual(model.rm_layer_num, 3)
        self.assertEqual(self.build_model(rm_layer_num=3).rm_layer_num, 3)
        self.assertEqual(model.cvr_layers, [2048, 2048, 256])
        self.assertEqual(model.rm_norm_type, 'rms_norm')
        self.assertEqual(model.rm_readout_type, 'pure_flat')
        self.assertIs(model.rm_optimize_tokenize, True)
        for invalid_depth in (2, 4):
            with self.subTest(depth=invalid_depth):
                with self.assertRaisesRegex(ValueError, 'requires rm_layer_num=3'):
                    self.build_model(rm_layer_num=invalid_depth)
        for overrides, error in (
                ({'cvr_layers': [256, 128]}, 'requires cvr_layers='),
                ({'rm_norm_type': 'layer_norm'}, 'requires rm_norm_type=rms_norm'),
                ({'rm_readout_type': 'mean_pool'}, 'requires rm_readout_type=pure_flat')):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, error):
                    self.build_model(**overrides)

    def test_stack_executes_three_distinct_indices_in_sequence(self):
        model = self.build_model()
        calls = []

        def block(inputs, block_idx, export):
            calls.append((inputs, block_idx, export))
            return inputs + (block_idx,)

        model._rm_block = block
        self.assertEqual(model._rm_stack((), export='probe'), (0, 1, 2))
        self.assertEqual(calls, [
            ((), 0, 'probe'), ((0,), 1, 'probe'), ((0, 1), 2, 'probe'),
        ])

    def test_parameter_budget_adds_exactly_one_block(self):
        original = self.build_model(variant=False)
        variant = self.build_model()
        one_swiglu = 32 * (2 * (256 * 704 + 704) + 704 * 256 + 256)
        one_block = 2 * 32 * 256 + 2 * one_swiglu
        self.assertEqual(one_block, 34725888)
        self.assertEqual(original._calculate_dense_trainable_params(), 102356069)
        self.assertEqual(variant._calculate_dense_trainable_params(), 137081957)
        self.assertEqual(variant.rm_dense_trainable_param_count,
                         original.rm_dense_trainable_param_count + one_block)
        self.assertEqual(variant.rm_dense_trainable_param_count,
                         variant._EXPECTED_DENSE_TRAINABLE_PARAMS)

    def test_bash_changes_only_model_entry_and_layer_count(self):
        entry, args, outer = parse_args(ARGS)
        entry_3, args_3, outer_3 = parse_args(ARGS_3)
        self.assertEqual(entry, 'models.rankmixer.cvr_bn_rankmixer_v6_e2_small.MLPModel')
        self.assertEqual(entry_3, 'models.rankmixer.cvr_bn_rankmixer_v6_e2_small_3.MLPModel')
        self.assertEqual(args.pop('rm_layer_num'), 2)
        self.assertEqual(args_3.pop('rm_layer_num'), 3)
        self.assertEqual(args, args_3)
        self.assertEqual(outer, outer_3)
        self.assertIn('--ignore_dense_checkpoint=True', outer_3)


if __name__ == '__main__':
    unittest.main()
