import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[4]
REFERENCE = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2.py'
SMALL = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small.py'
VERIFY = ROOT / 'src/models/rankmixer/tools/verify_rankmixer_v6_e2_small.py'
spec = importlib.util.spec_from_file_location('verify_rankmixer_v6_e2_small', VERIFY)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def method_map(model_class):
    return {node.name: node for node in model_class.body if isinstance(node, ast.FunctionDef)}


def literal_attribute(model_class, name):
    return next(ast.literal_eval(node.value) for node in model_class.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


class CosmeticNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value=node.value.replace(
                'RankMixer v6-E2-Small', 'RankMixer v6-E2')), node)
        return node


def normalized(node):
    return ast.dump(CosmeticNormalizer().visit(copy.deepcopy(node)), include_attributes=False)


class SmallArchitectureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = verifier.model_class_ast(REFERENCE)
        cls.small = verifier.model_class_ast(SMALL)
        cls.reference_methods = method_map(cls.reference)
        cls.small_methods = method_map(cls.small)

    def test_only_tokenization_and_constructor_change(self):
        self.assertEqual(set(self.reference_methods), set(self.small_methods))
        changed = {'__init__', '_semantic_tokenize'}
        unchanged = set(self.reference_methods) - changed
        self.assertGreater(len(unchanged), 35)
        for name in sorted(unchanged):
            self.assertEqual(normalized(self.reference_methods[name]),
                             normalized(self.small_methods[name]), name)
        # Includes forward core, residuals, BN/SENet, loss, LR, optimizer,
        # dataset mode/sampling, evaluation, export and checkpoint hooks.
        self.assertTrue({'_rm_block', '_rm_per_token_swiglu', 'model_fn',
                         'build_loss_op', 'build_optimizer_op', 'get_dataset',
                         'train', 'test', '_build_export'}.issubset(unchanged))

    def test_server_imports_and_model_independence(self):
        def imports(path):
            return [ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports(REFERENCE), imports(SMALL))
        self.assertEqual([ast.dump(base) for base in self.small.bases], ['Name(id=\'ModelBase\', ctx=Load())'])
        self.assertNotIn('from .cvr_bn_rankmixer', SMALL.read_text(encoding='utf-8'))

    def test_width_guard_and_parameter_total(self):
        self.assertEqual(literal_attribute(self.small, '_EXPECTED_DENSE_TRAINABLE_PARAMS'), 102356069)
        def required_dimensions(method):
            node = next(node for node in ast.walk(method)
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'required_architecture'
                                for target in node.targets))
            return {ast.literal_eval(key): ast.literal_eval(value.elts[1])
                    for key, value in zip(node.value.keys, node.value.values)}
        expected = required_dimensions(self.reference_methods['__init__'])
        expected['rm_hidden_dim'] = 256
        self.assertEqual(required_dimensions(self.small_methods['__init__']), expected)
        assignments = {target.attr: node.value for node in ast.walk(self.small_methods['__init__'])
                       if isinstance(node, ast.Assign) for target in node.targets
                       if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                       and target.value.id == 'self'}
        self.assertEqual(ast.literal_eval(assignments['rm_hidden_dim'].args[0].args[1]), 256)
        self.assertIs(ast.literal_eval(assignments['rm_optimize_tokenize'].args[1]), True)
        model = verifier.load_kernel_model(SMALL)
        self.assertEqual(model._calculate_dense_trainable_params(), 102356069)
        self.assertEqual(256 ** 2 + 378181 * 256 + 5476197, 102356069)
        self.assertEqual(model.rm_swiglu_hidden_dim, 704)
        self.assertEqual(model.cvr_layers, [2048, 2048, 256])

    def test_frozen_feature_and_token_abi(self):
        for name in ('_BUCKET_NAMES', '_EXPECTED_FIELD_COUNTS', '_GROUP_VERSION',
                     '_GROUP_CHECKSUMS', '_GROUP_SIZES'):
            self.assertEqual(literal_attribute(self.reference, name), literal_attribute(self.small, name))
        model = verifier.load_kernel_model(SMALL)
        groups = model._build_semantic_feature_groups()
        sizes = model._GROUP_SIZES
        all_ids = []
        for bucket in model._BUCKET_NAMES:
            self.assertEqual(tuple(len(fields) for _, fields in groups[bucket]), sizes[bucket])
            ids = [field for _, fields in groups[bucket] for field in fields]
            self.assertEqual(hashlib.sha256('|'.join(ids).encode('utf-8')).hexdigest(),
                             model._GROUP_CHECKSUMS[bucket])
            all_ids.extend(ids)
        self.assertEqual(len(all_ids), 1234)
        self.assertEqual(len(set(all_ids)), 1234)

    def test_projection_variables_are_unchanged(self):
        def variable_calls(method):
            return [normalized(node) for node in ast.walk(method)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'get_variable']
        self.assertEqual(variable_calls(self.reference_methods['_project_token_family']),
                         variable_calls(self.small_methods['_project_token_family']))

    def test_server_args_change_only_width_and_tokenization_switch(self):
        def parse(path):
            lines = path.read_text(encoding='utf-8').splitlines()
            model_line = next(line for line in lines if line.startswith("--model_args='"))
            return lines[0], json.loads(model_line[len("--model_args='"):-1]), [
                line for line in lines[1:] if not line.startswith("--model_args='")]
        _, reference_args, reference_outer = parse(ROOT / 'bash/set-rankmixer-v6-e2-args.txt')
        entry, small_args, small_outer = parse(ROOT / 'bash/set-rankmixer-v6-e2-small-args.txt')
        self.assertEqual(entry, 'models.rankmixer.cvr_bn_rankmixer_v6_e2_small.MLPModel')
        self.assertEqual(reference_outer, small_outer)
        self.assertEqual(reference_args.pop('rm_hidden_dim'), 512)
        self.assertEqual(small_args.pop('rm_hidden_dim'), 256)
        self.assertIs(small_args.pop('rm_optimize_tokenize'), True)
        self.assertEqual(reference_args, small_args)
        self.assertIn('--ignore_dense_checkpoint=True', small_outer)


@unittest.skipUnless(importlib.util.find_spec('tensorflow') is not None,
                     'Real TensorFlow runtime is required for numerical checks')
class SmallNumericalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        cls.tf, cls.version = verifier.tensorflow_v1()
        cls.config = cls.tf.ConfigProto(intra_op_parallelism_threads=32,
                                        inter_op_parallelism_threads=8,
                                        device_count={'GPU': 0})

    def test_strict_outputs_gradients_updates_and_restore(self):
        for partitions in (1, 3):
            with self.subTest(partitions=partitions):
                report = verifier.verify_partition(self.tf, self.np, self.config, partitions, True)
                self.assertTrue(report['strict_equal'])
                self.assertEqual(report['reference_forward_transposes'], 10)
                self.assertEqual(report['candidate_forward_transposes'], 10)
                self.assertEqual(report['reference_slice_gradients'], 31)
                self.assertEqual(report['candidate_slice_gradients'], 0)

    def test_reference_switch_restores_original_execution(self):
        report = verifier.verify_partition(self.tf, self.np, self.config, 3, False)
        self.assertTrue(report['strict_equal'])
        self.assertEqual(report['candidate_forward_transposes'], 10)
        self.assertEqual(report['candidate_slice_gradients'], 31)


if __name__ == '__main__':
    unittest.main()
