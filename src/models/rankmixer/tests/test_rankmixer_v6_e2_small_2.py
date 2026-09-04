"""Contracts for the Small-1 to Small-2 RankMixer-depth ablation."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[4]
SMALL_1 = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small_1.py'
SMALL_2 = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small_2.py'
VERIFY = ROOT / 'src/models/rankmixer/tools/verify_rankmixer_v6_e2_small.py'
ARGS_1 = ROOT / 'bash/set-rankmixer-v6-e2-small-1-args.txt'
ARGS_2 = ROOT / 'bash/set-rankmixer-v6-e2-small-2-args.txt'
INTRO = ROOT / 'introduce/rankmixer_v6_e2_small_2_introduction.md'


def model_ast(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'MLPModel')


def method_map(model):
    return {node.name: node for node in model.body
            if isinstance(node, ast.FunctionDef)}


def class_assignments(model):
    result = {}
    for node in model.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = node.value
    return result


def parse_args(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    model_args_line = next(line for line in lines
                           if line.startswith("--model_args='"))
    model_args = json.loads(model_args_line[len("--model_args='"):-1])
    outer_args = [line for line in lines[1:]
                  if not line.startswith('--model_args=')]
    return lines[0], model_args, outer_args


def module_node(body):
    module = ast.Module(body=body)
    if 'type_ignores' in ast.Module._fields:
        module.type_ignores = []
    return ast.fix_missing_locations(module)


class CosmeticNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace(
                'RankMixer v6-E2-Small-2', 'RankMixer v6-E2-Small-1')
        return node

    def visit_Str(self, node):
        node.s = node.s.replace(
            'RankMixer v6-E2-Small-2', 'RankMixer v6-E2-Small-1')
        return node


class LayerDepthNormalizer(CosmeticNormalizer):
    """Normalize only the two intended rm_layer_num literals in __init__."""

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == 'rm_layer_num'
                and isinstance(node.value, ast.Call)
                and node.value.args):
            get_call = node.value.args[0]
            if (isinstance(get_call, ast.Call)
                    and isinstance(get_call.func, ast.Attribute)
                    and get_call.func.attr == 'get'
                    and len(get_call.args) == 2
                    and ast.literal_eval(get_call.args[0]) == 'rm_layer_num'):
                get_call.args[1] = ast.copy_location(ast.Constant(value=2),
                                                     get_call.args[1])

        if (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'required_architecture'
                and isinstance(node.value, ast.Dict)):
            for key, value in zip(node.value.keys, node.value.values):
                if ast.literal_eval(key) == 'rm_layer_num':
                    value.elts[1] = ast.copy_location(ast.Constant(value=2),
                                                     value.elts[1])
        return node


def normalized(node, normalize_depth=False):
    transformer = LayerDepthNormalizer() if normalize_depth else CosmeticNormalizer()
    return ast.dump(transformer.visit(copy.deepcopy(node)),
                    include_attributes=False)


class Small2DepthAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_1 = model_ast(SMALL_1)
        cls.model_2 = model_ast(SMALL_2)
        cls.methods_1 = method_map(cls.model_1)
        cls.methods_2 = method_map(cls.model_2)

    def test_standalone_source_and_imports_match(self):
        def imports(path):
            return [ast.dump(node, include_attributes=False)
                    for node in ast.parse(
                        path.read_text(encoding='utf-8')).body
                    if isinstance(node, (ast.Import, ast.ImportFrom))]

        self.assertEqual(imports(SMALL_1), imports(SMALL_2))
        source = SMALL_2.read_text(encoding='utf-8')
        self.assertIn('由 cvr_bn_rankmixer_v6_e2_small_1.py 发展的三层主干消融版本',
                      source.splitlines()[1])
        self.assertNotIn('from .cvr_bn_rankmixer', source)

    def test_all_model_methods_match_except_depth_literals(self):
        self.assertEqual(set(self.methods_1), set(self.methods_2))
        for name in sorted(self.methods_1):
            self.assertEqual(
                normalized(self.methods_1[name], normalize_depth=name == '__init__'),
                normalized(self.methods_2[name], normalize_depth=name == '__init__'),
                name,
            )

    def test_class_contract_only_changes_expected_parameter_total(self):
        assignments_1 = class_assignments(self.model_1)
        assignments_2 = class_assignments(self.model_2)
        self.assertEqual(set(assignments_1), set(assignments_2))
        for name in sorted(assignments_1):
            left = ast.literal_eval(assignments_1[name])
            right = ast.literal_eval(assignments_2[name])
            if name == '_EXPECTED_DENSE_TRAINABLE_PARAMS':
                self.assertEqual(left, 80938853)
                self.assertEqual(right, 115664741)
            else:
                self.assertEqual(left, right, name)

    def test_stack_runs_three_independent_indices(self):
        stack_method = copy.deepcopy(self.methods_2['_rm_stack'])
        kernel = ast.ClassDef(
            name='Kernel',
            bases=[],
            keywords=[],
            body=[stack_method],
            decorator_list=[],
        )
        namespace = {}
        exec(compile(module_node([kernel]), str(SMALL_2), 'exec'), namespace)
        model = namespace['Kernel']()
        model.rm_layer_num = 3
        calls = []

        def block(inputs, block_idx, export):
            calls.append((block_idx, export))
            return inputs + block_idx + 1

        model._rm_block = block
        self.assertEqual(model._rm_stack(0, export='probe'), 6)
        self.assertEqual(calls, [(0, 'probe'), (1, 'probe'), (2, 'probe')])

    def test_exact_parameter_budget_adds_one_block(self):
        spec = importlib.util.spec_from_file_location('verify_small', VERIFY)
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)

        model_1 = verifier.load_kernel_model(SMALL_1)
        model_1.rm_layer_num = 2
        model_1.cvr_layers = [256, 128]
        total_1 = model_1._calculate_dense_trainable_params()

        model_2 = verifier.load_kernel_model(SMALL_2)
        model_2.rm_layer_num = 3
        model_2.cvr_layers = [256, 128]
        total_2 = model_2._calculate_dense_trainable_params()

        one_swiglu = 32 * (
            256 * 704 + 704
            + 256 * 704 + 704
            + 704 * 256 + 256
        )
        one_block = 2 * 32 * 256 + 2 * one_swiglu
        self.assertEqual(one_block, 34725888)
        self.assertEqual(total_1, 80938853)
        self.assertEqual(total_2, 115664741)
        self.assertEqual(total_2 - total_1, one_block)
        self.assertEqual(model_2._EXPECTED_DENSE_TRAINABLE_PARAMS, total_2)

    def test_bash_changes_only_entry_and_layer_count(self):
        entry_1, args_1, outer_1 = parse_args(ARGS_1)
        entry_2, args_2, outer_2 = parse_args(ARGS_2)
        self.assertEqual(
            entry_1,
            'models.rankmixer.cvr_bn_rankmixer_v6_e2_small_1.MLPModel',
        )
        self.assertEqual(
            entry_2,
            'models.rankmixer.cvr_bn_rankmixer_v6_e2_small_2.MLPModel',
        )
        self.assertEqual(outer_1, outer_2)
        self.assertEqual(args_1.pop('rm_layer_num'), 2)
        self.assertEqual(args_2.pop('rm_layer_num'), 3)
        self.assertEqual(args_1, args_2)
        self.assertIn('--ignore_dense_checkpoint=True', outer_2)

    def test_introduction_tracks_the_depth_ablation(self):
        document = INTRO.read_text(encoding='utf-8')
        for expected in (
                'rm_layer_num: 2 → 3',
                '34,725,888',
                '115,664,741',
                '256 → 128 → 1',
                'set-rankmixer-v6-e2-small-2-args.txt'):
            self.assertIn(expected, document)


if __name__ == '__main__':
    unittest.main()
