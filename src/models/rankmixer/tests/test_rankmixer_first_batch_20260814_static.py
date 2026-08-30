import ast
import copy
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
E2_MODEL_PATH = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2.py'
E3_MODEL_PATH = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e3.py'
V6_MODEL_PATH = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6.py'
BATCH_DIR = ROOT / 'bash/rankmixer_first_batch_20260814'
MANIFEST_PATH = BATCH_DIR / 'manifest.json'
DESIGN_PATH = ROOT / 'introduce/rankmixer_v6_e2_e3_ablation_design_20260814.md'
CONFIG_PATHS = {
    'E0_BASE': BATCH_DIR / '00-e0-base-args.txt',
    'E1_V6': BATCH_DIR / '01-e1-v6-args.txt',
    'E2_TML_FLAT_RMS': BATCH_DIR / '02-e2-tml-flat-rms-args.txt',
    'E3_FLAT_LN': BATCH_DIR / '03-e3-flat-ln-args.txt',
}
TOP_LEVEL_CONFIG_PATHS = {
    'E2_TML_FLAT_RMS': ROOT / 'bash/set-rankmixer-v6-e2-args.txt',
    'E3_FLAT_LN': ROOT / 'bash/set-rankmixer-v6-e3-args.txt',
}


def _model_ast(path):
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    model_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'MLPModel'
    )
    return source, tree, model_class


def _method(model_class, name):
    return next(
        node for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _method_names(model_class):
    return {
        node.name for node in model_class.body
        if isinstance(node, ast.FunctionDef)
    }


def _class_literal(model_class, name):
    assignment = next(
        node for node in model_class.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    )
    return ast.literal_eval(assignment.value)


def _semantic_groups(model_class):
    method = _method(model_class, '_build_semantic_feature_groups')
    return_node = next(
        node for node in ast.walk(method)
        if isinstance(node, ast.Return)
    )
    return ast.literal_eval(return_node.value)


def _load_config(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    model_args_index = next(
        index for index, line in enumerate(lines)
        if line.startswith("--model_args='")
    )
    model_args_line = lines[model_args_index]
    model_args = json.loads(model_args_line[len("--model_args='"):-1])
    outer_args = {}
    for line in lines[1:]:
        if not line.startswith('--') or line.startswith("--model_args='"):
            continue
        key, value = line[2:].split('=', 1)
        outer_args[key] = value
    return {
        'module': lines[0],
        'lines': lines,
        'model_args_index': model_args_index,
        'model_args': model_args,
        'outer_args': outer_args,
    }


def _parameter_total(norm_type):
    common_fields, item_fields, creative_fields = 385, 835, 14
    total_fields = common_fields + item_fields + creative_fields
    input_dim = total_fields * 17
    token_num, local_tokens, hidden_dim = 32, 31, 512
    swiglu_hidden, mixer_layers = 704, 2

    input_bn = 2 * input_dim
    senet = (
        common_fields * 128 + 128 * common_fields
        + (common_fields + item_fields) * 128 + 128 * item_fields
        + total_fields * 128 + 128 * creative_fields
        + 3 * 2 * 128
    )
    if norm_type == 'rms_norm':
        local_norm = local_tokens * hidden_dim
        global_norm = hidden_dim
        one_block_norm = 2 * token_num * hidden_dim
        final_norm = token_num * hidden_dim
    else:
        local_norm = 2 * hidden_dim
        global_norm = 2 * hidden_dim
        one_block_norm = 4 * hidden_dim
        final_norm = 2 * hidden_dim

    local_projection = input_dim * hidden_dim + local_tokens * hidden_dim + local_norm
    global_projection = (
        input_dim * hidden_dim + hidden_dim
        + hidden_dim * hidden_dim + hidden_dim
        + global_norm
    )
    one_swiglu = token_num * (
        hidden_dim * swiglu_hidden + swiglu_hidden
        + hidden_dim * swiglu_hidden + swiglu_hidden
        + swiglu_hidden * hidden_dim + hidden_dim
    )
    mixer = mixer_layers * (one_block_norm + 2 * one_swiglu)

    task_head = 0
    previous = token_num * hidden_dim
    for width in (2048, 2048, 256):
        task_head += previous * width + width + 2 * width
        previous = width
    task_head += previous + 1

    return sum([
        input_bn,
        senet,
        local_projection,
        global_projection,
        mixer,
        final_norm,
        task_head,
    ])


def _extended_flops(norm_type):
    input_dim = 20978
    token_num, local_tokens, hidden_dim = 32, 31, 512
    swiglu_hidden, mixer_layers = 704, 2
    norm_flops = (4 if norm_type == 'rms_norm' else 8) * hidden_dim + 2

    input_bn = 4 * input_dim
    senet = 1087414
    local_tokenizer = (
        2 * input_dim * hidden_dim
        + 9 * local_tokens * hidden_dim
        + local_tokens * norm_flops
    )
    global_token = (
        2 * input_dim * hidden_dim
        + 2 * hidden_dim * hidden_dim
        + 9 * hidden_dim
        + norm_flops
    )
    one_stage = token_num * (
        6 * hidden_dim * swiglu_hidden
        + 3 * swiglu_hidden
        + norm_flops
    )
    mixer = mixer_layers * (2 * one_stage + 2 * token_num * hidden_dim)
    final_norm = token_num * norm_flops
    task_head = (
        2 * (token_num * hidden_dim) * 2048 + 4 * 2048 + 9 * 2048
        + 2 * 2048 * 2048 + 4 * 2048 + 9 * 2048
        + 2 * 2048 * 256 + 4 * 256 + 9 * 256
        + 2 * 256 + 1
    )
    return sum([
        input_bn,
        senet,
        local_tokenizer,
        global_token,
        mixer,
        final_norm,
        task_head,
    ])


class _CosmeticStringNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace('RankMixer v6-E2', 'RankMixer v6-EX')
            value = value.replace('RankMixer v6-E3', 'RankMixer v6-EX')
            return ast.copy_location(ast.Constant(value=value), node)
        return node


class _InitNormTargetNormalizer(_CosmeticStringNormalizer):
    def visit_Constant(self, node):
        node = super().visit_Constant(node)
        if isinstance(node.value, str):
            value = node.value
            value = value.replace('rm_norm_type=rms_norm', 'rm_norm_type=TARGET_NORM')
            value = value.replace('rm_norm_type=layer_norm', 'rm_norm_type=TARGET_NORM')
            if value in ('rms_norm', 'layer_norm'):
                value = 'TARGET_NORM'
            return ast.copy_location(ast.Constant(value=value), node)
        return node


def _normalized_method_dump(model_class, name):
    node = copy.deepcopy(_method(model_class, name))
    node = _CosmeticStringNormalizer().visit(node)
    ast.fix_missing_locations(node)
    return ast.dump(node, include_attributes=False)


class RankMixerFirstBatchStaticTest(unittest.TestCase):
    def test_models_are_complete_independent_modules(self):
        required_lifecycle_methods = {
            'get_features_conf',
            'get_share_embedding_conf',
            'get_dataset',
            'build_loss_op',
            'build_optimizer_op',
            'build',
            'train',
            'test',
            'predict',
            'export',
            'evaluate',
            'get_hooks',
            'model_fn',
        }
        for path in (E2_MODEL_PATH, E3_MODEL_PATH):
            _, tree, model_class = _model_ast(path)
            self.assertEqual(
                [base.id for base in model_class.bases if isinstance(base, ast.Name)],
                ['ModelBase'],
            )
            self.assertTrue(required_lifecycle_methods.issubset(_method_names(model_class)))
            forbidden_model_imports = [
                node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module
                and 'cvr_bn_rankmixer' in node.module
            ]
            self.assertEqual(forbidden_model_imports, [])

    def test_v6_semantic_abi_is_identical_in_both_models(self):
        _, _, v6_class = _model_ast(V6_MODEL_PATH)
        expected_groups = _semantic_groups(v6_class)
        expected_sizes = _class_literal(v6_class, '_GROUP_SIZES')
        expected_checksums = _class_literal(v6_class, '_GROUP_CHECKSUMS')

        for path in (E2_MODEL_PATH, E3_MODEL_PATH):
            _, _, model_class = _model_ast(path)
            self.assertEqual(
                _class_literal(model_class, '_GROUP_VERSION'),
                'rankmixer_v6_semantic_balanced_v1',
            )
            self.assertEqual(_semantic_groups(model_class), expected_groups)
            self.assertEqual(_class_literal(model_class, '_GROUP_SIZES'), expected_sizes)
            self.assertEqual(
                _class_literal(model_class, '_GROUP_CHECKSUMS'),
                expected_checksums,
            )

        all_ids = []
        for bucket_name in ('common', 'item', 'creative'):
            ordered_ids = [
                field_id
                for _, field_ids in expected_groups[bucket_name]
                for field_id in field_ids
            ]
            self.assertEqual(
                hashlib.sha256('|'.join(ordered_ids).encode('utf-8')).hexdigest(),
                expected_checksums[bucket_name],
            )
            all_ids.extend(ordered_ids)
        self.assertEqual(len(all_ids), 1234)
        self.assertEqual(len(set(all_ids)), 1234)

    def test_exact_parameter_and_flops_budgets(self):
        _, _, e2_class = _model_ast(E2_MODEL_PATH)
        _, _, e3_class = _model_ast(E3_MODEL_PATH)
        self.assertEqual(
            _class_literal(e2_class, '_EXPECTED_DENSE_TRAINABLE_PARAMS'),
            199367013,
        )
        self.assertEqual(
            _class_literal(e3_class, '_EXPECTED_DENSE_TRAINABLE_PARAMS'),
            199275877,
        )
        self.assertEqual(_parameter_total('rms_norm'), 199367013)
        self.assertEqual(_parameter_total('layer_norm'), 199275877)
        self.assertEqual(
            _parameter_total('layer_norm') - _parameter_total('rms_norm'),
            -91136,
        )
        self.assertEqual(_extended_flops('rms_norm'), 398962687)
        self.assertEqual(_extended_flops('layer_norm'), 399355903)

    def test_norm_paths_are_fixed_and_cannot_be_misconfigured(self):
        e2_source, _, e2_class = _model_ast(E2_MODEL_PATH)
        e3_source, _, e3_class = _model_ast(E3_MODEL_PATH)
        e2_methods = _method_names(e2_class)
        e3_methods = _method_names(e3_class)

        self.assertIn('_rm_rms_norm', e2_methods)
        self.assertNotIn('_rm_rms_norm', e3_methods)
        self.assertNotIn('from .model_utils import layer_norm', e2_source)
        self.assertIn('from .model_utils import layer_norm', e3_source)
        self.assertIn('tf.rsqrt', e2_source)
        self.assertNotIn('tf.rsqrt', e3_source)

        e2_init = ast.get_source_segment(e2_source, _method(e2_class, '__init__'))
        e3_init = ast.get_source_segment(e3_source, _method(e3_class, '__init__'))
        self.assertIn("self.rm_norm_type != 'rms_norm'", e2_init)
        self.assertIn("self.rm_norm_type != 'layer_norm'", e3_init)

        e2_norm = ast.get_source_segment(e2_source, _method(e2_class, '_rm_norm'))
        e3_norm = ast.get_source_segment(e3_source, _method(e3_class, '_rm_norm'))
        self.assertIn('return self._rm_rms_norm(', e2_norm)
        self.assertNotIn('layer_norm(', e2_norm)
        self.assertIn('return layer_norm(', e3_norm)
        self.assertNotIn('_rm_rms_norm(', e3_norm)

        for source, model_class in ((e2_source, e2_class), (e3_source, e3_class)):
            structural_source = '\n'.join(
                ast.get_source_segment(source, _method(model_class, name))
                for name in ('_semantic_tokenize', '_build_global_token', '_rm_block', 'model_fn')
            )
            self.assertEqual(structural_source.count('self._rm_norm('), 5)

    def test_every_non_norm_method_ast_is_equivalent(self):
        _, _, e2_class = _model_ast(E2_MODEL_PATH)
        _, _, e3_class = _model_ast(E3_MODEL_PATH)
        e2_methods = _method_names(e2_class)
        e3_methods = _method_names(e3_class)
        intentionally_different = {
            '__init__',
            '_calculate_dense_trainable_params',
            '_rm_norm',
        }
        invariant_methods = sorted(
            (e2_methods & e3_methods) - intentionally_different
        )
        self.assertGreater(len(invariant_methods), 35)
        for method_name in invariant_methods:
            self.assertEqual(
                _normalized_method_dump(e2_class, method_name),
                _normalized_method_dump(e3_class, method_name),
                method_name,
            )

        e2_init = copy.deepcopy(_method(e2_class, '__init__'))
        e3_init = copy.deepcopy(_method(e3_class, '__init__'))
        e2_init = _InitNormTargetNormalizer().visit(e2_init)
        e3_init = _InitNormTargetNormalizer().visit(e3_init)
        ast.fix_missing_locations(e2_init)
        ast.fix_missing_locations(e3_init)
        self.assertEqual(
            ast.dump(e2_init, include_attributes=False),
            ast.dump(e3_init, include_attributes=False),
            '__init__ differs beyond the fixed RMSNorm/LayerNorm target',
        )

    def test_pure_flat_single_readout_and_head_width(self):
        for path in (E2_MODEL_PATH, E3_MODEL_PATH):
            source, _, model_class = _model_ast(path)
            method_names = _method_names(model_class)
            self.assertNotIn('_global_conditioned_pool', method_names)
            self.assertNotIn('_flatten_readout', method_names)
            model_fn_source = ast.get_source_segment(
                source,
                _method(model_class, 'model_fn'),
            )
            self.assertIn(
                'context_dim = self.rm_token_num * self.rm_hidden_dim',
                model_fn_source,
            )
            self.assertIn("name='rm_pure_flatten'", model_fn_source)
            self.assertNotIn('pool_weights', model_fn_source)
            self.assertNotIn('flatten_gate', model_fn_source)
            init_source = ast.get_source_segment(
                source,
                _method(model_class, '__init__'),
            )
            self.assertIn('self.cvr_layers != [2048, 2048, 256]', init_source)
            self.assertNotIn('_DENSE_TRAINABLE_PARAM_LIMIT', source)

            task_head_source = ast.get_source_segment(
                source,
                _method(model_class, '_task_head'),
            )
            for v6_scope in ('rm_v5_mlp', 'rm_v5_bn_', 'rm_v5_out'):
                self.assertIn(v6_scope, task_head_source)
            self.assertNotIn('rm_v6_ablation_', task_head_source)
            self.assertIn("scope='rm_final_rms_norm'", model_fn_source)

    def test_e2_e3_server_args_have_only_intended_differences(self):
        e2 = _load_config(CONFIG_PATHS['E2_TML_FLAT_RMS'])
        e3 = _load_config(CONFIG_PATHS['E3_FLAT_LN'])
        self.assertEqual(
            e2['module'],
            'models.rankmixer.cvr_bn_rankmixer_v6_e2.MLPModel',
        )
        self.assertEqual(
            e3['module'],
            'models.rankmixer.cvr_bn_rankmixer_v6_e3.MLPModel',
        )
        self.assertEqual(e2['outer_args'], e3['outer_args'])

        e2_args = dict(e2['model_args'])
        e3_args = dict(e3['model_args'])
        self.assertEqual(e2_args.pop('rm_norm_type'), 'rms_norm')
        self.assertEqual(e3_args.pop('rm_norm_type'), 'layer_norm')
        self.assertEqual(e2_args, e3_args)

        self.assertEqual(
            TOP_LEVEL_CONFIG_PATHS['E2_TML_FLAT_RMS'].read_bytes(),
            CONFIG_PATHS['E2_TML_FLAT_RMS'].read_bytes(),
        )
        self.assertEqual(
            TOP_LEVEL_CONFIG_PATHS['E3_FLAT_LN'].read_bytes(),
            CONFIG_PATHS['E3_FLAT_LN'].read_bytes(),
        )

    def test_all_four_jobs_share_outer_data_controls(self):
        configs = {
            experiment_id: _load_config(path)
            for experiment_id, path in CONFIG_PATHS.items()
        }
        reference_outer = configs['E0_BASE']['outer_args']
        for experiment_id, config in configs.items():
            self.assertEqual(config['outer_args'], reference_outer, experiment_id)
            self.assertEqual(
                config['outer_args']['train_dates'],
                '2026-08-14:2026-08-14',
            )
            self.assertEqual(
                config['outer_args']['test_date'],
                '2026-08-15:2026-08-15',
            )
            self.assertEqual(
                config['outer_args']['additional_checkpoint_dates'],
                '2026-08-13:2026-08-13',
            )
            self.assertEqual(config['outer_args']['ignore_dense_checkpoint'], 'True')
            self.assertEqual(config['outer_args']['ignore_sparse_checkpoint'], 'False')
            self.assertTrue(config['model_args']['save_predict_result'])

    def test_e1_e2_keep_the_same_v6_tokenmixer_core(self):
        e1 = _load_config(CONFIG_PATHS['E1_V6'])['model_args']
        e2 = _load_config(CONFIG_PATHS['E2_TML_FLAT_RMS'])['model_args']
        frozen_core_keys = (
            'use_rankmixer',
            'use_senet',
            'use_senet_bn',
            'senet_hidden_size',
            'rm_token_num',
            'rm_local_token_num',
            'rm_hidden_dim',
            'rm_layer_num',
            'rm_head_num',
            'rm_swiglu_hidden_dim',
            'rm_down_init_scale',
            'rm_rms_epsilon',
            'rm_token_proj_act',
            'rm_bucket_token_counts',
            'rm_group_version',
            'optimizer',
            'learning_rate',
            'batch_size',
            'embedding_size',
            'feature_version',
            'feature_version_old',
        )
        for key in frozen_core_keys:
            self.assertEqual(e1[key], e2[key], key)
        self.assertEqual(e1['cvr_layers'], [2048, 2048, 256])
        self.assertEqual(e2['cvr_layers'], [2048, 2048, 256])
        self.assertEqual(e2['rm_readout_type'], 'pure_flat')
        for removed_readout_arg in (
            'rm_pool_query_dim',
            'rm_flatten_dim',
            'rm_flatten_gate_init',
        ):
            self.assertIn(removed_readout_arg, e1)
            self.assertNotIn(removed_readout_arg, e2)

        _, _, v6_class = _model_ast(V6_MODEL_PATH)
        _, _, e2_class = _model_ast(E2_MODEL_PATH)
        _, _, e3_class = _model_ast(E3_MODEL_PATH)
        self.assertEqual(
            _normalized_method_dump(v6_class, '_task_head'),
            _normalized_method_dump(e2_class, '_task_head'),
        )
        self.assertEqual(
            _normalized_method_dump(e2_class, '_task_head'),
            _normalized_method_dump(e3_class, '_task_head'),
        )

    def test_manifest_and_introduce_design_match_implementation(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        experiments = {
            experiment['id']: experiment
            for experiment in manifest['experiments']
        }
        self.assertEqual(manifest['train_date'], '2026-08-14')
        self.assertEqual(manifest['test_date'], '2026-08-15')
        self.assertEqual(
            experiments['E2_TML_FLAT_RMS']['module'],
            'models.rankmixer.cvr_bn_rankmixer_v6_e2.MLPModel',
        )
        self.assertEqual(
            experiments['E3_FLAT_LN']['module'],
            'models.rankmixer.cvr_bn_rankmixer_v6_e3.MLPModel',
        )
        self.assertEqual(
            experiments['E2_TML_FLAT_RMS']['expected_dense_params'],
            199367013,
        )
        self.assertEqual(
            experiments['E3_FLAT_LN']['expected_dense_params'],
            199275877,
        )

        design = DESIGN_PATH.read_text(encoding='utf-8')
        for required in (
            '2026-08-14',
            '2026-08-15',
            'cvr_bn_rankmixer_v6_e2.py',
            'cvr_bn_rankmixer_v6_e3.py',
            '199,367,013',
            '199,275,877',
            'ignore_dense_checkpoint=True',
        ):
            self.assertIn(required, design)


if __name__ == '__main__':
    unittest.main()
