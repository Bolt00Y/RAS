import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEBUG_PATH = ROOT / 'src/models/rankmixer/cvr_senet_mature_rankmixer_debug.py'
MATURE_V1_PATH = ROOT / 'src/models/rankmixer/cvr_senet_mature_rankmixer_v1.py'
RANKMIXER_V10_PATH = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v10.py'
MATURE_PATHS = tuple(
    ROOT / 'src/models/rankmixer/cvr_senet_mature_rankmixer_v{}.py'.format(version)
    for version in (1, 2, 3)
)
MATURE_ARGS_PATHS = (
    ROOT / 'bash/set-rankmixer-mature-3bucket-d256-args.txt',
    ROOT / 'bash/set-rankmixer-mature-v2-args.txt',
    ROOT / 'bash/set-rankmixer-mature-v3-args.txt',
)
RANKMIXER_V10_ARGS_PATH = ROOT / 'bash/set-rankmixer-v10-args.txt'


ARCHITECTURE_METHODS = (
    '_gelu',
    '_rms_norm',
    '_layer_norm',
    '_dense',
    '_mix_up',
    '_add_weight',
    '_matmul_dense',
    '_per_token_swiglu',
    '_mlp_mixer_swiglu',
    '_mature_batch_norm',
    '_excitation2',
    '_embedding_to_tokens',
    '_bottom_embedding_to_global_token',
    '_creative_converter',
    '_task_head',
    '_concat_feature_ids',
    'model_fn',
)

PROVEN_RUNTIME_METHODS = (
    'get_dataset',
    'build',
    'build_dataset_op',
    'build_pred_results_op',
    'build_auc_copc_op',
    'build_summary',
    'build_optimizer_op',
    '_build_lr_schedule',
    '_schedule_lr',
    'get_optimizer',
    'train',
    'test',
    'predict',
    '_build_export',
    'export',
    'train_init',
    'evaluate',
    'list_all_member',
    'get_hooks',
)


def _parse(path):
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    model_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'MLPModel'
    )
    methods = {
        node.name: node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef)
    }
    return source, tree, model_class, methods


def _method_dump(methods, name):
    return ast.dump(methods[name], include_attributes=False)


def _class_literal(model_class, name):
    assignment = next(
        node for node in model_class.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def _module_id_groups(tree):
    groups = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.endswith('_IDS'):
            continue
        if not isinstance(node.value, ast.Call) or not node.value.args:
            continue
        groups[target.id] = tuple(ast.literal_eval(node.value.args[0]).split())
    return groups


class MatureRankMixerDebugStaticTest(unittest.TestCase):
    def test_algorithm_graph_is_identical_to_mature_v1(self):
        _, _, _, mature_methods = _parse(MATURE_V1_PATH)
        _, _, _, debug_methods = _parse(DEBUG_PATH)
        for method_name in ARCHITECTURE_METHODS:
            self.assertEqual(
                _method_dump(debug_methods, method_name),
                _method_dump(mature_methods, method_name),
                method_name,
            )

    def test_runtime_shell_is_identical_to_successful_v10(self):
        _, _, _, v10_methods = _parse(RANKMIXER_V10_PATH)
        _, _, _, debug_methods = _parse(DEBUG_PATH)
        for method_name in PROVEN_RUNTIME_METHODS:
            self.assertEqual(
                _method_dump(debug_methods, method_name),
                _method_dump(v10_methods, method_name),
                method_name,
            )

    def test_frozen_feature_order_and_parameter_budget(self):
        _, tree, model_class, _ = _parse(DEBUG_PATH)
        groups = _module_id_groups(tree)
        expected_sizes = {
            '_USER_V1_IDS': 102,
            '_USER_V2_IDS': 149,
            '_USER_V3_IDS': 134,
            '_ITEM_V1_IDS': 202,
            '_ITEM_V2_IDS': 203,
            '_ITEM_V3_IDS': 202,
            '_ITEM_V4_PLUS_IDS': 228,
            '_CREATIVE_IDS': 14,
        }
        self.assertEqual(
            {name: len(values) for name, values in groups.items()},
            expected_sizes,
        )
        ordered_names = (
            '_USER_V1_IDS', '_USER_V2_IDS', '_USER_V3_IDS',
            '_ITEM_V1_IDS', '_ITEM_V2_IDS', '_ITEM_V3_IDS',
            '_ITEM_V4_PLUS_IDS', '_CREATIVE_IDS',
        )
        all_ids = [feature_id for name in ordered_names for feature_id in groups[name]]
        self.assertEqual(len(all_ids), 1234)
        self.assertEqual(len(set(all_ids)), 1234)
        self.assertEqual(
            hashlib.sha256('\n'.join(all_ids).encode('utf-8')).hexdigest(),
            _class_literal(model_class, '_ALL_GROUPS_CHECKSUM'),
        )
        self.assertEqual(
            _class_literal(model_class, '_EXPECTED_DENSE_TRAINABLE_PARAMS'),
            109976671,
        )

    def test_server_runtime_attributes_are_retained(self):
        required = {
            'enable_dense_warmup', 'enable_mlt_warmup', 'hooks',
            'skip_tensors', 'warm_up_tensors', 'warmup_type',
            'warm_mlp_layer', 'use_mlp_gate', 'old_epoch_ckpt_import_dir',
            'ckpt_import_dir1', 'ckpt_import_dir2', 'warm_up_tensors1',
            'dense_tuning', 'dense_scale', 'dense_global_norm',
            'dense_clip_threshold', 'train_stage_param', 'enable_neg_sampler',
            'filter_pass_values', 'filter_label_names', 'filter_drop_values',
            'fq_table_config', 'seq_add_dim', 'dir2_all_tensor',
            'second_epoch_ckpt_import_dir', 'ffn_version', 'scale_type',
        }
        for model_path in (DEBUG_PATH,) + MATURE_PATHS:
            source, _, _, methods = _parse(model_path)
            init_method = methods['__init__']
            assigned_attributes = {
                node.attr
                for node in ast.walk(init_method)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == 'self'
                and isinstance(node.ctx, ast.Store)
            }
            self.assertTrue(
                required.issubset(assigned_attributes),
                str(model_path),
            )
            for forbidden in (
                    'new_rankmixer_0831', 'cayman.python',
                    'phalanx.tensorflow', 'attention_utils_new',
                    'mlp_mixer_swiglu_fuse_v4'):
                self.assertNotIn(forbidden, source, str(model_path))

    def test_formal_models_keep_proven_bn_and_dataset_mode_contract(self):
        for model_path in MATURE_PATHS:
            source, _, _, methods = _parse(model_path)
            bn_source = ast.get_source_segment(
                source, methods['_mature_batch_norm'])
            self.assertIn('ModelBase.batch_norm_layer_v2', bn_source)
            self.assertNotIn('phalanx', bn_source.lower())

            dataset_calls = [
                node for node in ast.walk(methods['build_dataset_op'])
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'self'
                and node.func.attr == 'get_dataset'
            ]
            self.assertEqual(len(dataset_calls), 1, str(model_path))
            self.assertGreaterEqual(len(dataset_calls[0].args), 2)
            self.assertIsInstance(dataset_calls[0].args[1], ast.Name)
            self.assertEqual(dataset_calls[0].args[1].id, 'flood_mode')

    def test_formal_args_keep_successful_v10_outer_runner_shell(self):
        def split_args(path):
            lines = path.read_text(encoding='utf-8').splitlines()
            model_args_line = next(
                line for line in lines if line.startswith('--model_args='))
            raw_model_args = model_args_line.split('=', 1)[1]
            self.assertEqual(raw_model_args[0], "'")
            self.assertEqual(raw_model_args[-1], "'")
            model_args = json.loads(raw_model_args[1:-1])
            outer_args = [
                line for index, line in enumerate(lines)
                if index != 0 and not line.startswith('--model_args=')
            ]
            return lines[0], model_args, outer_args

        _, _, v10_outer_args = split_args(RANKMIXER_V10_ARGS_PATH)
        for version, (model_path, args_path) in enumerate(
                zip(MATURE_PATHS, MATURE_ARGS_PATHS), 1):
            module_name, model_args, outer_args = split_args(args_path)
            self.assertEqual(outer_args, v10_outer_args, str(args_path))
            self.assertEqual(
                module_name,
                'models.rankmixer.cvr_senet_mature_rankmixer_v{}.MLPModel'.format(
                    version),
            )
            _, _, model_class, _ = _parse(model_path)
            self.assertEqual(
                model_args['runtime_build_id'],
                _class_literal(model_class, '_COMPAT_BUILD'),
            )
            self.assertNotIn('enable_phalanx', model_args)
            self.assertIn('--worker_numa_strategy=select', outer_args)
            self.assertFalse(any(
                line.startswith('--x_worker_numa_strategy=')
                for line in outer_args
            ))

    def test_single_head_adapter_keeps_mature_public_contract(self):
        source, _, _, methods = _parse(DEBUG_PATH)
        parse_source = ast.get_source_segment(source, methods['parse_examples'])
        loss_source = ast.get_source_segment(source, methods['build_loss_op'])
        model_source = ast.get_source_segment(source, methods['model_fn'])
        self.assertIn('features.pop(self.cvr_label_name)', parse_source)
        self.assertIn('self.loss_first = self.loss', loss_source)
        self.assertIn("'logits': logits", model_source)
        self.assertIn("'pred': predictions", model_source)
        for forbidden in (
                'sequence_attention', 'distill', 'replay_weight',
                'last_cvr', 'wide_cvr', 'delay_logits'):
            self.assertNotIn(forbidden, model_source)


if __name__ == '__main__':
    unittest.main()
