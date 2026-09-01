import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = ROOT / 'src/models/rankmixer/cvr_mature_rankmixer_fst_v1.py'
ALGORITHM_PATH = ROOT / 'src/models/rankmixer/cvr_senet_mature_rankmixer_v1.py'
RUNTIME_PATH = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v10.py'
FST_PATH = ROOT / 'src/models/seq_model/cvr_bn_senet_dcnm_fst.py'
ARGS_PATH = ROOT / 'bash/set-cvr-mature-rankmixer-fst-v1-args.txt'
RUNTIME_ARGS_PATH = ROOT / 'bash/set-rankmixer-v10-args.txt'


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


def _load_args(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    model_args_line = next(
        line for line in lines if line.startswith("--model_args='")
    )
    model_args = json.loads(model_args_line[len("--model_args='"):-1])
    outer_args = [
        line for index, line in enumerate(lines)
        if index != 0 and not line.startswith('--model_args=')
    ]
    return lines, model_args, outer_args


class CvrMatureRankMixerFstV1StaticTest(unittest.TestCase):
    def test_model_is_standalone(self):
        _, tree, _, _ = _parse(MODEL_PATH)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or '')

        forbidden_prefixes = (
            'new_rankmixer_0831',
            'models.rankmixer.cvr_senet_mature_rankmixer',
            'models.rankmixer.cvr_bn_rankmixer',
            'models.seq_model.cvr_bn_senet_dcnm_fst',
            'cayman',
            'phalanx',
        )
        for module_name in imported_modules:
            self.assertFalse(
                module_name.startswith(forbidden_prefixes),
                module_name,
            )

    def test_only_algorithm_graph_matches_mature_v1(self):
        _, _, _, algorithm_methods = _parse(ALGORITHM_PATH)
        _, _, _, model_methods = _parse(MODEL_PATH)
        for method_name in ARCHITECTURE_METHODS:
            self.assertEqual(
                _method_dump(model_methods, method_name),
                _method_dump(algorithm_methods, method_name),
                method_name,
            )

    def test_runtime_lifecycle_matches_successful_rankmixer_v10(self):
        _, _, _, runtime_methods = _parse(RUNTIME_PATH)
        _, _, _, model_methods = _parse(MODEL_PATH)
        for method_name in PROVEN_RUNTIME_METHODS:
            self.assertEqual(
                _method_dump(model_methods, method_name),
                _method_dump(runtime_methods, method_name),
                method_name,
            )

    def test_fst_utf8_schedule_increment_workaround_is_retained(self):
        model_source, model_tree, _, _ = _parse(MODEL_PATH)
        fst_source, _, _, _ = _parse(FST_PATH)
        popen_method = next(
            node for node in model_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == '_utf8_popen'
        )
        popen_source = ast.get_source_segment(model_source, popen_method)
        self.assertIn("encoding='utf-8'", popen_source)
        self.assertIn("errors='replace'", popen_source)
        self.assertIn('os._wrap_close', popen_source)
        self.assertIn('os.popen = _utf8_popen', model_source)
        self.assertIn('os.popen = _utf8_popen', fst_source)

    def test_single_fst_task_adapter(self):
        source, _, _, methods = _parse(MODEL_PATH)
        parse_source = ast.get_source_segment(source, methods['parse_examples'])
        loss_source = ast.get_source_segment(source, methods['build_loss_op'])
        model_source = ast.get_source_segment(source, methods['model_fn'])

        self.assertIn('features.pop(self.cvr_label_name)', parse_source)
        self.assertNotIn('last_cvr_label', parse_source)
        self.assertIn('self.loss_first = self.loss', loss_source)
        self.assertIn("'logits': logits", model_source)
        self.assertIn("'pred': predictions", model_source)
        for forbidden in (
                'sequence_attention', 'distill', 'replay_weight',
                'last_cvr', 'wide_cvr', 'delay_logits'):
            self.assertNotIn(forbidden, model_source)

    def test_frozen_feature_routing_and_parameter_budget(self):
        _, tree, model_class, _ = _parse(MODEL_PATH)
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

    def test_routing_exactly_matches_base_cold_feature_source(self):
        _, model_tree, _, _ = _parse(MODEL_PATH)
        groups = _module_id_groups(model_tree)
        feature_path = ROOT / 'src/data/cvr/cvr_fea_v10_base.py'
        feature_tree = ast.parse(feature_path.read_text(encoding='utf-8'))

        def ordered_dict_keys(assignment_name):
            assignment = next(
                node for node in feature_tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == assignment_name
                        for target in node.targets)
            )
            return set(ast.literal_eval(assignment.value.args[0]).keys())

        configured = {
            'common': ordered_dict_keys('user_features'),
            'item': ordered_dict_keys('item_features'),
            'creative': ordered_dict_keys('creative_features'),
        }
        routed = {
            'common': set(
                groups['_USER_V1_IDS']
                + groups['_USER_V2_IDS']
                + groups['_USER_V3_IDS']
            ),
            'item': set(
                groups['_ITEM_V1_IDS']
                + groups['_ITEM_V2_IDS']
                + groups['_ITEM_V3_IDS']
                + groups['_ITEM_V4_PLUS_IDS']
            ),
            'creative': set(groups['_CREATIVE_IDS']),
        }
        self.assertEqual(routed, configured)

    def test_independent_dense_parameter_budget(self):
        embedding_size = 17
        user_groups = ((102, 3), (149, 3), (134, 4))
        item_groups = ((202, 5), (203, 5), (202, 5), (228, 6))
        creative_fields = 14
        token_num, token_dim, mixer_layers, mixer_hidden = 32, 256, 3, 896

        user_width = sum(fields for fields, _ in user_groups) * embedding_size
        item_width = sum(fields for fields, _ in item_groups) * embedding_size
        creative_width = creative_fields * embedding_size
        input_bn = 2 * (user_width + item_width + creative_width)

        senet = 0
        for input_width, lowrank, output_width in (
                (user_width, 256, user_width),
                (user_width + item_width, 128, item_width),
                (creative_width, 128, creative_width)):
            senet += input_width * lowrank + lowrank
            senet += 2 * lowrank
            senet += lowrank * output_width + output_width

        local_tokens = 0
        for fields, group_tokens in user_groups + item_groups:
            output_width = group_tokens * token_dim
            local_tokens += fields * embedding_size * output_width
            local_tokens += output_width + 2 * output_width

        global_input = user_width + item_width
        global_token = (
            2 * global_input
            + global_input * 512 + 512
            + 512 * token_dim + token_dim
            + 2 * token_dim
        )
        per_mixer_layer = (
            2 * (
                token_num * token_dim * mixer_hidden
                + token_num * mixer_hidden
            )
            + token_num * mixer_hidden * token_dim
            + token_num * token_dim
            + 2 * token_dim + mixer_hidden + token_dim
        )
        mixer = mixer_layers * per_mixer_layer + 2 * token_dim
        creative = (
            creative_width * 256 + 256 + 256
            + 256 * 32 + 32 + 32
            + 2 * (256 + 32)
        )
        task_head = (
            (token_dim + 32) * 256 + 256 + 2 * 256
            + 256 * 128 + 128 + 2 * 128
            + 128 + 1
        )
        total = sum((
            input_bn,
            senet,
            local_tokens,
            global_token,
            mixer,
            creative,
            task_head,
        ))
        self.assertEqual(total, 109976671)

    def test_args_reuse_successful_outer_runtime_contract(self):
        lines, model_args, outer_args = _load_args(ARGS_PATH)
        _, _, runtime_outer_args = _load_args(RUNTIME_ARGS_PATH)
        _, _, model_class, _ = _parse(MODEL_PATH)

        self.assertEqual(
            lines[0],
            'models.rankmixer.cvr_mature_rankmixer_fst_v1.MLPModel',
        )
        self.assertEqual(outer_args, runtime_outer_args)
        self.assertEqual(
            model_args['runtime_build_id'],
            _class_literal(model_class, '_RUNTIME_BUILD_ID'),
        )
        self.assertEqual(
            model_args['feature_version'],
            'data.cvr.cvr_fea_v10_base_cold',
        )
        self.assertEqual(model_args['feature_version_old'], model_args['feature_version'])
        self.assertEqual(model_args['cvr_layers'], [256, 128])
        self.assertEqual(model_args['mixup_token_dim'], 256)
        self.assertEqual(model_args['mlp_mixer_layers'], 3)
        self.assertFalse(model_args['enable_dense_warmup'])
        self.assertFalse(model_args['enable_rpy_neg_sampler'])
        self.assertNotIn('enable_phalanx', model_args)
        self.assertIn('--ignore_dense_checkpoint=True', outer_args)
        self.assertIn('--ignore_sparse_checkpoint=False', outer_args)
        self.assertIn('--train_fuse_bn=false', outer_args)
        self.assertIn('--worker_numa_strategy=select', outer_args)

    def test_minimum_model_specific_upload_files_exist(self):
        required = (
            MODEL_PATH,
            ARGS_PATH,
            ROOT / 'src/data/cvr/cvr_fea_v10_base_cold.py',
            ROOT / 'src/data/cvr/cvr_fea_v10_base.py',
        )
        for path in required:
            self.assertTrue(path.is_file(), str(path))


if __name__ == '__main__':
    unittest.main()
