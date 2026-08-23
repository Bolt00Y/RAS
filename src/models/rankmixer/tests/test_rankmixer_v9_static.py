import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v9.py'
ARGS_PATH = ROOT / 'bash/set-rankmixer-v9-args.txt'


def _model_ast():
    source = MODEL_PATH.read_text(encoding='utf-8')
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


def _class_literal(model_class, name):
    assignment = next(
        node for node in model_class.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def _load_args():
    lines = ARGS_PATH.read_text(encoding='utf-8').splitlines()
    model_args_line = next(
        line for line in lines if line.startswith("--model_args='")
    )
    model_args = json.loads(model_args_line[len("--model_args='"):-1])
    return lines, model_args


class RankMixerV9StaticTest(unittest.TestCase):
    def test_frozen_semantic_group_abi(self):
        _, _, model_class = _model_ast()
        group_method = _method(model_class, '_build_semantic_feature_groups')
        return_node = next(
            node for node in ast.walk(group_method)
            if isinstance(node, ast.Return)
        )
        groups = ast.literal_eval(return_node.value)
        expected_sizes = _class_literal(model_class, '_GROUP_SIZES')
        expected_checksums = _class_literal(model_class, '_GROUP_CHECKSUMS')

        all_ids = set()
        expected_counts = {'common': 385, 'item': 835, 'creative': 14}
        for bucket_name in ('common', 'item', 'creative'):
            actual_sizes = tuple(len(field_ids) for _, field_ids in groups[bucket_name])
            self.assertEqual(actual_sizes, expected_sizes[bucket_name])
            ordered_ids = [
                field_id
                for _, field_ids in groups[bucket_name]
                for field_id in field_ids
            ]
            self.assertEqual(len(ordered_ids), expected_counts[bucket_name])
            self.assertEqual(len(set(ordered_ids)), len(ordered_ids))
            self.assertTrue(all_ids.isdisjoint(ordered_ids))
            all_ids.update(ordered_ids)
            checksum = hashlib.sha256(
                '|'.join(ordered_ids).encode('utf-8')
            ).hexdigest()
            self.assertEqual(checksum, expected_checksums[bucket_name])
        self.assertEqual(len(all_ids), 1234)

    def test_exact_dense_parameter_budget(self):
        source, _, model_class = _model_ast()
        expected_total = _class_literal(
            model_class, '_EXPECTED_DENSE_TRAINABLE_PARAMS'
        )

        common_fields, item_fields, creative_fields = (385, 835, 14)
        total_fields = common_fields + item_fields + creative_fields
        input_dim = total_fields * 17
        token_num, local_tokens, hidden_dim = 32, 31, 512
        swiglu_hidden, mixer_layers = 512, 2

        input_bn = 2 * input_dim
        senet = (
            common_fields * 128 + 128 * common_fields
            + (common_fields + item_fields) * 128 + 128 * item_fields
            + total_fields * 128 + 128 * creative_fields
            + 3 * 2 * 128
        )
        one_dcnm = (
            input_dim * 500 + 500
            + 500 * input_dim + input_dim
            + 2 * input_dim
        )
        dcnm = 2 * one_dcnm

        local_projection = (
            2 * input_dim * hidden_dim
            + local_tokens * hidden_dim
            + local_tokens * hidden_dim
        )
        global_projection = (
            input_dim * hidden_dim + hidden_dim
            + hidden_dim * hidden_dim + hidden_dim
            + hidden_dim
        )
        one_swiglu = token_num * (
            hidden_dim * swiglu_hidden + swiglu_hidden
            + hidden_dim * swiglu_hidden + swiglu_hidden
            + swiglu_hidden * hidden_dim + hidden_dim
        )
        mixer = mixer_layers * (
            2 * token_num * hidden_dim + 2 * one_swiglu
        )
        final_norm = token_num * hidden_dim
        pool = 2 * (hidden_dim * 128 + 128)
        flatten = local_tokens * hidden_dim * 256 + 256 + 256 + 1

        shortcut_dim = 512
        shortcut = input_dim * shortcut_dim + shortcut_dim + 2 * shortcut_dim

        task_head = 0
        previous = 2 * hidden_dim + 256 + shortcut_dim
        for width in (2048, 2048, 256):
            task_head += previous * width + width + 2 * width
            previous = width
        task_head += previous + 1

        actual_total = sum([
            input_bn,
            senet,
            dcnm,
            local_projection,
            global_projection,
            mixer,
            final_norm,
            pool,
            flatten,
            shortcut,
            task_head,
        ])
        self.assertEqual(actual_total, 199445658)
        self.assertEqual(actual_total, expected_total)
        self.assertLess(actual_total, 200000000)

        graph_verify_source = ast.get_source_segment(
            source,
            _method(model_class, '_verify_graph_dense_trainable_params'),
        )
        self.assertIn('tf.GraphKeys.TRAINABLE_VARIABLES', graph_verify_source)
        self.assertIn('scope=dense_scope', graph_verify_source)
        self.assertIn('_EXPECTED_DENSE_TRAINABLE_PARAMS', graph_verify_source)

    def test_end_to_end_raw_cross_fusion_structure(self):
        source, tree, model_class = _model_ast()
        model_fn_source = ast.get_source_segment(
            source, _method(model_class, 'model_fn')
        )
        semantic_source = ast.get_source_segment(
            source, _method(model_class, '_semantic_tokenize')
        )
        global_source = ast.get_source_segment(
            source, _method(model_class, '_build_global_token')
        )
        shortcut_source = ast.get_source_segment(
            source, _method(model_class, '_dcnm_shortcut')
        )
        task_head_source = ast.get_source_segment(
            source, _method(model_class, '_task_head')
        )
        loss_source = ast.get_source_segment(
            source, _method(model_class, 'build_loss_op')
        )

        self.assertNotIn('tf.stop_gradient', model_fn_source)
        self.assertNotIn('anchor_', model_fn_source)
        self.assertNotIn('residual_pred', model_fn_source)
        self.assertIn('raw_bucket_field_maps', semantic_source)
        self.assertIn('crossed_bucket_field_maps', semantic_source)
        self.assertIn('[raw_input, crossed_input]', semantic_source)
        self.assertIn('global_input = crossed_input', global_source)
        self.assertIn('num_outputs=self.rm_dcnm_shortcut_dim', shortcut_source)
        self.assertIn('ModelBase.batch_norm_layer_v2', shortcut_source)
        self.assertIn("scope='rm_v9_mlp{}'", task_head_source)
        self.assertIn("scope='rm_v9_out'", task_head_source)
        self.assertIn('tf.losses.log_loss', loss_source)
        self.assertNotIn('loss_anchor', loss_source)
        self.assertNotIn('loss_residual', loss_source)
        self.assertIn("'logits': logits", model_fn_source)
        self.assertIn("'pred': predictions", model_fn_source)

        forbidden_imports = [
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and 'cvr_bn_rankmixer_v' in node.module
        ]
        self.assertEqual(forbidden_imports, [])

    def test_base_scope_abi(self):
        source, _, model_class = _model_ast()
        senet_source = ast.get_source_segment(
            source, _method(model_class, 'senet_layer')
        )
        dcnm_source = ast.get_source_segment(
            source, _method(model_class, 'dcnm_cross_layer')
        )
        shortcut_source = ast.get_source_segment(
            source, _method(model_class, '_dcnm_shortcut')
        )
        head_source = ast.get_source_segment(
            source, _method(model_class, '_task_head')
        )
        model_fn_source = ast.get_source_segment(
            source, _method(model_class, 'model_fn')
        )
        for variable_name in (
            'common_weight_in',
            'common_weight_out',
            'common_item_weight_in',
            'item_weight_out',
            'common_item_creative_weight_in',
            'creative_weight_out',
        ):
            self.assertIn(variable_name, senet_source)
        self.assertGreaterEqual(senet_source.count('2.0 * tf.nn.sigmoid'), 3)
        for scope in (
            'dcnm-cross',
            'dcnm_cross_layer0_',
            'dcnm_cross_layer1_',
            'dcnm_ln_',
        ):
            self.assertIn(scope, dcnm_source)
        self.assertIn(
            'tf.multiply(inputs, deep_inputs_cvr) + last_layer_cvr',
            dcnm_source,
        )
        self.assertIn('layer_norm_for_train is None', dcnm_source)
        self.assertIn("'rm_dcnm_shortcut'", shortcut_source)
        self.assertIn("scope_bn='bn'", shortcut_source)
        for scope in ('rm_v9_mlp', 'rm_v9_bn_', 'rm_v9_out'):
            self.assertIn(scope, head_source)
        self.assertIn('weights_initializer=self.get_init(input_dim)', head_source)
        self.assertIn(
            'self._verify_graph_dense_trainable_params(',
            model_fn_source,
        )

        architecture_order = [
            model_fn_source.index('self.senet_layer('),
            model_fn_source.index('self.dcnm_cross_layer('),
            model_fn_source.index('self._semantic_tokenize('),
            model_fn_source.index('self._rm_stack('),
            model_fn_source.index('self._dcnm_shortcut('),
            model_fn_source.index('self._task_head('),
        ]
        self.assertEqual(architecture_order, sorted(architecture_order))

    def test_fixed_context_dimensions(self):
        source, _, model_class = _model_ast()
        model_fn_source = ast.get_source_segment(
            source, _method(model_class, 'model_fn')
        )
        _, args = _load_args()

        rankmixer_context_dim = (
            2 * args['rm_hidden_dim'] + args['rm_flatten_dim']
        )
        fusion_dim = rankmixer_context_dim + args['rm_dcnm_shortcut_dim']
        self.assertEqual(rankmixer_context_dim, 1280)
        self.assertEqual(fusion_dim, 1792)
        self.assertIn('rankmixer_context.set_shape', model_fn_source)
        self.assertIn('fusion_context.set_shape', model_fn_source)
        self.assertIn(
            'self._task_head(fusion_context, is_train, export)',
            model_fn_source,
        )

    def test_cold_start_training_args(self):
        lines, args = _load_args()
        self.assertEqual(
            lines[0],
            'models.rankmixer.cvr_bn_rankmixer_v9.MLPModel',
        )
        self.assertIn('--train_dates=2026-07-01:2026-07-01', lines)
        self.assertIn('--test_date=2026-07-02:2026-07-02', lines)
        self.assertIn('--ignore_dense_checkpoint=True', lines)
        self.assertIn('--ignore_sparse_checkpoint=False', lines)
        self.assertEqual(args['feature_version'], 'data.cvr.cvr_fea_v10_base_cold')
        self.assertEqual(args['embedding_size'], 17)
        self.assertEqual(args['rm_bucket_token_counts'], [10, 20, 1])
        self.assertEqual(args['rm_hidden_dim'], 512)
        self.assertEqual(args['rm_swiglu_hidden_dim'], 512)
        self.assertEqual(args['rm_flatten_dim'], 256)
        self.assertEqual(args['rm_dcnm_shortcut_dim'], 512)
        self.assertEqual(args['dcnm_layer'], 500)
        self.assertEqual(args['cross_num'], 2)
        self.assertEqual(args['cvr_layers'], [2048, 2048, 256])
        self.assertTrue(args['save_predict_result'])
        for scope in (
                'dcnm-cross',
                'rm_local_tokenize',
                'rm_global_token',
                'rm_block',
                'rm_dcnm_shortcut',
                'rm_v9_mlp',
                'rm_v9_bn',
                'rm_v9_out'):
            self.assertIn(scope, args['skip_tensors'])
            self.assertIn(scope, args['warm_up_tensors'])
        for removed_arg in (
                'rm_residual_head_layers',
                'rm_alpha_max',
                'rm_alpha_init',
                'rm_main_loss_weight',
                'rm_anchor_loss_weight',
                'rm_residual_loss_weight',
                'rm_verify_gradient_isolation'):
            self.assertNotIn(removed_arg, args)


if __name__ == '__main__':
    unittest.main()
