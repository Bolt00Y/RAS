import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v10.py'
ARGS_PATH = ROOT / 'bash/set-rankmixer-v10-args.txt'
DOC_PATH = ROOT / 'introduce/rankmixer_v10_introduction.md'


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


class RankMixerV10StaticTest(unittest.TestCase):
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
            actual_sizes = tuple(
                len(field_ids) for _, field_ids in groups[bucket_name]
            )
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
        swiglu_hidden, mixer_layers = 704, 2

        input_bn = 2 * input_dim
        senet = (
            common_fields * 128 + 128 * common_fields
            + (common_fields + item_fields) * 128 + 128 * item_fields
            + total_fields * 128 + 128 * creative_fields
            + 3 * 2 * 128
        )
        local_projection = (
            input_dim * hidden_dim
            + local_tokens * hidden_dim
            + 2 * hidden_dim
        )
        global_projection = (
            input_dim * hidden_dim + hidden_dim
            + hidden_dim * hidden_dim + hidden_dim
            + 2 * hidden_dim
        )
        one_swiglu = token_num * (
            hidden_dim * swiglu_hidden + swiglu_hidden
            + hidden_dim * swiglu_hidden + swiglu_hidden
            + swiglu_hidden * hidden_dim + hidden_dim
        )
        mixer = mixer_layers * (4 * hidden_dim + 2 * one_swiglu)
        final_norm = 2 * hidden_dim

        task_head = 0
        previous = token_num * hidden_dim
        for width in (2048, 2048, 256):
            task_head += previous * width + width + 2 * width
            previous = width
        task_head += previous + 1

        actual_total = sum([
            input_bn,
            senet,
            local_projection,
            global_projection,
            mixer,
            final_norm,
            task_head,
        ])
        self.assertEqual(actual_total, 199275877)
        self.assertEqual(actual_total, expected_total)
        self.assertLess(actual_total, 200000000)

        calculate_source = ast.get_source_segment(
            source,
            _method(model_class, '_calculate_dense_trainable_params'),
        )
        verify_source = ast.get_source_segment(
            source,
            _method(model_class, '_verify_graph_dense_trainable_params'),
        )
        self.assertIn('2 * self.rm_hidden_dim', calculate_source)
        self.assertIn('tf.GraphKeys.TRAINABLE_VARIABLES', verify_source)
        self.assertIn('_EXPECTED_DENSE_TRAINABLE_PARAMS', verify_source)

    def test_exact_extended_flops_and_document(self):
        input_dim = 20978
        token_num, local_tokens, hidden_dim = 32, 31, 512
        swiglu_hidden, mixer_layers = 704, 2

        input_bn = 4 * input_dim
        senet = 1087414
        local_tokenizer = (
            2 * input_dim * hidden_dim
            + 9 * local_tokens * hidden_dim
            + local_tokens * (8 * hidden_dim + 2)
        )
        global_token = (
            2 * input_dim * hidden_dim
            + 2 * hidden_dim * hidden_dim
            + 9 * hidden_dim
            + (8 * hidden_dim + 2)
        )
        one_stage = token_num * (
            6 * hidden_dim * swiglu_hidden
            + 3 * swiglu_hidden
            + (8 * hidden_dim + 2)
        )
        mixer = mixer_layers * (
            2 * one_stage + 2 * token_num * hidden_dim
        )
        final_norm = token_num * (8 * hidden_dim + 2)
        task_head = (
            2 * (token_num * hidden_dim) * 2048 + 4 * 2048 + 9 * 2048
            + 2 * 2048 * 2048 + 4 * 2048 + 9 * 2048
            + 2 * 2048 * 256 + 4 * 256 + 9 * 256
            + 2 * 256 + 1
        )
        total = sum([
            input_bn,
            senet,
            local_tokenizer,
            global_token,
            mixer,
            final_norm,
            task_head,
        ])
        self.assertEqual(total, 399355903)

        document = DOC_PATH.read_text(encoding='utf-8')
        for required in (
                '199,275,877',
                '399,355,903',
                '[B,32,512]',
                '[B,16384]',
                'rm_norm_type',
                'layer_norm',
                'rm_readout_type',
                'pure_flat'):
            self.assertIn(required, document)

    def test_standard_layer_norm_path(self):
        source, _, model_class = _model_ast()
        norm_source = ast.get_source_segment(
            source, _method(model_class, '_rm_layer_norm')
        )
        semantic_source = ast.get_source_segment(
            source, _method(model_class, '_semantic_tokenize')
        )
        global_source = ast.get_source_segment(
            source, _method(model_class, '_build_global_token')
        )
        block_source = ast.get_source_segment(
            source, _method(model_class, '_rm_block')
        )
        model_fn_source = ast.get_source_segment(
            source, _method(model_class, 'model_fn')
        )

        self.assertIn('return layer_norm(inputs', norm_source)
        self.assertIn('export=export', norm_source)
        self.assertIn("scope='local_token_layer_norm'", semantic_source)
        self.assertIn("scope='layer_norm'", global_source)
        self.assertIn("scope='mixed_layer_norm'", block_source)
        self.assertIn("scope='original_layer_norm'", block_source)
        self.assertIn("scope='rm_final_layer_norm'", model_fn_source)
        for forbidden in (
                '_rm_rms_norm',
                'rm_final_rms_norm',
                'tf.rsqrt',
                'rm_rms_epsilon'):
            self.assertNotIn(forbidden, source)

    def test_mixing_reverting_long_residual_order(self):
        source, _, model_class = _model_ast()
        block_source = ast.get_source_segment(
            source, _method(model_class, '_rm_block')
        )
        expected_order = [
            block_source.index('self._rm_mix_tokens('),
            block_source.index("scope='mixed_layer_norm'"),
            block_source.index("scope='mixed_swiglu'"),
            block_source.index('self._rm_revert_tokens('),
            block_source.index("scope='original_layer_norm'"),
            block_source.index("scope='original_swiglu'"),
            block_source.index('output = inputs + original_update'),
        ]
        self.assertEqual(expected_order, sorted(expected_order))
        self.assertIn('mixed_hidden = mixed + mixed_update', block_source)

    def test_pure_flat_single_readout(self):
        source, _, model_class = _model_ast()
        method_names = {
            node.name for node in model_class.body
            if isinstance(node, ast.FunctionDef)
        }
        model_fn_source = ast.get_source_segment(
            source, _method(model_class, 'model_fn')
        )
        head_source = ast.get_source_segment(
            source, _method(model_class, '_task_head')
        )

        self.assertNotIn('_global_conditioned_pool', method_names)
        self.assertNotIn('_flatten_readout', method_names)
        self.assertIn(
            'context_dim = self.rm_token_num * self.rm_hidden_dim',
            model_fn_source,
        )
        self.assertIn('final_tokens,', model_fn_source)
        self.assertIn("name='rm_pure_flatten'", model_fn_source)
        self.assertIn('self._task_head(context, is_train, export)', model_fn_source)
        self.assertNotIn('pool_weights', model_fn_source)
        self.assertNotIn('flatten_gate', model_fn_source)
        self.assertNotIn('dcnm', model_fn_source.lower())
        self.assertIn("scope='rm_v10_mlp{}'", head_source)
        self.assertIn("scope='rm_v10_out'", head_source)

    def test_independent_model_and_single_bce(self):
        source, tree, model_class = _model_ast()
        loss_source = ast.get_source_segment(
            source, _method(model_class, 'build_loss_op')
        )
        model_fn_source = ast.get_source_segment(
            source, _method(model_class, 'model_fn')
        )
        forbidden_imports = [
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and 'cvr_bn_rankmixer_v' in node.module
        ]
        self.assertEqual(forbidden_imports, [])
        self.assertIn('tf.losses.log_loss', loss_source)
        self.assertNotIn('aux', loss_source.lower())
        self.assertIn('"logits": logits', model_fn_source)
        self.assertIn('"pred": predictions', model_fn_source)

    def test_cold_start_training_args(self):
        lines, args = _load_args()
        self.assertEqual(
            lines[0],
            'models.rankmixer.cvr_bn_rankmixer_v10.MLPModel',
        )
        self.assertIn('--train_dates=2026-07-01:2026-07-01', lines)
        self.assertIn('--test_date=2026-07-02:2026-07-02', lines)
        self.assertIn('--ignore_dense_checkpoint=True', lines)
        self.assertIn('--ignore_sparse_checkpoint=False', lines)
        self.assertEqual(args['feature_version'], 'data.cvr.cvr_fea_v10_base_cold')
        self.assertEqual(args['embedding_size'], 17)
        self.assertEqual(args['rm_bucket_token_counts'], [10, 20, 1])
        self.assertEqual(args['rm_token_num'], 32)
        self.assertEqual(args['rm_local_token_num'], 31)
        self.assertEqual(args['rm_hidden_dim'], 512)
        self.assertEqual(args['rm_layer_num'], 2)
        self.assertEqual(args['rm_head_num'], 32)
        self.assertEqual(args['rm_swiglu_hidden_dim'], 704)
        self.assertEqual(args['rm_norm_type'], 'layer_norm')
        self.assertEqual(args['rm_readout_type'], 'pure_flat')
        self.assertEqual(args['cvr_layers'], [2048, 2048, 256])
        self.assertTrue(args['save_predict_result'])
        self.assertNotIn('rm_rms_epsilon', args)
        self.assertNotIn('rm_pool_query_dim', args)
        self.assertNotIn('rm_flatten_dim', args)
        self.assertNotIn('rm_flatten_gate_init', args)
        for scope in (
                'rm_local_tokenize',
                'rm_global_token',
                'rm_block',
                'rm_final_layer_norm',
                'rm_v10_mlp',
                'rm_v10_bn',
                'rm_v10_out',
                'bn_input',
                'senet'):
            self.assertIn(scope, args['skip_tensors'])
            self.assertIn(scope, args['warm_up_tensors'])
        for removed_scope in (
                'rm_final_rms_norm',
                'rm_global_conditioned_pool',
                'rm_flatten_readout',
                'rm_v5_mlp'):
            self.assertNotIn(removed_scope, args['skip_tensors'])
            self.assertNotIn(removed_scope, args['warm_up_tensors'])


if __name__ == '__main__':
    unittest.main()
