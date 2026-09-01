import ast
import hashlib
import json
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
V4_PATH = ROOT / 'src/models/rankmixer/cvr_senet_mature_rankmixer_v4.py'
V5_PATH = ROOT / 'src/models/rankmixer/cvr_senet_mature_rankmixer_v5.py'
UNIMIXER_PATH = ROOT / 'src/models/rankmixer/cvr_bn_unimixer_v1.py'
V4_ARGS_PATH = ROOT / 'bash/set-rankmixer-mature-v4-args.txt'
V5_ARGS_PATH = ROOT / 'bash/set-rankmixer-mature-v5-args.txt'


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


def _module_assignment(tree, name):
    return next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )


def _semantic_literal(node):
    if isinstance(node, ast.Dict):
        return {
            _semantic_literal(key): _semantic_literal(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_semantic_literal(value) for value in node.elts)
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name) and node.func.id == '_ids'
        return tuple(ast.literal_eval(node.args[0]).split())
    return ast.literal_eval(node)


def _unimixer_semantic_groups(tree):
    builder = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == 'build_semantic_feature_groups'
    )
    return_node = next(
        node for node in ast.walk(builder) if isinstance(node, ast.Return)
    )
    raw = ast.literal_eval(return_node.value)
    return {
        bucket_name: tuple(
            (group_name, tuple(feature_ids))
            for group_name, feature_ids in raw[bucket_name]
        )
        for bucket_name in ('common', 'item')
    }


def _split_args(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    model_args_line = next(
        line for line in lines if line.startswith('--model_args=')
    )
    raw = model_args_line.split('=', 1)[1]
    model_args = json.loads(raw[1:-1])
    outer_args = [
        line for line in lines[1:] if not line.startswith('--model_args=')
    ]
    return lines[0], model_args, outer_args


class MatureRankMixerV5StaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v4_source, _, _, cls.v4_methods = _parse(V4_PATH)
        cls.v5_source, cls.v5_tree, cls.v5_class, cls.v5_methods = _parse(
            V5_PATH)
        _, cls.unimixer_tree, _, _ = _parse(UNIMIXER_PATH)
        cls.v5_groups = _semantic_literal(
            _module_assignment(
                cls.v5_tree, '_LOCAL_SEMANTIC_FEATURE_GROUPS'))
        cls.reference_groups = _unimixer_semantic_groups(
            cls.unimixer_tree)

    def test_semantic_groups_exactly_match_unimixer_common_item(self):
        self.assertEqual(set(self.v5_groups), {'common', 'item'})
        self.assertEqual(self.v5_groups, self.reference_groups)
        self.assertEqual(
            tuple(len(ids) for _, ids in self.v5_groups['common']),
            (39, 39, 39, 39, 39, 38, 38, 38, 38, 38),
        )
        self.assertEqual(
            tuple(len(ids) for _, ids in self.v5_groups['item']),
            (42, 42, 42, 42, 42, 35, 35, 35, 35, 35, 35,
             42, 42, 42, 42, 42, 41, 41, 41, 41, 41),
        )

    def test_semantic_groups_cover_common_item_once_and_exclude_creative(self):
        module_groups = {}
        for name in ('_USER_V1_IDS', '_USER_V2_IDS', '_USER_V3_IDS',
                     '_ITEM_V1_IDS', '_ITEM_V2_IDS', '_ITEM_V3_IDS',
                     '_ITEM_V4_PLUS_IDS', '_CREATIVE_IDS'):
            module_groups[name] = _semantic_literal(
                _module_assignment(self.v5_tree, name))

        common_ids = tuple(
            feature_id
            for _, feature_ids in self.v5_groups['common']
            for feature_id in feature_ids
        )
        item_ids = tuple(
            feature_id
            for _, feature_ids in self.v5_groups['item']
            for feature_id in feature_ids
        )
        expected_common = set(
            module_groups['_USER_V1_IDS']
            + module_groups['_USER_V2_IDS']
            + module_groups['_USER_V3_IDS'])
        expected_item = set(
            module_groups['_ITEM_V1_IDS']
            + module_groups['_ITEM_V2_IDS']
            + module_groups['_ITEM_V3_IDS']
            + module_groups['_ITEM_V4_PLUS_IDS'])
        creative = set(module_groups['_CREATIVE_IDS'])

        self.assertEqual(len(common_ids), 385)
        self.assertEqual(len(item_ids), 835)
        self.assertEqual(len(common_ids), len(set(common_ids)))
        self.assertEqual(len(item_ids), len(set(item_ids)))
        self.assertEqual(set(common_ids), expected_common)
        self.assertEqual(set(item_ids), expected_item)
        self.assertFalse(set(common_ids + item_ids).intersection(creative))

    def test_frozen_semantic_checksums_are_correct(self):
        group_checksums = ast.literal_eval(
            _module_assignment(
                self.v5_tree, '_LOCAL_SEMANTIC_GROUP_CHECKSUMS'))
        bucket_checksums = ast.literal_eval(
            _module_assignment(
                self.v5_tree, '_LOCAL_SEMANTIC_BUCKET_CHECKSUMS'))
        for bucket_name in ('common', 'item'):
            bucket_ids = []
            for group_name, feature_ids in self.v5_groups[bucket_name]:
                bucket_ids.extend(feature_ids)
                self.assertEqual(
                    hashlib.sha256(
                        '\n'.join(feature_ids).encode('utf-8')).hexdigest(),
                    group_checksums[group_name],
                )
            self.assertEqual(
                hashlib.sha256(
                    '\n'.join(bucket_ids).encode('utf-8')).hexdigest(),
                bucket_checksums[bucket_name],
            )

    def test_tokenizer_is_independent_linear_then_bn(self):
        projection_source = ast.get_source_segment(
            self.v5_source, self.v5_methods['_project_semantic_group'])
        tokenizer_source = ast.get_source_segment(
            self.v5_source, self.v5_methods['_semantic_tokenize'])
        self.assertIn('activation=None', projection_source)
        self.assertIn('tf.random_normal_initializer', projection_source)
        self.assertIn('1.0 / math.sqrt(float(input_dim))', projection_source)
        self.assertIn('tf.zeros_initializer()', projection_source)
        self.assertIn('ModelBase.batch_norm_layer_v2', projection_source)
        self.assertIn("scope_bn='token_bn'", projection_source)
        self.assertNotIn('_gelu', projection_source)
        self.assertNotIn('_mature_batch_norm', projection_source)
        self.assertIn("for bucket_name in ('common', 'item')", tokenizer_source)
        self.assertIn('tf.stack(tokens, axis=1', tokenizer_source)
        self.assertNotIn('creative', tokenizer_source)

    def test_model_wires_post_senet_fields_and_keeps_v4_global_creative(self):
        model_source = ast.get_source_segment(
            self.v5_source, self.v5_methods['model_fn'])
        self.assertIn("'common': self._split_bucket_fields(\n                    user_senet", model_source)
        self.assertIn("'item': self._split_bucket_fields(\n                    item_senet", model_source)
        self.assertIn('local_tokens = self._semantic_tokenize(', model_source)
        self.assertIn("[user_bn, item_bn]", model_source)
        self.assertIn('self._creative_converter(\n                creative_senet', model_source)
        self.assertNotIn('_embedding_to_tokens', model_source)
        self.assertIn("name='all_31_local_plus_global'", model_source)
        self.assertIn("name='rankmixer_input_tokens'", model_source)

    def test_default_dense_parameter_budget(self):
        calculate_source = textwrap.dedent(ast.get_source_segment(
            self.v5_source,
            self.v5_methods['_calculate_dense_trainable_params'],
        ))
        namespace = {'_CREATIVE_IDS': tuple(range(14))}
        exec(calculate_source, namespace)

        class FakeModel(object):
            pass

        model = FakeModel()
        model._USER_GROUPS = (
            ('u1', tuple(range(102)), 3),
            ('u2', tuple(range(149)), 3),
            ('u3', tuple(range(134)), 4),
        )
        model._ITEM_GROUPS = (
            ('i1', tuple(range(202)), 5),
            ('i2', tuple(range(203)), 5),
            ('i3', tuple(range(202)), 5),
            ('i4', tuple(range(228)), 6),
        )
        model._LOCAL_SEMANTIC_FEATURE_GROUPS = self.v5_groups
        model.embedding_size = 17
        model.mixup_token_dim = 384
        model.mixup_token_num = 32
        model.mixer_hidden_dim = 1344
        model.batch_norm = True
        model.use_senet = True
        model.use_senet_bn = True
        model.user_senet_lowrank = 384
        model.item_senet_lowrank = 192
        model.creative_senet_lowrank = 192
        model.global_token_hidden_dim = 768
        model.mlp_mixer_layers = 3
        model.creative_hidden_dim = 384
        model.creative_output_dim = 48
        model.cvr_layers = [384, 192]

        breakdown = namespace['_calculate_dense_trainable_params'](model)
        self.assertEqual(breakdown, {
            'input_bn': 41956,
            'senet': 11848754,
            'local_tokens': 7999872,
            'global_token': 16266632,
            'mixer': 99264576,
            'creative_bypass': 111552,
            'task_head': 241537,
            'total': 135774879,
        })

    def test_non_tokenizer_model_methods_remain_v4_identical(self):
        allowed_changes = {
            '__init__',
            '_validate_feature_contract',
            '_calculate_dense_trainable_params',
            '_embedding_to_tokens',
            'model_fn',
        }
        common_methods = set(self.v4_methods).intersection(self.v5_methods)
        for method_name in sorted(common_methods - allowed_changes):
            self.assertEqual(
                ast.dump(self.v5_methods[method_name], include_attributes=False),
                ast.dump(self.v4_methods[method_name], include_attributes=False),
                method_name,
            )

    def test_args_only_change_entry_and_runtime_build_id(self):
        v4_module, v4_args, v4_outer = _split_args(V4_ARGS_PATH)
        v5_module, v5_args, v5_outer = _split_args(V5_ARGS_PATH)
        self.assertEqual(
            v4_module,
            'models.rankmixer.cvr_senet_mature_rankmixer_v4.MLPModel')
        self.assertEqual(
            v5_module,
            'models.rankmixer.cvr_senet_mature_rankmixer_v5.MLPModel')
        self.assertEqual(v5_outer, v4_outer)
        self.assertEqual(
            v5_args.pop('runtime_build_id'),
            'mature_rankmixer_v5_semantic_psilu_d384_tf_only_20260901')
        self.assertEqual(
            v4_args.pop('runtime_build_id'),
            'mature_rankmixer_v4_psilu_d384_tf_only_20260901')
        self.assertEqual(v5_args, v4_args)
        self.assertTrue(v5_args['batch_norm'])
        self.assertEqual(v5_args['mixup_token_num'], 32)
        self.assertEqual(v5_args['mixup_token_dim'], 384)


if __name__ == '__main__':
    unittest.main()
