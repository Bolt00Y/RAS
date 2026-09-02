"""v6-E4 architecture contracts and isolated, real TensorFlow kernel checks.

Numerical tests load the production methods with AST, without importing Flood
or contacting HDFS/PS. They cover local tokenization, final LayerNorm and SENet
dependency routing (BN disabled only for that isolated routing probe). They do
not substitute kernels or claim to validate a full Flood training job.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[4]
SMALL = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e2_small.py'
E4 = ROOT / 'src/models/rankmixer/cvr_bn_rankmixer_v6_e4.py'
V5 = ROOT / 'src/models/rankmixer/cvr_senet_mature_rankmixer_v5.py'
REPLAY_MIXER = ROOT / 'new_rankmixer_0831/models/seq_model/mlp_mixer_swiglu_fuse_v4.py'
MODEL_BASE = ROOT / 'src/models/model_base.py'
ARGS = ROOT / 'bash/set-rankmixer-v6-e4-args.txt'
KERNEL_METHODS = {
    'get_init', '_build_semantic_feature_groups', '_validate_semantic_feature_groups',
    '_calculate_dense_trainable_params', 'senet_layer', '_rm_rms_norm', '_rm_norm',
    '_project_token_family', '_semantic_tokenize', '_rm_final_layer_norm',
}


def model_ast(path):
    return next(node for node in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(node, ast.ClassDef) and node.name in ('MLPModel', 'ModelBase'))


def method_map(model):
    return {node.name: node for node in model.body if isinstance(node, ast.FunctionDef)}


def class_value(model, name):
    return ast.literal_eval(next(node.value for node in model.body if isinstance(node, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)))


def parse_args(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    line = next(line for line in lines if line.startswith("--model_args='"))
    return lines[0], json.loads(line[len("--model_args='"):-1]), [
        line for line in lines[1:] if not line.startswith('--model_args=')]


def module_node(body):
    node = ast.Module(body=body)
    if 'type_ignores' in ast.Module._fields:
        node.type_ignores = []
    return ast.fix_missing_locations(node)


def kernel_model(tf=None, np=None, partitions=1, optimized=True):
    cls = model_ast(E4)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in KERNEL_METHODS]
    assert {node.name for node in methods} == KERNEL_METHODS
    attributes = [node for node in cls.body if isinstance(node, ast.Assign)]
    activation = method_map(model_ast(MODEL_BASE))['get_act_func']
    kernel = ast.ClassDef(name='KernelModel', bases=[], keywords=[],
                          body=attributes + methods + [activation], decorator_list=[])
    namespace = {'tf': tf, 'np': np, 'math': math, 'logging': logging, 'hashlib': hashlib}
    exec(compile(module_node([kernel]), str(E4), 'exec'), namespace)
    model = namespace['KernelModel']()
    for key, value in parse_args(ARGS)[1].items():
        setattr(model, key, value)
    model.rm_semantic_feature_groups = model._build_semantic_feature_groups()
    model.fea_conf_obj = SimpleNamespace(**{
        bucket + '_fea_map': dict.fromkeys(
            field for _, fields in groups for field in fields)
        for bucket, groups in model.rm_semantic_feature_groups.items()
    })
    model.partitioner = (tf.min_max_variable_partitioner(
        max_partitions=partitions, min_slice_size=1024000) if tf is not None else None)
    model.rm_optimize_tokenize = optimized
    return model


class CosmeticNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace('RankMixer v6-E2-Small', 'RankMixer v6-E4')
        return node

    def visit_Str(self, node):
        node.s = node.s.replace('RankMixer v6-E2-Small', 'RankMixer v6-E4')
        return node


def normalized(node):
    node = CosmeticNormalizer().visit(copy.deepcopy(node))
    if isinstance(node, ast.FunctionDef) and ast.get_docstring(node) is not None:
        node.body = node.body[1:]
    return ast.dump(node, include_attributes=False)


class RankMixerE4ArchitectureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_class = model_ast(E4)
        cls.methods = method_map(cls.model_class)

    def test_provenance_and_original_server_imports(self):
        def imports(path):
            return [ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports(SMALL), imports(E4))
        self.assertIn('由 cvr_bn_rankmixer_v6_e2_small.py 发展而来',
                      E4.read_text(encoding='utf-8').splitlines()[1])
        self.assertEqual(len(self.model_class.bases), 1)
        self.assertEqual(self.model_class.bases[0].id, 'ModelBase')

    def test_backbone_and_training_lifecycle_match_small(self):
        small_methods = method_map(model_ast(SMALL))
        changed = {'__init__', '_build_semantic_feature_groups',
                   '_calculate_dense_trainable_params', 'senet_layer',
                   '_semantic_tokenize', '_build_global_token', 'model_fn'}
        self.assertEqual(set(self.methods) - set(small_methods),
                         {'_creative_converter', '_rm_final_layer_norm'})
        for name in sorted(set(small_methods) - changed):
            self.assertEqual(normalized(small_methods[name]), normalized(self.methods[name]), name)
        self.assertTrue({'_rm_block', '_rm_per_token_swiglu', '_task_head',
                         'build_loss_op', 'build_optimizer_op', 'get_dataset',
                         'train', 'test', '_build_export'}.isdisjoint(changed))

    def test_local_groups_match_v5_and_cover_features_once(self):
        def literal(node):
            if isinstance(node, ast.Dict):
                return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
            if isinstance(node, (ast.Tuple, ast.List)):
                return [literal(value) for value in node.elts]
            if isinstance(node, ast.Call):
                self.assertEqual(node.func.id, '_ids')
                return ast.literal_eval(node.args[0]).split()
            return ast.literal_eval(node)
        reference = literal(next(node.value for node in ast.parse(V5.read_text(encoding='utf-8')).body
                                 if isinstance(node, ast.Assign) and any(
                                     isinstance(t, ast.Name) and t.id == '_LOCAL_SEMANTIC_FEATURE_GROUPS'
                                     for t in node.targets)))
        reference = {bucket: [(name, fields) for name, fields in groups]
                     for bucket, groups in reference.items()}
        model = kernel_model()
        groups = model.rm_semantic_feature_groups
        local_buckets = model._LOCAL_BUCKET_NAMES
        self.assertEqual(local_buckets, ('common', 'item'))
        self.assertEqual({name: groups[name] for name in local_buckets}, reference)
        self.assertEqual([len(groups[name]) if name in local_buckets else 0
                          for name in model._BUCKET_NAMES], [10, 21, 0])
        self.assertEqual([sum(len(fields) for _, fields in groups[name])
                          for name in model._BUCKET_NAMES], [385, 835, 14])
        all_ids = [field for bucket in groups.values() for _, fields in bucket for field in fields]
        self.assertEqual(len(set(all_ids)), 1234)
        model._validate_semantic_feature_groups()

    def test_group_contract_rejects_changed_field_order(self):
        model = kernel_model()
        ids = model.rm_semantic_feature_groups['item'][0][1]
        ids[0], ids[1] = ids[1], ids[0]
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            model._validate_semantic_feature_groups()

    def test_exact_dense_parameter_budget(self):
        model = kernel_model()
        self.assertEqual(model._EXPECTED_DENSE_TRAINABLE_PARAMS, 80739301)
        self.assertEqual(model._calculate_dense_trainable_params(), 80739301)
        self.assertEqual(sum([41956, 365952, 5325312, 5375744,
                              69451776, 512, 70272, 107777]), 80739301)
        assignments = [node for node in ast.walk(self.methods['__init__'])
                       if isinstance(node, ast.Assign)]
        required = next(node.value for node in assignments
                        if any(isinstance(t, ast.Name) and t.id == 'required_architecture'
                               for t in node.targets))
        guards = {ast.literal_eval(key): ast.literal_eval(value.elts[1])
                  for key, value in zip(required.keys, required.values)}
        self.assertEqual(guards, {
            'senet_hidden_size': 128, 'rm_token_num': 32, 'rm_local_token_num': 31,
            'rm_hidden_dim': 256, 'rm_layer_num': 2, 'rm_head_num': 32,
            'rm_swiglu_hidden_dim': 704, 'creative_hidden_dim': 256, 'creative_output_dim': 32,
        })
        self.assertEqual(model.cvr_layers, [256, 128])
        self.assertEqual((model.rm_hidden_dim, model.rm_swiglu_hidden_dim, model.rm_layer_num), (256, 704, 2))

    def test_model_routes_pre_senet_global_and_direct_fusion(self):
        assignments = {target.id: node.value for node in ast.walk(self.methods['model_fn'])
                       if isinstance(node, ast.Assign) for target in node.targets
                       if isinstance(target, ast.Name)}
        global_call = assignments['global_token']
        self.assertEqual(global_call.func.attr, '_build_global_token')
        self.assertEqual(global_call.args[0].id, 'normalized_buckets')
        final_call = assignments['final_tokens']
        self.assertEqual(final_call.func.attr, '_rm_final_layer_norm')
        self.assertEqual(final_call.args[0].id, 'hidden_tokens')
        pool = assignments['mixer_context']
        self.assertEqual(pool.func.attr, 'reduce_mean')
        self.assertEqual(pool.args[0].id, 'final_tokens')
        self.assertEqual(ast.literal_eval(next(k.value for k in pool.keywords if k.arg == 'axis')), 1)
        creative = assignments['creative_context']
        self.assertEqual(creative.func.attr, '_creative_converter')
        self.assertEqual(ast.dump(creative.args[0]),
                         ast.dump(ast.parse("bucket_tensors['creative']", mode='eval').body))
        concat = assignments['context']
        self.assertEqual(concat.func.attr, 'concat')
        self.assertEqual([node.id for node in concat.args[0].elts], ['mixer_context', 'creative_context'])
        for method in ('_semantic_tokenize', '_build_global_token'):
            self.assertIn("attr='_LOCAL_BUCKET_NAMES'", ast.dump(self.methods[method]))
            self.assertNotIn("attr='_BUCKET_NAMES'", ast.dump(self.methods[method]))

    def test_creative_converter_has_two_bn_and_parameterized_swish_stages(self):
        method = self.methods['_creative_converter']
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        dense = [node for node in calls if isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'fully_connected']
        self.assertEqual(len(dense), 2)
        self.assertEqual([next(k.value.attr for k in node.keywords if k.arg == 'num_outputs')
                          for node in dense], ['creative_hidden_dim', 'creative_output_dim'])
        self.assertEqual(len([node for node in calls if isinstance(node.func, ast.Name)
                              and node.func.id == 'apply_bn']), 2)
        self.assertEqual(len([node for node in calls if isinstance(node.func, ast.Attribute)
                              and node.func.attr == 'sigmoid']), 2)
        self.assertEqual(len([node for node in calls if isinstance(node.func, ast.Attribute)
                              and node.func.attr == 'constant_initializer'
                              and ast.literal_eval(node.args[0]) == 1.702]), 2)
        self.assertNotIn('_rm_norm', ast.dump(method))
        self.assertNotIn('layer_norm', ast.dump(method))

    def test_args_change_architecture_and_keep_server_settings(self):
        _, old, old_outer = parse_args(ROOT / 'bash/set-rankmixer-v6-e2-small-args.txt')
        entry, new, new_outer = parse_args(ARGS)
        self.assertEqual(entry, 'models.rankmixer.cvr_bn_rankmixer_v6_e4.MLPModel')
        self.assertEqual(old_outer, new_outer)
        self.assertIn('--ignore_dense_checkpoint=True', new_outer)
        expected = {
            'rm_readout_type': 'mean_pool_creative', 'rm_bucket_token_counts': [10, 21, 0],
            'rm_group_version': 'rankmixer_v6_e4_common_item_semantic_v1',
            'rm_final_norm_type': 'layer_norm', 'rm_final_ln_epsilon': 1e-8,
            'creative_hidden_dim': 256, 'creative_output_dim': 32, 'cvr_layers': [256, 128],
        }
        for name, value in expected.items():
            self.assertEqual(new.pop(name), value, name)
            old.pop(name, None)
        for name in ('skip_tensors', 'warm_up_tensors'):
            self.assertEqual(new.pop(name), old.pop(name).replace(
                'rm_final_rms_norm', 'rm_final_layer_norm') + ';rm_creative')
        self.assertEqual(old, new)


@unittest.skipUnless(importlib.util.find_spec('tensorflow') is not None,
                     'Real TensorFlow runtime is required for numerical checks')
class RankMixerE4NumericalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import tensorflow as tensorflow
        cls.np = np
        cls.tf = tensorflow if int(tensorflow.__version__.split('.')[0]) == 1 else tensorflow.compat.v1
        if int(tensorflow.__version__.split('.')[0]) >= 2:
            cls.tf.disable_v2_behavior()
        cls.config = cls.tf.ConfigProto(intra_op_parallelism_threads=32,
                                        inter_op_parallelism_threads=8,
                                        device_count={'GPU': 0})

    @unittest.skipUnless(REPLAY_MIXER.is_file(), 'Replay reference source is needed for this comparison')
    def test_final_layer_norm_matches_replay_reference(self):
        tf, np = self.tf, self.np
        graph = tf.Graph()
        with graph.as_default():
            model = kernel_model(tf, np)
            inputs = tf.placeholder(tf.float32, [None, 32, 256])
            actual = model._rm_final_layer_norm(inputs)
            reference_def = next(node for node in ast.parse(REPLAY_MIXER.read_text(encoding='utf-8')).body
                                 if isinstance(node, ast.FunctionDef) and node.name == 'layer_norm')
            namespace = {'tf': tf}
            exec(compile(module_node([reference_def]), str(REPLAY_MIXER), 'exec'), namespace)
            expected = namespace['layer_norm'](inputs, scope='reference', scale_factor=False)
            variables = tf.trainable_variables()
            self.assertEqual([v.shape.as_list() for v in variables], [[256]] * 4)
            gamma = np.linspace(0.7, 1.3, 256).astype(np.float32)
            beta = np.linspace(-0.2, 0.2, 256).astype(np.float32)
            assign = [tf.assign(v, gamma if '/gamma/' in v.op.name or v.op.name.endswith('/gamma') else beta)
                      for v in variables]
        rng = np.random.RandomState(20260902)
        with tf.Session(graph=graph, config=self.config) as session:
            session.run(assign)
            for data in (rng.normal(size=(1, 32, 256)).astype(np.float32),
                         rng.normal(size=(7, 32, 256)).astype(np.float32),
                         np.ones((3, 32, 256), np.float32)):
                left, right = session.run([expected, actual], {inputs: data})
                self.assertTrue(np.all(np.isfinite(right)))
                np.testing.assert_array_equal(left, right)

    def test_field_senet_dependency_routing(self):
        tf, np = self.tf, self.np
        graph = tf.Graph()
        with graph.as_default():
            model = kernel_model(tf, np)
            # Only the gate's dependency graph is under test, independently of Flood BN.
            model.use_senet_bn = False
            inputs = [tf.placeholder(tf.float32, [None, width]) for width in (6545, 14195, 238)]
            outputs = model.senet_layer(*inputs, is_train=False, export=False)
            dependencies = [[gradient is not None for gradient in tf.gradients(tf.reduce_sum(out), inputs)]
                            for out in outputs]
            self.assertEqual(dependencies, [[True, False, False], [True, True, False], [False, False, True]])
            init = tf.global_variables_initializer()
        rng = np.random.RandomState(20260902)
        data = [rng.normal(size=(3, width)).astype(np.float32) for width in (6545, 14195, 238)]
        with tf.Session(graph=graph, config=self.config) as session:
            session.run(init)
            original = session.run(outputs, dict(zip(inputs, data)))
            altered = list(data)
            altered[2] = data[2] + 3
            actual = session.run(outputs, dict(zip(inputs, altered)))
            for value in original + actual:
                self.assertTrue(np.all(np.isfinite(value)))
            np.testing.assert_array_equal(original[0], actual[0])
            np.testing.assert_array_equal(original[1], actual[1])
            self.assertFalse(np.array_equal(original[2], actual[2]))

    def test_tokenization_optimization_outputs_and_gradients(self):
        tf, np = self.tf, self.np
        graph = tf.Graph()
        with graph.as_default():
            model = kernel_model(tf, np, partitions=3)
            inputs = tf.placeholder(tf.float32, [None, 20740])
            cotangent = tf.placeholder(tf.float32, [None, 31, 256])
            columns = tf.split(inputs, [17] * 1220, axis=1)
            field_maps = {'creative': None}  # A creative read would fail immediately.
            offset = 0
            for bucket in model._LOCAL_BUCKET_NAMES:
                fields = [field for _, ids in model.rm_semantic_feature_groups[bucket] for field in ids]
                field_maps[bucket] = dict(zip(fields, columns[offset:offset + len(fields)]))
                offset += len(fields)
            with tf.variable_scope('tokenizer', reuse=tf.AUTO_REUSE, partitioner=model.partitioner):
                model.rm_optimize_tokenize = False
                expected = model._semantic_tokenize(field_maps, export=False)
                model.rm_optimize_tokenize = True
                actual = model._semantic_tokenize(field_maps, export=False)
            self.assertEqual(actual.shape.as_list(), [None, 31, 256])
            variables = tf.trainable_variables()
            self.assertEqual(sum(int(np.prod(v.shape.as_list())) for v in variables), 5325312)
            expected_grads = tf.gradients(tf.reduce_sum(expected * cotangent), [inputs] + variables)
            before = len(graph.get_operations())
            actual_grads = tf.gradients(tf.reduce_sum(actual * cotangent), [inputs] + variables)
            self.assertNotIn('StridedSliceGrad', [op.type for op in graph.get_operations()[before:]])
            self.assertTrue(all(grad is not None for grad in expected_grads + actual_grads))
            expected_grads = [tf.convert_to_tensor(grad) for grad in expected_grads]
            actual_grads = [tf.convert_to_tensor(grad) for grad in actual_grads]
            init = tf.global_variables_initializer()
        rng = np.random.RandomState(20260902)
        with tf.Session(graph=graph, config=self.config) as session:
            session.run(init)
            for batch_size in (1, 3, 7):
                data = rng.normal(0, 0.5, (batch_size, 20740)).astype(np.float32)
                probe = rng.normal(0, 0.02, (batch_size, 31, 256)).astype(np.float32)
                left, right = session.run([[expected] + expected_grads, [actual] + actual_grads],
                                          {inputs: data, cotangent: probe})
                for reference, candidate in zip(left, right):
                    self.assertTrue(np.all(np.isfinite(candidate)))
                    np.testing.assert_array_equal(reference, candidate)


if __name__ == '__main__':
    unittest.main()
