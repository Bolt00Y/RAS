# -*- coding: utf-8 -*-
"""不导入 TensorFlow/Flood 的单文件 UniMixer v1 静态契约测试。"""

import ast
import json
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = PROJECT_ROOT / "src/models/rankmixer/cvr_bn_unimixer_v1.py"
BASE_MODEL_PATH = (
    PROJECT_ROOT / "src/models/seq_model/cvr_bn_senet_dcnm_fst.py"
)
ARGS_PATH = PROJECT_ROOT / "bash/set-unimixer-v1-args.txt"
DOCUMENT_PATH = PROJECT_ROOT / "introduce/unimixer_v1_introduction.md"
FEATURE_PATH = PROJECT_ROOT / "src/data/cvr/cvr_fea_v10_base.py"


def _assignment_literal(tree, name):
    for node in tree.body:
        if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            return ast.literal_eval(node.value)
    raise AssertionError("assignment not found: {}".format(name))


def _semantic_groups(tree):
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_semantic_feature_groups"
    )
    return_node = next(
        node for node in function.body if isinstance(node, ast.Return)
    )
    return ast.literal_eval(return_node.value)


def _feature_source_maps():
    tree = ast.parse(FEATURE_PATH.read_text(encoding="utf-8"))
    source_names = (
        "user_features",
        "item_features",
        "creative_features",
        "coupon_features",
    )
    result = {name: {} for name in source_names}
    for node in tree.body:
        if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in result
                and isinstance(node.value, ast.Call)
                and node.value.args):
            result[node.targets[0].id].update(
                ast.literal_eval(node.value.args[0])
            )
        elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "update"
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id in result
                and node.value.args):
            update_call = node.value.args[0]
            if isinstance(update_call, ast.Call) and update_call.args:
                result[node.value.func.value.id].update(
                    ast.literal_eval(update_call.args[0])
                )
    return result


class SingleFileUniMixerV1Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = MODEL_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.base_source = BASE_MODEL_PATH.read_text(encoding="utf-8")
        cls.base_tree = ast.parse(cls.base_source)
        cls.bucket_names = _assignment_literal(cls.tree, "BUCKET_NAMES")
        cls.bucket_token_counts = _assignment_literal(
            cls.tree,
            "EXPECTED_BUCKET_TOKEN_COUNTS",
        )
        cls.groups = _semantic_groups(cls.tree)
        cls.ordered_groups = [
            (bucket_name, group_name, feature_ids)
            for bucket_name in cls.bucket_names
            for group_name, feature_ids in cls.groups[bucket_name]
        ]
        cls.document = DOCUMENT_PATH.read_text(encoding="utf-8")

        lines = [
            line.strip()
            for line in ARGS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.model_entry = lines[0]
        flags = {}
        for line in lines[1:]:
            if not line.startswith("--") or "=" not in line:
                raise AssertionError("invalid args line: {}".format(line))
            key, value = line[2:].split("=", 1)
            flags[key] = value
        cls.flags = flags
        cls.model_args = json.loads(flags["model_args"].strip("'"))

    def _model_class(self):
        return next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MLPModel"
        )

    def _method(self, name):
        return next(
            node for node in self._model_class().body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def test_runtime_implementation_is_one_direct_file(self):
        self.assertTrue(MODEL_PATH.is_file())
        self.assertFalse(
            (PROJECT_ROOT / "src/models/rankmixer/unimixer_v1").exists()
        )
        self.assertEqual(
            "models.rankmixer.cvr_bn_unimixer_v1.MLPModel",
            self.model_entry,
        )
        self.assertNotIn("models.unimixer.unimixer", self.source)
        self.assertNotIn("models.rankmixer.unimixer_v1", self.source)
        self.assertNotIn("BaseCVRModel", self.source)
        self.assertNotIn("models.seq_model", self.source)

    def test_all_unimixer_algorithms_are_inlined(self):
        function_names = {
            node.name for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for required in (
                "build_semantic_feature_groups",
                "validate_semantic_feature_groups",
                "pertoken_swiglu",
                "anneal_tau",
                "to_doubly_stochastic",
                "unimixing_lite",
                "semantic_unimixer_stack"):
            self.assertIn(required, function_names)

        for scope_name in (
                '"um_semantic_tokenize"',
                '"semantic_unimixer"',
                '"unimixing_lite"',
                '"pertoken_swiglu"',
                '"um_task_tower"'):
            self.assertIn(scope_name, self.source)

    def test_feature_config_and_base_lifecycle_are_defined_directly(self):
        model_class = self._model_class()
        self.assertIsInstance(model_class.bases[0], ast.Name)
        self.assertEqual("ModelBase", model_class.bases[0].id)

        direct_methods = {
            node.name: node for node in model_class.body
            if isinstance(node, ast.FunctionDef)
        }
        original_class = next(
            node for node in self.base_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MLPModel"
        )
        original_methods = {
            node.name: node for node in original_class.body
            if isinstance(node, ast.FunctionDef)
        }

        # 原始 base 的所有入口都必须物理存在于当前单文件的 MLPModel 中；
        # __init__ 加入单目标约束，model_fn 替换为 UniMixer，因此只排除这两项
        # 的逐 AST 等价比较。
        self.assertTrue(set(original_methods).issubset(direct_methods))
        for method_name, original_method in original_methods.items():
            if method_name in {"__init__", "model_fn"}:
                continue
            self.assertEqual(
                ast.dump(original_method, include_attributes=False),
                ast.dump(direct_methods[method_name], include_attributes=False),
                method_name,
            )

        for required in (
                "get_features_conf",
                "get_share_embedding_conf",
                "get_dataset",
                "build",
                "build_loss_op",
                "build_optimizer_op",
                "train",
                "test",
                "predict",
                "export",
                "evaluate",
                "get_hooks"):
            self.assertIn(required, direct_methods)

    def test_semantic_group_contract_and_unique_coverage(self):
        self.assertEqual(("common", "item", "creative"), self.bucket_names)
        self.assertEqual((10, 21, 1), self.bucket_token_counts)
        self.assertEqual(
            self.bucket_token_counts,
            tuple(len(self.groups[name]) for name in self.bucket_names),
        )
        expected_fields = (385, 835, 14)
        all_seen = set()
        for bucket_name, expected_count in zip(
                self.bucket_names,
                expected_fields):
            ids = [
                feature_id
                for _, feature_ids in self.groups[bucket_name]
                for feature_id in feature_ids
            ]
            self.assertEqual(expected_count, len(ids))
            self.assertEqual(len(ids), len(set(ids)))
            self.assertFalse(all_seen.intersection(ids))
            all_seen.update(ids)
        self.assertEqual(1234, len(all_seen))

    def test_semantic_groups_exactly_cover_feature_source(self):
        source_maps = _feature_source_maps()
        self.assertFalse(source_maps["coupon_features"])
        bucket_to_source = {
            "common": "user_features",
            "item": "item_features",
            "creative": "creative_features",
        }
        for bucket_name, source_name in bucket_to_source.items():
            grouped_ids = {
                feature_id
                for _, feature_ids in self.groups[bucket_name]
                for feature_id in feature_ids
            }
            self.assertEqual(set(source_maps[source_name]), grouped_ids)

    def test_linear_then_independent_token_bn(self):
        method = self._method("_project_semantic_group")
        method_source = ast.get_source_segment(self.source, method)
        self.assertLess(
            method_source.index("fully_connected"),
            method_source.index("batch_norm_layer_v2"),
        )
        self.assertIn("group_name", method_source)
        self.assertIn('scope_bn="token_bn"', method_source)
        self.assertIn("use_riemann_bn=self.use_riemann_bn", method_source)
        self.assertIn("export=export", method_source)
        self.assertNotIn("gelu", method_source.lower())

    def test_server_model_args_match_fixed_architecture(self):
        expected = {
            "um_token_num": 32,
            "um_token_dim": 512,
            "um_bucket_token_counts": [10, 21, 1],
            "um_use_token_bn": True,
            "um_num_blocks": 2,
            "um_block_size": 32,
            "um_rank": 128,
            "um_num_bases": 8,
            "um_swiglu_expansion": 2,
            "use_senet": True,
            "use_senet_bn": True,
            "use_riemann_bn": True,
        }
        for key, value in expected.items():
            self.assertEqual(value, self.model_args[key], key)
        self.assertEqual("first_cvr", self.model_args["opt_goal"])
        self.assertEqual("fst_cvr_label", self.model_args["cvr_label_name"])
        for key in (
                "enable_last_cvr",
                "enable_wide_cvr",
                "enable_mlt_loss",
                "enable_delay_train_mode"):
            self.assertIs(False, self.model_args[key], key)

    def test_all_embedded_server_json_is_valid(self):
        for key in (
                "model_args",
                "ps_config_args",
                "tf_config_proto_args",
                "feature_parameter_args"):
            parsed = json.loads(self.flags[key].strip("'"))
            self.assertIsInstance(parsed, dict, key)

    def test_document_describes_single_file_contract(self):
        self.assertIn(
            "src/models/rankmixer/cvr_bn_unimixer_v1.py",
            self.document,
        )
        self.assertIn(
            "models.rankmixer.cvr_bn_unimixer_v1.MLPModel",
            self.document,
        )
        self.assertIn("直接继承 `ModelBase`", self.document)
        self.assertIn("`get_features_conf()` 现在直接定义", self.document)
        self.assertNotIn("自动解析到父类", self.document)
        for stale_path in (
                "src/models/rankmixer/unimixer_v1/",
                "models.rankmixer.unimixer_v1",
                "semantic_groups.py",
                "unimixer_v1/tests/"):
            self.assertNotIn(stale_path, self.document, stale_path)

    def test_document_field_appendix_matches_single_file(self):
        details = re.findall(
            r"<summary>Token (\d+): `([^`]+)`[^<]*"
            r"（(\d+) fields，输入 (\d+) 维）</summary>\n\n"
            r"(.*?)\n\n</details>",
            self.document,
            flags=re.DOTALL,
        )
        self.assertEqual(32, len(details))
        for index, detail in enumerate(details):
            actual_index, actual_name, field_count, input_dim, body = detail
            _, expected_name, expected_ids = self.ordered_groups[index]
            self.assertEqual(str(index), actual_index)
            self.assertEqual(expected_name, actual_name)
            self.assertEqual(len(expected_ids), int(field_count))
            self.assertEqual(len(expected_ids) * 17, int(input_dim))
            self.assertEqual(expected_ids, re.findall(r"`([^`]+)`", body))

    def test_document_main_token_table_matches_single_file(self):
        rows = re.findall(
            r"^\| (\d+) \| (common|item|creative) \| `([^`]+)` \|"
            r"[^|]+\| (\d+) \| (\d+) \| 512 \|$",
            self.document,
            flags=re.MULTILINE,
        )
        self.assertEqual(32, len(rows))
        for index, row in enumerate(rows):
            bucket_name, expected_name, expected_ids = self.ordered_groups[index]
            self.assertEqual(str(index), row[0])
            self.assertEqual(bucket_name, row[1])
            self.assertEqual(expected_name, row[2])
            self.assertEqual(len(expected_ids), int(row[3]))
            self.assertEqual(len(expected_ids) * 17, int(row[4]))


if __name__ == "__main__":
    unittest.main()
