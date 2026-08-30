import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT / "bash" / "ablation_20260814"


def load_args_file(name):
    path = CONFIG_DIR / name
    lines = path.read_text(encoding="utf-8").splitlines()
    module = lines[0]
    flags = {}
    for line in lines[1:]:
        if line.startswith("--") and "=" in line:
            key, value = line[2:].split("=", 1)
            flags[key] = value
    raw_model_args = flags["model_args"]
    model_args = json.loads(raw_model_args[1:-1])
    return module, flags, model_args


def class_literal(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    model_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MLPModel"
    )
    for node in model_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def normalized_class_methods(path):
    class NormalizeVersionStrings(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                value = (
                    node.value.replace("v5", "vX")
                    .replace("v6", "vX")
                    .replace("V5", "VX")
                    .replace("V6", "VX")
                )
                return ast.copy_location(ast.Constant(value=value), node)
            return node

    tree = NormalizeVersionStrings().visit(ast.parse(path.read_text(encoding="utf-8")))
    ast.fix_missing_locations(tree)
    model_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MLPModel"
    )
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in model_class.body
        if isinstance(node, ast.FunctionDef)
    }


def expected_rankmixer_params(hidden_dim, swiglu_hidden=704, flatten_dim=512):
    embedding_width = 20_978
    local_tokens = 31
    total_tokens = 32
    layers = 2
    pool_dim = 128

    input_bn = 41_956
    senet = 522_112
    local_tokenizer = embedding_width * hidden_dim + 2 * local_tokens * hidden_dim
    global_token = (
        embedding_width * hidden_dim + hidden_dim * hidden_dim + 3 * hidden_dim
    )
    one_swiglu_stage = total_tokens * (
        3 * hidden_dim * swiglu_hidden + 2 * swiglu_hidden + 2 * hidden_dim
    )
    mixer_blocks = layers * 2 * one_swiglu_stage
    final_rms_norm = total_tokens * hidden_dim
    global_pool = 2 * hidden_dim * pool_dim + 2 * pool_dim
    flatten_readout = (
        local_tokens * hidden_dim * flatten_dim + 2 * flatten_dim + 1
    )

    head_input = 2 * hidden_dim + flatten_dim
    head_dims = [2048, 2048, 256]
    task_head = 0
    previous = head_input
    for width in head_dims:
        # Dense kernel + bias, followed by trainable BN gamma and beta.
        task_head += previous * width + 3 * width
        previous = width
    task_head += previous + 1

    return sum(
        [
            input_bn,
            senet,
            local_tokenizer,
            global_token,
            mixer_blocks,
            final_rms_norm,
            global_pool,
            flatten_readout,
            task_head,
        ]
    )


class RankMixerAblation20260814StaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_args_file("00-base-dcnm-args.txt")
        cls.random_1024 = load_args_file(
            "10-rm-v5-balanced-random-d1024-args.txt"
        )
        cls.random_512 = load_args_file(
            "11-rm-v5-balanced-random-d512-args.txt"
        )
        cls.semantic_512 = load_args_file("12-rm-v6-semantic-d512-args.txt")

    def test_protocol_is_identical_and_dense_cold(self):
        runs = [self.base, self.random_1024, self.random_512, self.semantic_512]
        expected = {
            "train_dates": "2026-08-14:2026-08-14",
            "test_date": "2026-08-15:2026-08-15",
            "additional_checkpoint_dates": "2026-08-13:2026-08-13",
            "ignore_dense_checkpoint": "True",
            "ignore_sparse_checkpoint": "False",
            "strict_test_date": "True",
            "test_batch_num": "-1",
        }
        reference_checkpoint = runs[0][1]["checkpoint_import_dir"]
        for _, flags, model_args in runs:
            for key, value in expected.items():
                self.assertEqual(flags[key], value)
            self.assertEqual(flags["checkpoint_import_dir"], reference_checkpoint)
            self.assertFalse(model_args["enable_dense_warmup"])
            self.assertTrue(model_args["save_predict_result"])

    def test_v5_width_bridge_changes_only_hidden_dim(self):
        module_a, _, args_a = self.random_1024
        module_b, _, args_b = self.random_512
        self.assertEqual(module_a, module_b)
        self.assertEqual(args_a["rm_hidden_dim"], 1024)
        self.assertEqual(args_b["rm_hidden_dim"], 512)
        normalized_a = dict(args_a, rm_hidden_dim=512)
        self.assertEqual(normalized_a, args_b)

    def test_d512_grouping_bridge_is_parameter_matched(self):
        module_b, _, args_b = self.random_512
        module_c, _, args_c = self.semantic_512
        self.assertTrue(module_b.endswith("cvr_bn_rankmixer_v5.MLPModel"))
        self.assertTrue(module_c.endswith("cvr_bn_rankmixer_v6.MLPModel"))
        self.assertEqual(args_b["rm_hidden_dim"], 512)
        self.assertEqual(args_c["rm_hidden_dim"], 512)
        self.assertNotEqual(args_b["rm_group_version"], args_c["rm_group_version"])
        normalized_b = dict(args_b)
        normalized_c = dict(args_c)
        normalized_b.pop("rm_group_version")
        normalized_c.pop("rm_group_version")
        self.assertEqual(normalized_b, normalized_c)

    def test_frozen_group_abis_have_same_capacity_but_different_members(self):
        v5 = ROOT / "src" / "models" / "rankmixer" / "cvr_bn_rankmixer_v5.py"
        v6 = ROOT / "src" / "models" / "rankmixer" / "cvr_bn_rankmixer_v6.py"
        self.assertEqual(class_literal(v5, "_GROUP_SIZES"), class_literal(v6, "_GROUP_SIZES"))
        self.assertNotEqual(
            class_literal(v5, "_GROUP_CHECKSUMS"), class_literal(v6, "_GROUP_CHECKSUMS")
        )
        self.assertEqual(
            class_literal(v5, "_GROUP_VERSION"), "rankmixer_v5_balanced_v1"
        )
        self.assertEqual(
            class_literal(v6, "_GROUP_VERSION"),
            "rankmixer_v6_semantic_balanced_v1",
        )

    def test_v5_v6_non_grouping_methods_are_structurally_identical(self):
        v5 = ROOT / "src" / "models" / "rankmixer" / "cvr_bn_rankmixer_v5.py"
        v6 = ROOT / "src" / "models" / "rankmixer" / "cvr_bn_rankmixer_v6.py"
        methods_v5 = normalized_class_methods(v5)
        methods_v6 = normalized_class_methods(v6)
        self.assertEqual(set(methods_v5), set(methods_v6))
        allowed_to_differ = {
            "__init__",  # D default and v6's group-version startup guard.
            "_build_semantic_feature_groups",
            "_validate_semantic_feature_groups",
        }
        for name in sorted(set(methods_v5) - allowed_to_differ):
            self.assertEqual(methods_v5[name], methods_v6[name], name)
        self.assertEqual(len(methods_v5) - len(allowed_to_differ), 41)

    def test_exact_parameter_budgets(self):
        self.assertEqual(expected_rankmixer_params(1024), 348_432_486)
        self.assertEqual(expected_rankmixer_params(512), 177_217_126)
        manifest = json.loads((CONFIG_DIR / "manifest.json").read_text(encoding="utf-8"))
        params = {item["id"]: item["dense_params"] for item in manifest["experiments"]}
        self.assertEqual(params["E1_RANDOM_D1024"], expected_rankmixer_params(1024))
        self.assertEqual(params["E2_RANDOM_D512"], expected_rankmixer_params(512))
        self.assertEqual(params["E3_SEMANTIC_D512"], expected_rankmixer_params(512))
        analysis = manifest["prediction_analysis"]
        self.assertEqual(
            analysis["tool"],
            "src/models/rankmixer/tools/paired_auc_analysis.py",
        )
        self.assertEqual(
            analysis["finalization_tool"],
            "src/models/rankmixer/tools/finalize_ablation_report.py",
        )
        self.assertIn("jackknife", analysis["paired_inference"])

if __name__ == "__main__":
    unittest.main()
