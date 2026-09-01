# cvr_mature_rankmixer_fst_v1 服务器运行说明

## 入口与参数

- 模型入口：`models.rankmixer.cvr_mature_rankmixer_fst_v1.MLPModel`
- 模型文件：`src/models/rankmixer/cvr_mature_rankmixer_fst_v1.py`
- 完整参数：`bash/set-cvr-mature-rankmixer-fst-v1-args.txt`
- 特征版本：`data.cvr.cvr_fea_v10_base_cold`
- 任务：仅 `fst_cvr_label -> first_cvr`

参数文件完整继承已成功运行的 RankMixer v10 外层 Flood 参数，只替换模型入口和 `model_args`。首日配置固定为 dense 冷启动、sparse checkpoint 恢复：

- `--ignore_dense_checkpoint=True`
- `--ignore_sparse_checkpoint=False`
- `--enable_dense_warmup=false`
- `--train_fuse_bn=false`

因此不可把生产 replay/Phalanx 参数（例如 `enable_phalanx`、`enable_rpy_neg_sampler`、`rpy_fst_cvr_label`、`data.cvr.cvr_fea_v10_rankmixer_v2`）混入本配置。

## 最小上传集合

如果服务器已经包含本仓库当前版本的公共 `src` 训练框架，模型专项最小集合是：

1. `src/models/rankmixer/cvr_mature_rankmixer_fst_v1.py`
2. `src/data/cvr/cvr_fea_v10_base_cold.py`
3. `src/data/cvr/cvr_fea_v10_base.py`
4. `bash/set-cvr-mature-rankmixer-fst-v1-args.txt`

第 3 项不能省略，因为 `cvr_fea_v10_base_cold.py` 直接继承它。服务器还必须已有与成功 RankMixer v1-v10 相同版本的 `src/main.py`、`models/model_base.py`、`data/feature.py`、`data/feature_config_base.py`、`framework/`、`utils/`、`validator/`，以及外部 TensorFlow 1.x/Flood 运行包。

如果不能确认服务器公共代码与当前仓库一致，稳妥的上传集合是整个 `src/` 加上述参数文件。无需上传 `new_rankmixer_0831/`，新模型也没有从该目录导入任何代码。

## 提交前检查

```bash
python3 -m py_compile src/models/rankmixer/cvr_mature_rankmixer_fst_v1.py
python3 -m unittest src.models.rankmixer.tests.test_cvr_mature_rankmixer_fst_v1_static -v
bash -n bash/set-cvr-mature-rankmixer-fst-v1-args.txt
```

实际提交前，只按业务日期同步调整 `train_dates`、`test_date`、`additional_checkpoint_dates` 和 sparse `checkpoint_import_dir`；模型入口、特征版本、冷启动开关及 `model_args` 架构字段应保持不变。
