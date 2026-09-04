# -*- coding: utf-8 -*-
# RankMixer v6-E2-Small-2：由 cvr_bn_rankmixer_v6_e2_small_1.py 发展的三层主干消融版本。
# 1. 输入塔保持 Base/v6 的三桶 BN、Hierarchical SENet 和冻结的 31 组语义字段映射。
# 2. 31 个 Local Token 与 1 个 Global Token 均投影为 D=256；投影后及三个
#    RankMixer Block 内部保留 v6 RMSNorm，末端改为 mature_v1 的共享参数 LayerNorm。
# 3. 主干仅将 RankMixer 堆叠数从 L=2 改为 L=3；H=T=32、D=256、
#    Mixing/Reverting 和 M=704 的双 Per-token SwiGLU 全部保持不变。
# 4. 末端 [B,32,256] 沿 Token 轴等权 mean pooling 为 [B,256]，再进入
#    mature_v1 风格的 [256,128] 单任务头；creative 保留在 v6 Token 主干中，不新增旁路。
# 5. Dense 可训练参数量为 115,664,741；该数值仅用于防止实现漂移，不作为实验约束；
#    不包含稀疏 Embedding 表、优化器状态、
#    指标变量和 BN moving statistics。
# 6. 保留原有投影与转置，仅用 Unpack 替代逐 Token 切片，消除重复梯度回填。
#    rm_optimize_tokenize=false 可切回 D=256 的参考执行路径。
# 7. 本版本相对 Small-1 仅增加一个同构 RankMixer Block，不承诺预测/AUC 等价。
# 8. 本文件是完整独立实现，不导入、不继承任何旧 RankMixer Python 实现。
import os
import math
import hashlib
from pydoc import locate

import tensorflow as tf
import logging
from logging import Formatter, getLogger, FileHandler

import flood
from data.feature import FeatureColumnBuilder
from flood.python.training.optimizer import FloodOptimizer
from flood.python.ops import parsing_ops
from framework.hooks.new_branch_warmup_hook import Senet2NewWarmupHook

from utils.accumulated_metrics import *
from flood.python.utils import lookup_utils
from utils.file_utils import upload_hdfs, mkdir_hdfs
from flood.python.ops.auc import flood_auc
from ..model_base import ModelBase

from utils.odds import get_sparse_fc_key
from flood.python.data import data_util as flood_data_util
from utils import learning_rate as learning_rate_utils


class MLPModel(ModelBase):
    _BUCKET_NAMES = ('common', 'item', 'creative')
    _EXPECTED_FIELD_COUNTS = (385, 835, 14)
    _EXPECTED_DENSE_TRAINABLE_PARAMS = 115664741
    _GROUP_VERSION = 'rankmixer_v6_semantic_balanced_v1'
    _GROUP_CHECKSUMS = {
        'common': '61602847a993a6103b9c21b4d6ff2d1817a848d8717e7b201eea4be6fc29bda3',
        'item': '0517491a05e73f3aac890cc3f9ab900b795da05011914c842c2715ff30af49e3',
        'creative': '956056a173d6daa8b62602b62bf9bd83c638e362c6824aa9cd2ef1300490d10c',
    }
    _GROUP_SIZES = {
        'common': (39, 39, 39, 39, 39, 38, 38, 38, 38, 38),
        'item': (42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                 42, 42, 42, 42, 42, 41, 41, 41, 41, 41),
        'creative': (14,),
    }

    def __init__(self, **_kwargs):
        for key, value in _kwargs.items():
            setattr(self, key, value)

        self.batch_size = _kwargs.get('batch_size', 2048)
        self.eval_batch_size = _kwargs.get('eval_batch_size', 20480)
        self.l2_deep = _kwargs.get('l2_deep', 0.000001)
        self.grad_clip_value = _kwargs.get('grad_clip_value', 15)
        self.dropout = _kwargs.get('dropout', None)
        self.max_partitions = _kwargs.get('max_partitions', None)
        self.act_type = _kwargs.get('act_type', 'relu')
        self.init_type = _kwargs.get('init_type', 'xavier')
        self.embedding_size = _kwargs.get('embedding_size', 17)
        self.pretrain_embedding_size = _kwargs.get('pretrain_embedding_size', 64)
        self.log_nn_vars = _kwargs.get('log_nn_vars', False)

        # tf config
        self.tf_config = _kwargs.get('tf_config', None)
        self.worker_id = self.tf_config['task']['index']
        self.is_chief = self.worker_id == 0

        # warmup conf
        self.enable_dense_warmup = _kwargs.get("enable_dense_warmup", False)
        self.enable_mlt_warmup = _kwargs.get("enable_mlt_warmup", False)
        self.hooks = _kwargs.get('hooks', [])
        self.skip_tensors = _kwargs.get("skip_tensors", "")
        self.warm_up_tensors = _kwargs.get("warm_up_tensors", "")
        self.warmup_type = _kwargs.get('warmup_type', 'default')
        self.warm_mlp_layer = _kwargs.get("warm_mlp_layer", [])
        self.use_mlp_gate = _kwargs.get('use_mlp_gate', False)
        self.old_epoch_ckpt_import_dir = _kwargs.get("old_epoch_ckpt_import_dir", None)
        self.ckpt_import_dir1 = _kwargs.get("ckpt_import_dir1", None)
        self.ckpt_import_dir2 = _kwargs.get("ckpt_import_dir2", None)
        self.warm_up_tensors1 = _kwargs.get("warm_up_tensors1", "")
        self.dense_tuning = _kwargs.get('dense_tuning', False)

        # bn conf
        self.batch_norm = _kwargs.get('batch_norm', False)
        self.batch_norm_decay = _kwargs.get('batch_norm_decay', 0.9)
        self.mlp_act_type = _kwargs.get('mlp_act_type', 'gelu_2')
        self.use_riemann_bn = _kwargs.get('use_riemann_bn', True)
        self.clip_val = _kwargs.get('clip_val', 50)
        self.embed_use_renorm = _kwargs.get('embed_use_renorm', False)
        self.embed_renorm_decay = _kwargs.get('embed_renorm_decay', 0.99)

        # optimizer conf
        self.optimizer = _kwargs.get('optimizer', 'Adagrad')

        # learning rate conf
        self.decay = _kwargs.get('decay', '')
        self.learning_rate = _kwargs.get('learning_rate', 0.00001)
        self.schedule_config = _kwargs.get('schedule_config',
                                           {'type': 'gauss_decay', 'warmup_steps': 60000, 'decay_steps': 40000,
                                            'min_rate': 0.1})
        for schedule_cf in self.schedule_config.items():
            logging.info(f"schedule_cf is: {schedule_cf}")

        # predict and model conf
        self.model_dir = _kwargs.get('model_dir', None)
        self.predict_path = _kwargs.get('predict_path', None)
        self.timeout = int(_kwargs.get('timeout', 60 * 20) * 1000)
        self.upload_log = _kwargs.get('upload_log', False)
        self.save_predict_result = _kwargs.get('save_predict_result', False)

        # 两阶段参数
        self.ps_stage = _kwargs.get('ps_stage', 'update')
        self.update_model_dir = _kwargs.get('update_model_dir', None)

        # cvr fea conf
        try:
            # 特征配置路径，demo data.cvr.cvr_feature_config_v7
            self.feature_version = _kwargs.get('feature_version', None)
            self.feature_version_old = _kwargs.get('feature_version_old', self.feature_version)

            module = locate(self.feature_version)
            module_old = locate(self.feature_version_old)

            logging.info(f"feature_version is {self.feature_version} \n"
                         f"feature_version_old is {self.feature_version_old}")

        except Exception:
            raise ValueError('feature_version: {} not valid'.format(self.feature_version))

        self.fea_conf_obj = module.FeatureConfig()
        self.fea_conf_obj_old = module_old.FeatureConfig()

        self.features = FeatureColumnBuilder(feature_config=self.fea_conf_obj,
                                             default_embedding_size=self.embedding_size)
        self.features_old = FeatureColumnBuilder(feature_config=self.fea_conf_obj_old,
                                                 default_embedding_size=self.embedding_size)

        # sequence conf
        self.default_sequence_len = _kwargs.get('default_sequence_len', 100)

        # SENet configuration retained exactly as the strong base input tower.
        self.senet_hidden_size = _kwargs.get('senet_hidden_size', 128)
        self.use_senet = _kwargs.get('use_senet', False)
        self.use_senet_bn = _kwargs.get('use_senet_bn', False)

        # cvr model conf
        self.cvr_layers = [int(value) for value in _kwargs.get('cvr_layers', [256, 128])]
        if not self.cvr_layers or any(value <= 0 for value in self.cvr_layers):
            raise ValueError('cvr_layers must contain positive dimensions')
        self.opt_goal = _kwargs.get('opt_goal', 'first_cvr')
        self.export_name = _kwargs.get('export_name', 'first_cvr')
        self.cvr_label_name = _kwargs.get('cvr_label_name', 'fst_cvr_label')

        # RankMixer v6-E2-Small-2: 31 local tokens + 1 global token, H=T=32, D=256, L=3.
        self.use_rankmixer = _kwargs.get('use_rankmixer', True)
        self.rm_token_num = int(_kwargs.get('rm_token_num', 32))
        self.rm_local_token_num = int(_kwargs.get('rm_local_token_num', 31))
        self.rm_hidden_dim = int(_kwargs.get('rm_hidden_dim', 256))
        self.rm_layer_num = int(_kwargs.get('rm_layer_num', 3))
        self.rm_head_num = _kwargs.get('rm_head_num', self.rm_token_num)
        self.rm_head_num = int(self.rm_head_num)
        self.rm_swiglu_hidden_dim = int(_kwargs.get('rm_swiglu_hidden_dim', 704))
        self.rm_down_init_scale = float(_kwargs.get('rm_down_init_scale', 0.01))
        self.rm_rms_epsilon = float(_kwargs.get('rm_rms_epsilon', 1e-6))
        self.rm_token_proj_act = _kwargs.get('rm_token_proj_act', 'gelu_2')
        self.rm_norm_type = _kwargs.get('rm_norm_type')
        self.rm_final_norm_type = _kwargs.get('rm_final_norm_type', 'layer_norm')
        self.rm_final_ln_epsilon = float(_kwargs.get('rm_final_ln_epsilon', 1e-8))
        self.rm_readout_type = _kwargs.get('rm_readout_type', 'mean_pool')
        self.rm_optimize_tokenize = _kwargs.get('rm_optimize_tokenize', True)
        if not isinstance(self.rm_optimize_tokenize, bool):
            raise ValueError('rm_optimize_tokenize must be a JSON boolean')

        if not self.use_rankmixer:
            raise ValueError('RankMixer v6-E2-Small-2 requires use_rankmixer=true')
        if self.embedding_size != 17:
            raise ValueError('RankMixer v6-E2-Small-2 requires embedding_size=17, got {}'.format(self.embedding_size))
        if self.rm_token_num != self.rm_local_token_num + 1:
            raise ValueError(
                'rm_token_num={} must equal rm_local_token_num+1={}'.format(
                    self.rm_token_num,
                    self.rm_local_token_num + 1,
                )
            )
        if self.rm_head_num != self.rm_token_num:
            raise ValueError('RankMixer v6-E2-Small-2 requires rm_head_num == rm_token_num')
        if self.rm_hidden_dim % self.rm_head_num != 0:
            raise ValueError(
                'rm_hidden_dim={} must be divisible by rm_head_num={}'.format(
                    self.rm_hidden_dim,
                    self.rm_head_num,
                )
            )
        if self.rm_swiglu_hidden_dim <= 0:
            raise ValueError('rm_swiglu_hidden_dim must be positive')
        if self.rm_layer_num <= 0:
            raise ValueError('rm_layer_num must be positive')
        if not 0.0 <= self.rm_down_init_scale <= 1.0:
            raise ValueError('rm_down_init_scale must be in [0, 1]')
        if self.rm_norm_type != 'rms_norm':
            raise ValueError(
                'RankMixer v6-E2-Small-2 requires rm_norm_type=rms_norm, got {}'.format(
                    self.rm_norm_type
                )
            )
        if self.rm_rms_epsilon <= 0.0:
            raise ValueError('RankMixer v6-E2-Small-2 requires rm_rms_epsilon > 0')
        if self.rm_final_norm_type != 'layer_norm':
            raise ValueError(
                'RankMixer v6-E2-Small-2 requires rm_final_norm_type=layer_norm'
            )
        if self.rm_final_ln_epsilon <= 0.0:
            raise ValueError('RankMixer v6-E2-Small-2 requires rm_final_ln_epsilon > 0')
        if self.rm_readout_type != 'mean_pool':
            raise ValueError('RankMixer v6-E2-Small-2 requires rm_readout_type=mean_pool')

        unsupported_buckets = {
            'coupon': self.fea_conf_obj.coupon_fea_map,
            'dense': self.fea_conf_obj.dense_fea_map,
            'sequence': self.fea_conf_obj.seq_fea_map,
            'gattr': self.fea_conf_obj.gattr_fea_map,
            'din': self.fea_conf_obj.din_fea_map,
        }
        nonempty_unsupported = {
            name: len(mapping) for name, mapping in unsupported_buckets.items() if mapping
        }
        if nonempty_unsupported:
            raise ValueError(
                'RankMixer v6-E2-Small-2 accepts only common/item/creative; '
                'non-empty extra buckets: {}'.format(nonempty_unsupported)
            )

        field_counts = [
            len(self.fea_conf_obj.common_fea_map),
            len(self.fea_conf_obj.item_fea_map),
            len(self.fea_conf_obj.creative_fea_map),
        ]
        self.rm_group_version = _kwargs.get('rm_group_version', self._GROUP_VERSION)
        if self.rm_group_version != self._GROUP_VERSION:
            raise ValueError(
                'rm_group_version={} must match model group version={}'.format(
                    self.rm_group_version,
                    self._GROUP_VERSION,
                )
            )
        self.rm_semantic_feature_groups = self._build_semantic_feature_groups()
        self._validate_semantic_feature_groups()
        semantic_bucket_token_counts = [
            len(self.rm_semantic_feature_groups[name]) for name in self._BUCKET_NAMES
        ]
        configured_counts = _kwargs.get('rm_bucket_token_counts')
        if configured_counts is not None:
            configured_counts = [int(value) for value in configured_counts]
            if configured_counts != semantic_bucket_token_counts:
                raise ValueError(
                    'rm_bucket_token_counts={} must match hard-coded semantic groups={}'.format(
                        configured_counts, semantic_bucket_token_counts
                    )
                )
        self.rm_bucket_token_counts = semantic_bucket_token_counts

        if sum(self.rm_bucket_token_counts) != self.rm_local_token_num:
            raise ValueError(
                'hard-coded local token count={} must equal rm_local_token_num={}'.format(
                    sum(self.rm_bucket_token_counts), self.rm_local_token_num
                )
            )

        required_architecture = {
            'senet_hidden_size': (self.senet_hidden_size, 128),
            'rm_token_num': (self.rm_token_num, 32),
            'rm_local_token_num': (self.rm_local_token_num, 31),
            'rm_hidden_dim': (self.rm_hidden_dim, 256),
            'rm_layer_num': (self.rm_layer_num, 3),
            'rm_head_num': (self.rm_head_num, 32),
            'rm_swiglu_hidden_dim': (self.rm_swiglu_hidden_dim, 704),
        }
        for arg_name, (actual_value, required_value) in required_architecture.items():
            if actual_value != required_value:
                raise ValueError(
                    'RankMixer v6-E2-Small-2 requires {}={}, got {}'.format(
                        arg_name, required_value, actual_value
                    )
                )
        if tuple(field_counts) != self._EXPECTED_FIELD_COUNTS:
            raise ValueError(
                'RankMixer v6-E2-Small-2 requires field counts={}, got {}'.format(
                    self._EXPECTED_FIELD_COUNTS,
                    tuple(field_counts),
                )
            )
        if self.cvr_layers != [256, 128]:
            raise ValueError('RankMixer v6-E2-Small-2 requires cvr_layers=[256, 128]')
        if not self.use_senet or not self.use_senet_bn or not self.batch_norm:
            raise ValueError(
                'RankMixer v6-E2-Small-2 requires use_senet/use_senet_bn/batch_norm all true'
            )
        if self.use_mlp_gate:
            raise ValueError('RankMixer v6-E2-Small-2 requires use_mlp_gate=false for a single task path')
        if self.rm_token_proj_act != 'gelu_2' or self.mlp_act_type != 'gelu_2':
            raise ValueError('RankMixer v6-E2-Small-2 requires GELU2 projections and task head')
        if abs(self.rm_down_init_scale - 0.01) > 1e-12:
            raise ValueError('RankMixer v6-E2-Small-2 requires rm_down_init_scale=0.01')

        self.rm_expected_dense_trainable_params = self._EXPECTED_DENSE_TRAINABLE_PARAMS
        self.rm_dense_trainable_param_count = self._calculate_dense_trainable_params()
        if self.rm_dense_trainable_param_count != self.rm_expected_dense_trainable_params:
            raise ValueError(
                'RankMixer v6-E2-Small-2 dense trainable parameter count={}, expected={}'.format(
                    self.rm_dense_trainable_param_count,
                    self.rm_expected_dense_trainable_params,
                )
            )
        logging.info(
            'RankMixer v6-E2-Small-2: group_version=%s, fields=%s, local_bucket_tokens=%s, '
            'T=%d, H=%d, D=%d, L=%d, swiglu_hidden=%d, down_init_scale=%s, '
            'token_proj_act=%s, senet=%s, block_norm=%s, final_norm=%s, '
            'readout=%s, head=%s',
            self._GROUP_VERSION,
            field_counts,
            self.rm_bucket_token_counts,
            self.rm_token_num,
            self.rm_head_num,
            self.rm_hidden_dim,
            self.rm_layer_num,
            self.rm_swiglu_hidden_dim,
            self.rm_down_init_scale,
            self.rm_token_proj_act,
            self.use_senet,
            self.rm_norm_type,
            self.rm_final_norm_type,
            self.rm_readout_type,
            self.cvr_layers,
        )
        logging.info(
            'RankMixer v6-E2-Small-2 tokenization: optimized=%s; '
            'same projection/layout/activation; eliminate repeated slice-gradient buffers',
            self.rm_optimize_tokenize,
        )

        # dense 相关
        self.dense_scale = _kwargs.get("dense_scale", 0.01)
        self.dense_global_norm = _kwargs.get("dense_global_norm", True)
        self.dense_clip_threshold = _kwargs.get("dense_clip_threshold", [-2000000.0, 2000000.0])

        # train data conf
        self.epochs = _kwargs.get('epochs', None)
        self.prefetch_num = _kwargs.get('prefetch_num', 100)
        self.interleave = _kwargs.get('interleave', 8)
        self.test_interleave = _kwargs.get('test_interleave', 8)
        self.sampler_stat = _kwargs.get('sampler_stat', False)
        self.async_pull = _kwargs.get('async_pull', False)
        self.test_async_pull = _kwargs.get('test_async_pull', True)
        self.max_prefetched_pull = _kwargs.get('max_prefetched_pull', -1)
        self.test_batch_num = _kwargs.get('test_batch_num', 4000 * 10000)
        self.drop_last_files = _kwargs.get('drop_last_files', 2)
        self.slow_worker_timeout = _kwargs.get('slow_worker_timeout', 3600000)
        self.slow_worker_num_limit = _kwargs.get('slow_worker_num_limit', 0)
        self.train_stage_param = _kwargs.get('train_stage_param', 'replay##dist2')
        self.sampler_label_name = _kwargs.get('sampler_label_name', '')
        self.sampler_positive_rate = _kwargs.get('sampler_positive_rate', 1.0)
        self.sampler_negative_rate = _kwargs.get('sampler_negative_rate', 1.0)
        self.enable_neg_sampler = _kwargs.get('enable_neg_sampler', True)
        self.filter_pass_values = _kwargs.get('filter_pass_values', '')
        self.filter_label_names = _kwargs.get('filter_label_names', '')
        self.filter_drop_values = _kwargs.get('filter_drop_values', '')
        self.filter_pass_empty = _kwargs.get('filter_pass_empty', True)

        self.eval_count = 0
        self.num_ps = 1
        self.num_worker = 1
        if self.tf_config:
            self.num_ps = len(self.tf_config["cluster"]["ps"])
            self.num_worker = len(self.tf_config["cluster"]["worker"])

        self.task_index = self.tf_config['task']['index']

        self.train_reset_interval = _kwargs.get('train_reset_interval', 10000)
        self.train_reset_count = 0

        self.strict_test_date = _kwargs.get('strict_test_date', False)
        self.order_by_date = _kwargs.get('order_by_date', False)
        self.random_feature = _kwargs.get('random_feature', None)
        self.parallel_feature_analysis = _kwargs.get('parallel_feature_analysis', False)

        if _kwargs.get('log_gflags', True) and self.random_feature is None:
            self.list_all_member()

        self.train_count = 0

        # flood 需要的参数，暂时不能删除
        self.fq_table_config = _kwargs.get('fq_table_config', 'shrink_only_config')
        self.seq_add_dim = _kwargs.get('seq_add_dim', 0)
        self.dir2_all_tensor = _kwargs.get('dir2_all_tensor', "None")
        self.second_epoch_ckpt_import_dir = _kwargs.get('second_epoch_ckpt_import_dir', '')
        self.ffn_version = _kwargs.get('ffn_version', 'v1')
        self.scale_type = _kwargs.get('scale_type', 0)

        super().__init__()

    @staticmethod
    def _build_semantic_feature_groups():
        """Return the frozen v6 semantic-balanced field groups."""
        # Runtime never hashes or reshuffles fields. Token order and membership are model ABI.
        return {
            'common': [
                # 用户画像、设备、地域、生命周期与购买力（39 个字段）
                ('common_user_profile_device_geo_lifecycle', [
                    '1001', '1006', '1014', '1034', '1035', '1036', '1041', '1042',
                    '1043', '1501', '1504', '1505', '1527', '25003', '866024', '868023',
                    '10231', '10232', '10233', '10522', '10601', '1502', '19013', '19016',
                    '201704', '20517', '20521', '21403', '21404', '25000', '25001', '25002',
                    '25006', '25044', '25700', '340121', '790249', '790250', '790251',
                ]),
                # 用户下单、购买与消费价值（39 个字段）
                ('common_user_order_consumption_value', [
                    '10442', '1104', '1106', '200306', '201702', '201914', '201915', '202218',
                    '2066', '210000', '210001', '21264', '231056', '231065', '24082411', '24082412',
                    '24082413', '26017', '26021', '26025', '26107', '863712', '863729', '866012',
                    '866014', '866023', '866027', '866029', '866054', '866064', '866065', '866070',
                    '870277', '870324', '795602', '340123', '340125', '110153', '110011',
                ]),
                # 历史购买价格、行为时距与复购信号（39 个字段）
                ('common_user_purchase_price_recency', [
                    '131480', '16725', '16727', '16731', '16733', '16735', '16737', '16739',
                    '200300', '200302', '200303', '200304', '200320', '201756', '201757', '201939',
                    '21749', '21750', '862355', '862376', '866034', '866063', '866069', '33866903',
                    '870322', '340093', '110151', '340122', '200318', '200319', '204530', '204543',
                    '241125006', '4418196', '860031', '860034', '860037', '860042', '860045',
                ]),
                # 长期浏览、曝光与实体兴趣（39 个字段）
                ('common_longterm_view_exposure_interest', [
                    '10600', '1063', '1064', '1065', '1509', '1512', '1521', '1524',
                    '18021', '18094', '18098', '18100', '18105', '18214', '19024', '200124',
                    '21055', '21233', '21238', '21239', '21240', '21246', '21257', '21258',
                    '21260', '4500', '4501', '4502', '4503', '4504', '863014', '866072',
                    '870311', '2017702', '4418192', '031090', '340063', '340092', '340054',
                ]),
                # 长期点击、收藏、停留与行为兴趣（39 个字段）
                ('common_longterm_click_fav_interest', [
                    '1121', '12403', '12438', '200714', '200715', '200762', '200764', '201720',
                    '201905', '201906', '201909', '202144', '2073', '210015', '210042', '21355',
                    '21602', '21610', '231383', '231384', '231484', '25045', '25046', '2509',
                    '25702', '25703', '26035', '33600031', '863018', '866041', '866066', '866068',
                    '866071', '866082', '867603', '340059', '340086', '340037', '340001',
                ]),
                # Query 文本、NER、词项与搜索意图（38 个字段）
                ('common_query_text_intent', [
                    '12209', '12402', '15000', '15002', '16743', '16744', '25136', '25138',
                    '27516', '3006', '3007', '3008', '3009', '6910', '6911', '6912',
                    '790220', '790221', '790222', '790230', '794734', '794768', '794802', '863044',
                    '863046', '866013', '2014601', '2015703', '160034', '87560211', '340453', '340394',
                    '340364', '340374', '340367', '340451', '340377', '340455',
                ]),
                # Query 召回、候选命中与相关性上下文（38 个字段）
                ('common_query_retrieval_relevance', [
                    '200200', '200214', '200758', '2104', '211121', '2112', '600154', '7007737',
                    '7007741', '7007746', '7007755', '794014', '794015', '794030', '794031', '794164',
                    '794178', '794179', '794200', '794208', '794209', '794210', '794214', '794215',
                    '863141', '866250', '866251', '868413', '868414', '870025', '795014', '795012',
                    '2015709', '2015745', '2022401', '300091', '306045', '310614',
                ]),
                # 实时会话动作与短周期行为（38 个字段）
                ('common_realtime_session_action', [
                    '13037', '13038', '13039', '200752', '201930', '201931', '201932', '201937',
                    '21010', '21012', '21013', '2123', '2503', '2504', '25049', '2505',
                    '2506', '2507', '300000', '3014', '3015', '3016', '6900', '860023',
                    '861818', '862311', '863030', '866061', '868404', '868405', '868407', '868427',
                    '870038', '881402', '881404', '4439006', '340109', '340160',
                ]),
                # 短期曝光、点击与候选漏斗（38 个字段）
                ('common_shortterm_candidate_funnel', [
                    '12235', '18073', '18078', '18083', '200413', '201900', '202223', '202330',
                    '202333', '202334', '21351', '21359', '21402', '21663', '865600', '866103',
                    '867645', '867648', '870059', '870069', '870130', '881102', '881104', '881665',
                    '881687', '881691', '881711', '882303', '882304', '882306', '882305', '881820',
                    '881816', '881842', '881818', '881834', '881817', '304322',
                ]),
                # 页面、位置、时间与搜索会话上下文（38 个字段）
                ('common_shortterm_funnel_page_context', [
                    '21303', '21307', '21340', '20518', '2100', '2101', '2102', '2103',
                    '215401', '246003', '24904004', '3001', '1070', '200305', '202425', '202426',
                    '21030', '21031', '21032', '21033', '215311', '215312', '215343', '215373',
                    '24082402', '3003', '3004', '3020', '3102', '3103', '863024', '866073',
                    '881203', '881204', '881206', '881215', '881663', '881664',
                ]),
            ],
            'item': [
                # 商品、类目、品牌与候选身份（42 个字段）
                ('item_goods_category_brand_identity', [
                    '6007', '10003', '10013', '10014', '10016', '10018', '10020', '10021',
                    '10022', '10012', '10062', '1086', '13020', '13021', '13022', '17194',
                    '200313', '27631', '302185', '5001', '6001', '6004', '6008', '6013',
                    '6021', '6501', '7001', '7501', '10068', '10410', '24021', '1600912',
                    '19041', '19042', '19044', '19047', '200727', '200729', '201705', '24082417',
                    '241215001', '241215101',
                ]),
                # 店铺、静态质量、服务与属性（42 个字段）
                ('item_shop_static_quality_service', [
                    '600022', '6206', '10059', '200311', '200314', '206056', '500000', '500300',
                    '500301', '500302', '600024', '820000', '820004', '27632', '160070', '302503',
                    '302552', '302595', '304911', '304946', '304952', '340076', '5014', '600100',
                    '600101', '6012', '6016', '7002', '7007708', '7007710', '7007711', '7007713',
                    '820001', '820025', '820061', '881226', '881237', '881709', '881721', '881733',
                    '881757', '2015723',
                ]),
                # 标题、Query、词项与 NER 字面相关性（42 个字段）
                ('item_title_query_lexical_ner', [
                    '18504', '6893', '25116', '6892', '13009', '13010', '18501', '18503',
                    '25113', '25120', '340483', '341358', '4012', '6871', '6894', '7809',
                    '13005', '28013', '28017', '28019', '28023', '2115', '25106', '28003',
                    '341105', '341353', '4007', '4009', '6870', '7806', '13006', '8112',
                    '87560214', '10219', '10419', '24808118', '4003', '5410', '6914', '8501',
                    '87560127', '87560133',
                ]),
                # 语义、类目与文本向量相关性（42 个字段）
                ('item_semantic_category_relevance', [
                    '211100', '340100', '341320', '37615', '37616', '37617', '37618', '770584',
                    '3402761', '13002', '211130', '3400141', '340044', '6888', '8502', '204202',
                    '33204162', '33204180', '33204182', '862616', '862844', '864132', '864157', '864215',
                    '204242', '33204187', '864386', '864410', '340116', '3401661', '341265', '33866914',
                    '33868929', '865682', '882235', '770656', '770657', '770607', '864553', '865093',
                    '865344', '865349',
                ]),
                # 图像、视频与多模态向量相似性（42 个字段）
                ('item_image_video_embedding_similarity', [
                    '200640', '200780', '201021', '203742', '212502', '864743', '864744', '864770',
                    '864774', '865118', '865275', '865416', '865421', '33203301', '33203302', '33203303',
                    '33203308', '33203320', '33203330', '33203332', '33203333', '160077', '2015493', '206201',
                    '206301', '206389', '206563', '206585', '212402', '212422', '212432', '33203334',
                    '33205180', '33205227', '4418073', '621856', '6802', '870001', '882223', '882225',
                    '882227', '882233',
                ]),
                # 当前价格、SKU 供给与商品价值（42 个字段）
                ('item_current_price_supply', [
                    '16759', '27303', '6046', '863060', '27308', '22102', '22119', '302533',
                    '500150', '500151', '131485', '21762', '27443', '27445', '27459', '27606',
                    '770521', '500103', '16728', '16742', '16746', '20512', '500158', '6133',
                    '6134', '6859', '241215065', '4017', '500003', '500015', '6041', '12204',
                    '12205', '12206', '201716', '201735', '206550', '302502', '861219', '870008',
                    '870012', '870303',
                ]),
                # 优惠券、促销、折扣与活动供给（42 个字段）
                ('item_coupon_promotion_discount', [
                    '22120', '27635', '276351', '500121', '500136', '500137', '868029', '868030',
                    '10524', '140707', '27447', '27626', '27634', '27640', '500120', '500134',
                    '500135', '868291', '10528', '24530', '27311', '27316', '27321', '274471',
                    '622316', '622555', '6852', '780011', '2022429', '2022444', '16726', '27102',
                    '27616', '500159', '622530', '622533', '10520', '110041', '140700', '500001',
                    '864219', '870310',
                ]),
                # 用户购买价格与消费偏好（42 个字段）
                ('item_user_purchase_price_preference', [
                    '131474', '131475', '131476', '131478', '131479', '131482', '203708', '208000',
                    '208001', '770461', '770462', '770470', '770471', '10359', '11006', '131466',
                    '131467', '131468', '131470', '131472', '131473', '131483', '131484', '160065',
                    '206081', '206082', '21702', '21743', '21746', '21752', '22106', '33204181',
                    '900086', '206510', '22101', '600001', '881108', '10216', '10387', '10388',
                    '25027', '5019',
                ]),
                # 用户浏览点击价格偏好（42 个字段）
                ('item_user_view_click_price_preference', [
                    '200181', '21708', '21728', '22129', '22131', '24330', '24332', '27402',
                    '3401321', '4418101', '770460', '770469', '867665', '867685', '870313', '870315',
                    '206077', '215393', '21668', '21669', '21726', '21729', '246004', '246005',
                    '246006', '246007', '246014', '33203310', '33203311', '33203312', '33203321', '33203331',
                    '33866909', '33866912', '33866915', '33866926', '33868954', '33868970', '33868978', '870177',
                    '208030', '208034',
                ]),
                # 价格差、价格排序与竞争力（42 个字段）
                ('item_price_gap_rank_competitiveness', [
                    '340824', '24541', '206310', '208011', '208012', '208013', '208014', '208015',
                    '21760', '24328', '24496', '24497', '24498', '27367', '27507', '33204185',
                    '33204186', '33204196', '33205186', '33795608', '33795609', '33795610', '33868952', '33868953',
                    '33868961', '33868965', '33868969', '33868973', '33868976', '33868977', '794165', '794201',
                    '794212', '794213', '900643', '208016', '310601', '310602', '310604', '6011',
                    '60119', '6047',
                ]),
                # 商品类目全局漏斗统计（42 个字段）
                ('item_goods_category_global_funnel', [
                    '10207', '10213', '10154', '10160', '10310', '24108', '24116', '24121',
                    '24701', '24703', '24705', '24707', '24708', '24709', '24711', '24710',
                    '25506', '600233', '25504', '10152', '24218', '24246', '621415', '810107',
                    '810109', '820003', '19035', '24702', '24704', '24706', '25501', '25717',
                    '340028', '341102', '25515', '33758666', '10210', '6131', '600253', '600255',
                    '12122', '212611',
                ]),
                # 店铺品牌全局质量统计（42 个字段）
                ('item_shop_brand_global_quality', [
                    '6224', '6804', '10407', '10413', '25008', '25015', '621414', '622496',
                    '863069', '160067', '24107', '24115', '25010', '25011', '25012', '600102',
                    '241215011', '241215038', '24231', '24237', '24242', '24531', '25014', '304913',
                    '600112', '6052', '621412', '621416', '621872', '621877', '621878', '863056',
                    '6811', '770568', '810103', '810132', '7502', '868036', '302190', '600200',
                    '600201', '600202',
                ]),
                # 购买、下单与收藏正向亲和（42 个字段）
                ('item_purchase_order_fav_affinity', [
                    '12157', '25093', '4014', '770560', '870279', '131048', '131049', '21053',
                    '200106', '21054', '21201', '21202', '26003', '26007', '302554', '10010',
                    '1110', '1111', '12100', '12118', '12119', '12120', '17033', '200105',
                    '200310', '200325', '201809', '201916', '202096', '206029', '2111', '25073',
                    '25711', '25721', '25741', '302302', '304393', '3400731', '341103', '341421',
                    '870270', '203797',
                ]),
                # 长期曝光浏览亲和（42 个字段）
                ('item_longterm_exposure_view_affinity', [
                    '21051', '200104', '200765', '21050', '21052', '304395', '621842', '861534',
                    '869300', '870257', '304394', '25059', '7007715', '7007716', '870263', '870264',
                    '231334', '231344', '231374', '304451', '304452', '770459', '770468', '863132',
                    '863210', '1602601', '1602631', '200324', '200615', '200751', '200753', '21034',
                    '21035', '21036', '21037', '215334', '215337', '25048', '3401371', '870166',
                    '863054', '340756',
                ]),
                # 点击、停留与深度互动（42 个字段）
                ('item_click_stay_engagement', [
                    '206157', '231333', '3029611', '28060', '200107', '200585', '201717', '201910',
                    '201911', '20500', '20501', '20504', '20505', '21242', '231494', '200315',
                    '200316', '25751', '25752', '25754', '302342', '302374', '33203304', '33203306',
                    '33203607', '200317', '33866919', '33866925', '4418001', '7007714', '7704561', '770473',
                    '770570', '770571', '860066', '863133', '33868943', '33868950', '340761', '865341',
                    '865342', '865711',
                ]),
                # 短期候选曝光点击漏斗（41 个字段）
                ('item_shortterm_candidate_funnel', [
                    '12111', '12104', '12110', '12112', '12117', '12101', '12113', '12115',
                    '12155', '10002', '10007', '10008', '12088', '12092', '12094', '25060',
                    '770626', '863009', '863047', '18010', '215350', '215399', '340070', '770583',
                    '770627', '770630', '820027', '820028', '820029', '820035', '863087', '863286',
                    '868500', '868513', '17135', '17136', '17137', '17139', '868486', '241215027',
                    '241215127',
                ]),
                # 当前会话、页面与位置上下文（41 个字段）
                ('item_session_page_position_context', [
                    '24082404', '881284', '341888', '600249', '600254', '863062', '881220', '881221',
                    '881265', '881267', '881309', '881353', '881681', '881705', '881717', '21110',
                    '21114', '21115', '12134', '12137', '12138', '12140', '160063', '340096',
                    '881634', '206051', '340317', '340321', '340335', '3403491', '770472', '882326',
                    '882353', '882354', '882385', '882416', '882417', '882419', '882369', '882371',
                    '206206',
                ]),
                # i2i、图关系与邻居召回（41 个字段）
                ('item_i2i_graph_neighbor_recall', [
                    '17053', '17062', '17086', '17107', '794005', '794021', '794169', '870402',
                    '17027', '17177', '18197', '200754', '201825', '201856', '201912', '201918',
                    '24082423', '241125018', '247030061', '247031681', '302987', '304383', '3044501', '304456',
                    '310588', '340296', '4061', '620000', '770588', '770590', '770591', '770592',
                    '860076', '860090', '881025', '900017', '900647', '909043', '865618', '14237',
                    '341104',
                ]),
                # u2i、q2i 与 Query 触发召回（41 个字段）
                ('item_u2i_q2i_query_recall', [
                    '861124', '861201', '861540', '870357', '27533', '861213', '870373', '87560205',
                    '861504', '863780', '863802', '863808', '863811', '870340', '17058', '17071',
                    '17111', '17178', '27525', '861060', '861612', '862388', '862391', '870128',
                    '870195', '870250', '87580093', '17088', '18088', '200406', '200756', '200757',
                    '280501', '280502', '280602', '280611', '7704581', '770467', '909116', '340827',
                    '820007',
                ]),
                # 召回源、命中、排序与路径（41 个字段）
                ('item_recall_source_hit_rank_path', [
                    '340859', '794202', '794203', '794204', '794205', '794206', '794207', '794211',
                    '867638', '867689', '160033', '200768', '340856', '770587', '794007', '794022',
                    '794023', '794170', '794171', '864578', '864738', '865420', '870283', '880448',
                    '881331', '882401', '131052', '160043', '160044', '160049', '18004', '18007',
                    '18035', '200210', '200269', '200283', '200284', '310585', '310586', '87560220',
                    '865726',
                ]),
            ],
            'creative': [
                # 创意图片、展示形态与促销表达（14 个字段）
                ('creative_display_offer', [
                    '780110', '780111', '780112', '780113', '780117', '8001', '8002', '8003',
                    '8007', '8310', '500157', '900137', '8203', '8207',
                ]),
            ],
        }

    def _validate_semantic_feature_groups(self):
        expected_bucket_ids = {
            'common': set(self.fea_conf_obj.common_fea_map.keys()),
            'item': set(self.fea_conf_obj.item_fea_map.keys()),
            'creative': set(self.fea_conf_obj.creative_fea_map.keys()),
        }
        all_seen_ids = set()
        for bucket_name in self._BUCKET_NAMES:
            bucket_groups = self.rm_semantic_feature_groups.get(bucket_name, [])
            if not bucket_groups:
                raise ValueError('semantic feature groups are empty for {}'.format(bucket_name))
            bucket_seen_ids = set()
            group_names = set()
            for group_name, feature_ids in bucket_groups:
                if group_name in group_names:
                    raise ValueError('duplicated semantic group name: {}'.format(group_name))
                group_names.add(group_name)
                if not feature_ids:
                    raise ValueError('semantic group {} is empty'.format(group_name))
                feature_id_set = set(feature_ids)
                if len(feature_id_set) != len(feature_ids):
                    raise ValueError('semantic group {} contains duplicated feature ids'.format(group_name))
                duplicate_ids = bucket_seen_ids.intersection(feature_id_set)
                if duplicate_ids:
                    raise ValueError('features assigned to multiple semantic groups: {}'.format(sorted(duplicate_ids)))
                bucket_seen_ids.update(feature_id_set)
            missing_ids = expected_bucket_ids[bucket_name] - bucket_seen_ids
            unknown_ids = bucket_seen_ids - expected_bucket_ids[bucket_name]
            if missing_ids or unknown_ids:
                raise ValueError(
                    'semantic mapping mismatch for {}: missing={}, unknown={}'.format(
                        bucket_name, sorted(missing_ids), sorted(unknown_ids)
                    )
                )
            cross_bucket_ids = all_seen_ids.intersection(bucket_seen_ids)
            if cross_bucket_ids:
                raise ValueError('semantic features cross buckets: {}'.format(sorted(cross_bucket_ids)))
            all_seen_ids.update(bucket_seen_ids)

            actual_group_sizes = tuple(len(feature_ids) for _, feature_ids in bucket_groups)
            expected_group_sizes = self._GROUP_SIZES[bucket_name]
            if actual_group_sizes != expected_group_sizes:
                raise ValueError(
                    'RankMixer v6-E2-Small-2 group sizes mismatch for {}: actual={}, expected={}'.format(
                        bucket_name,
                        actual_group_sizes,
                        expected_group_sizes,
                    )
                )
            ordered_ids = []
            for _, feature_ids in bucket_groups:
                ordered_ids.extend(feature_ids)
            checksum = hashlib.sha256('|'.join(ordered_ids).encode('utf-8')).hexdigest()
            expected_checksum = self._GROUP_CHECKSUMS[bucket_name]
            if checksum != expected_checksum:
                raise ValueError(
                    'RankMixer v6-E2-Small-2 group checksum mismatch for {}: actual={}, expected={}'.format(
                        bucket_name,
                        checksum,
                        expected_checksum,
                    )
                )
            logging.info(
                'RankMixer v6-E2-Small-2 frozen groups %s: version=%s, checksum=%s, groups=%s',
                bucket_name,
                self._GROUP_VERSION,
                checksum,
                [(name, len(ids)) for name, ids in bucket_groups],
            )

    def _calculate_dense_trainable_params(self):
        """Return the exact fixed-architecture dense trainable parameter count."""
        common_fields, item_fields, creative_fields = self._EXPECTED_FIELD_COUNTS
        total_fields = common_fields + item_fields + creative_fields
        input_dim = total_fields * self.embedding_size

        input_bn_params = 2 * input_dim
        senet_params = (
            common_fields * self.senet_hidden_size
            + self.senet_hidden_size * common_fields
            + (common_fields + item_fields) * self.senet_hidden_size
            + self.senet_hidden_size * item_fields
            + total_fields * self.senet_hidden_size
            + self.senet_hidden_size * creative_fields
            + 3 * 2 * self.senet_hidden_size
        )

        local_norm_params = self.rm_local_token_num * self.rm_hidden_dim
        global_norm_params = self.rm_hidden_dim
        one_block_norm_params = 2 * self.rm_token_num * self.rm_hidden_dim
        # mature_v1 final LayerNorm shares gamma/beta across all token positions.
        final_norm_params = 2 * self.rm_hidden_dim

        # Projection kernels/biases are token-specific. Local/Block RMSNorm uses
        # per-token gamma while Global RMSNorm uses shared gamma.
        local_token_params = (
            input_dim * self.rm_hidden_dim
            + self.rm_local_token_num * self.rm_hidden_dim
            + local_norm_params
        )
        global_token_params = (
            input_dim * self.rm_hidden_dim
            + self.rm_hidden_dim
            + self.rm_hidden_dim * self.rm_hidden_dim
            + self.rm_hidden_dim
            + global_norm_params
        )

        one_swiglu_params = self.rm_token_num * (
            self.rm_hidden_dim * self.rm_swiglu_hidden_dim
            + self.rm_swiglu_hidden_dim
            + self.rm_hidden_dim * self.rm_swiglu_hidden_dim
            + self.rm_swiglu_hidden_dim
            + self.rm_swiglu_hidden_dim * self.rm_hidden_dim
            + self.rm_hidden_dim
        )
        block_params = self.rm_layer_num * (
            one_block_norm_params
            + 2 * one_swiglu_params
        )

        task_head_params = 0
        task_head_input_dim = self.rm_hidden_dim
        for layer_size in self.cvr_layers:
            task_head_params += task_head_input_dim * layer_size + layer_size
            task_head_params += 2 * layer_size
            task_head_input_dim = layer_size
        task_head_params += task_head_input_dim + 1

        return sum([
            input_bn_params,
            senet_params,
            local_token_params,
            global_token_params,
            block_params,
            final_norm_params,
            task_head_params,
        ])

    def _verify_graph_dense_trainable_params(self, dense_scope):
        """Verify the actual partitioned graph variables before training starts."""
        dense_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=dense_scope,
        )
        log_manifest = not getattr(self, '_rm_logged_dense_manifest', False)
        actual_total = 0
        for variable in dense_variables:
            shape = variable.shape.as_list()
            if any(dimension is None for dimension in shape):
                raise ValueError(
                    'RankMixer v6-E2-Small-2 dense variable has unknown shape: {} {}'.format(
                        variable.op.name,
                        shape,
                    )
                )
            variable_total = 1
            for dimension in shape:
                variable_total *= dimension
            actual_total += variable_total
            if log_manifest:
                logging.info(
                    'RankMixer v6-E2-Small-2 dense variable: name=%s shape=%s params=%d',
                    variable.op.name,
                    shape,
                    variable_total,
                )

        if actual_total != self.rm_expected_dense_trainable_params:
            raise ValueError(
                'RankMixer v6-E2-Small-2 graph dense trainable parameters={}, expected={}'.format(
                    actual_total,
                    self.rm_expected_dense_trainable_params,
                )
            )
        self._rm_logged_dense_manifest = True
        return actual_total

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @classmethod
    def get_features_conf(cls, **kwargs):
        features_conf = {}

        feature_version = kwargs.get('feature_version', None)
        module = locate(feature_version)
        fea_conf_obj = module.FeatureConfig()

        embedding_size = kwargs.get('embedding_size', 17)

        for key, v_map in fea_conf_obj.feature_details.items():
            if bool(int(v_map.get("model_ignore", 0))):
                logging.info(f"fea key {key} will not save")
                continue
            if v_map.get("fea_class", "common") in ["dense", "label", "extra"]:
                logging.info(f"skip fea key {key}")
                continue
            conf = {
                "embedding_size": int(v_map.get("embedding_size", embedding_size)),
                "pooling_type": v_map.get("pooling_type", "SUM_POOLING"),
                "feature_parameter_args": {
                    "accessor": {
                        "stats_param": {
                            "constant_feature": bool(int(v_map.get("constant_feature", 0)))
                        }
                    }
                }
            }
            stats_param = conf["feature_parameter_args"]["accessor"]["stats_param"]

            if "delete_threshold" in v_map:
                delete_threshold = v_map["delete_threshold"]
                stats_param["delete_threshold"] = delete_threshold
                logging.info(f"Feature '{key}': delete_threshold set to {delete_threshold}.")

            if "create_nonclk_prob" in v_map:
                create_nonclk_prob = v_map["create_nonclk_prob"]
                stats_param["create_nonclk_prob"] = create_nonclk_prob
                logging.info(f"Feature '{key}': create_nonclk_prob set to {create_nonclk_prob}.")

            if "create_click_prob" in v_map:
                create_click_prob = v_map["create_click_prob"]
                stats_param["create_nonclk_prob"] = create_click_prob
                logging.info(f"Feature '{key}': create_click_prob set to {create_click_prob}.")

            features_conf[key] = conf
        logging.info(f"features_conf is {features_conf}, features_conf size is {len(features_conf)}")
        return features_conf

    @classmethod
    def get_share_embedding_conf(cls, **kwargs):
        feature_version = kwargs.get('feature_version', None)
        if feature_version:
            module = locate(feature_version)
            fea_conf_obj = module.FeatureConfig()
            return fea_conf_obj.features_share_map
        else:
            return {}

    def get_dataset(self, data_paths, mode, use_dynamic_file=True, take_batch_num=0):
        """获取数据集"""
        parquet_cols = self.features.parquet_reader_columns
        features_spec = tf.feature_column.make_parse_example_spec(parquet_cols)
        size_limits_map = self.fea_conf_obj.feature_size_limit_map
        feature_name_map = self.fea_conf_obj.features_multi_map
        visible_feature_lst = self.fea_conf_obj.visible_fea_map.keys()

        return {
            'dataset': flood_data_util.get_parquet_data(
                features=features_spec,
                data_paths=data_paths,
                batch_size=self.batch_size if mode == "train" else self.eval_batch_size,
                size_limits_map=size_limits_map,
                feature_name_map=feature_name_map,
                sparse_features_to_tensor=list(visible_feature_lst),
                sampler_label_name=self.sampler_label_name,
                sampler_positive_rate=self.sampler_positive_rate,
                sampler_negative_rate=self.sampler_negative_rate,
                filter_pass_empty=self.filter_pass_empty,
                shuffle=True if mode == "train" else False,
                use_dynamic_files=use_dynamic_file if mode != "predict" else False,
                take_batch_num=0 if mode == "train" else take_batch_num,
                random_feature="" if mode == "train" else self.random_feature,
                join_key_name='pk',
                epochs=1,
                prefetch_num=self.prefetch_num,
                sampler_stat=self.sampler_stat,
                drop_last_files=self.drop_last_files if mode == 'train' else 0,
                async_pull=self.async_pull,
                max_prefetched_pull=-1,
                drop_remainder=True if mode == 'train' else False,
                interleave=self.test_interleave if mode in ["test", "predict"] else self.interleave,
                slow_worker_timeout=self.slow_worker_timeout,
                slow_worker_num_limit=self.slow_worker_num_limit,
                range_size_limit=100 * 1024 * 1024,
                hole_size_limit=10 * 1024 * 1024
            )
        }

    def build(self, input_paths, test_paths, mode='train', config=None, use_dynamic=True, **kwargs):
        """构建完整的模型计算图"""
        self.global_step = tf.train.get_or_create_global_step()
        self.global_step_op = tf.assign_add(self.global_step, 1)
        for tmp_mode in ['train', 'test']:
            logging.info(f"{'*' * 10} {tmp_mode} {'*' * 10}")
            data_paths = test_paths if tmp_mode == 'test' else input_paths
            self.build_dataset_op(data_paths, mode=tmp_mode, flood_mode=mode)
            self.build_pred_results_op(mode=tmp_mode, flood_mode=mode)
            self.build_auc_copc_op(mode=tmp_mode)
            if tmp_mode == 'train':
                self.build_loss_op(mode=tmp_mode)
                self.build_summary(mode=tmp_mode)
                self.build_optimizer_op()
        self._build_export(config=config)
        self.run_metadata = tf.RunMetadata()
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, timeout_in_ms=self.timeout)
        self.timeout_options = tf.RunOptions(timeout_in_ms=self.timeout)

        if self.log_nn_vars:
            global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            logging.info('global_vars:')
            for var in global_vars:
                logging.info('{}'.format(var))

    def build_dataset_op(self, data_paths, mode, flood_mode):
        if mode == 'train':
            use_dynamic_files = (flood_mode == 'train')
        else:
            use_dynamic_files = self.strict_test_date and self.order_by_date

        logging.info(
            f"flood_mode is {flood_mode}, {mode}_paths: {data_paths[:2]}, use_dynamic_files is {use_dynamic_files}")

        dataset_op = self.get_dataset(
            data_paths,
            flood_mode,
            use_dynamic_file=use_dynamic_files,
            take_batch_num=self.test_batch_num if mode == 'test' else 0
        )

        dataset = dataset_op['dataset'].map(self.parse_examples, num_parallel_calls=None)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_initializable_iterator()

        self[f'{mode}_iterator'] = iterator
        self[f'{mode}_init_op'] = iterator.initializer

        res = self[f'{mode}_iterator'].get_next()
        for key, value in res.items():
            self[f'{mode}_{key}'] = value

    def parse_examples(self, *example_batch):
        """解析输入数据批次，只保留 fst 主任务 label"""
        columns = self.features.parquet_reader_columns
        features = parsing_ops.parse_parquet(
            example_batch,
            tf.feature_column.make_parse_example_spec(columns),
            reserved_keys=self.fea_conf_obj.visible_fea_map,
            unique=False,
            share_embedding_conf=self.fea_conf_obj.features_share_map,
            global_hash=False,
            psv2=True
        )
        features["sampleid"] = flood.generate_sample_id(
            search_ids=features["search_id"].values,
            example_ids=features["example_ids"].values)
        label_cvr_first = tf.cast(features.pop('fst_cvr_label'), tf.float32)
        sampleid = tf.cast(features.pop('sampleid'), tf.float32)
        search_id = features["search_id"].values
        example_id = features["example_ids"].values

        return {
            'features': features,
            'labels': label_cvr_first,
            'sampleid': sampleid,
            'search_id': search_id,
            'example_id': example_id
        }

    def build_pred_results_op(self, mode, flood_mode=None):
        fn_mode = mode if mode == 'test' else flood_mode
        results = self.model_fn(self[f'{mode}_features'], self[f'{mode}_labels'], mode=fn_mode)

        for key, value in results.items():
            self[f'{mode}_{key}'] = value

    def build_loss_op(self, mode):
        """只保留 fst 主任务损失"""
        labels = tf.reshape(self[f'{mode}_labels'], shape=[-1])
        pred = tf.reshape(self[f'{mode}_pred'], shape=[-1])
        self.loss = tf.reduce_mean(tf.losses.log_loss(predictions=pred, labels=labels))
        self.labels_pos_cvr_count = tf.reduce_sum(labels)

    def build_auc_copc_op(self, mode):
        """只保留 cvr 主指标"""
        self[f'{mode}_auc'] = flood_auc(self[f'{mode}_labels'], self[f'{mode}_pred'], name='auc/cvr',
                                        num_thresholds=2000)
        self[f'{mode}_copc'] = tf.reduce_sum(self[f'{mode}_pred']) / (tf.reduce_sum(self[f'{mode}_labels']) + 1e-8)
        self[f'{mode}_auc_values'] = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES, scope='auc')
        self[f'{mode}_reset_auc_op'] = tf.variables_initializer(var_list=self[f'{mode}_auc_values'])
        self[f'{mode}_pred_mean'] = tf.reduce_mean(self[f'{mode}_pred'])

    def build_summary(self, mode):
        auc_summary = tf.summary.scalar(f'{mode}/auc', self[f'{mode}_auc'])
        loss_summary = tf.summary.scalar(f'{mode}/loss', self.loss)
        copc_summary = tf.summary.scalar(f'{mode}/copc', self[f'{mode}_copc'])
        self.eval_summary = tf.summary.merge(
            [
                loss_summary,
                auc_summary,
                copc_summary,
            ],
            name='eval_summary',
        )

    def build_optimizer_op(self):
        """构建优化器操作，包括梯度计算和应用"""
        if "circle_restart" in self.decay:
            self.learning_rate = tf.train.cosine_decay_restarts(
                learning_rate=self.learning_rate,
                global_step=tf.train.get_global_step(),
                first_decay_steps=800000,
                t_mul=2.0,
                m_mul=1.0,
                alpha=0.000005
            )
        elif "exp" in self.decay:
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.learning_rate,
                global_step=tf.train.get_global_step(),
                decay_steps=500000,
                decay_rate=0.98,
                staircase=False,
                name=None
            )
        else:
            self._build_lr_schedule()

        optimizer = self.get_optimizer(self.optimizer, self.learning_rate)
        self.optimizer = FloodOptimizer(optimizer)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        for (grad, var) in grads_and_vars:
            logging.info(f'[normal gradiant] {grad} {var}')
            if grad is not None:
                tf.summary.histogram('train/' + var.op.name + '/gradients', grad)
        self.train_op = [self.optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())]

    def _build_lr_schedule(self):
        learning_rate = self.learning_rate
        learning_rate = self._schedule_lr(learning_rate, self.schedule_config)
        self.learning_rate = learning_rate

    def _schedule_lr(self, lr, schedule_config: dict):
        lr = tf.convert_to_tensor(lr)
        if 'type' in schedule_config:
            logging.info('use lr decay schedule')
            learning_rate_utils.get_or_create_milestone_step_reset_op()
            schedule_type = schedule_config['type']
            lr = learning_rate_utils.learning_rate_schedule(
                lr,
                schedule_type,
                **schedule_config)
        return lr

    def get_optimizer(self, optimizer='Adagrad', learning_rate=0.001):
        optimizer = optimizer.strip()
        logging.info('use optimitzer: ' + optimizer)
        if optimizer == 'Adam':
            return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif optimizer == "flood_adam":
            from flood.python.training.adam_optimizer import AdamOptimizer as FloodAdamOptimizer
            optimizer = FloodAdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1e-8)
            return optimizer
        elif optimizer == 'Adagrad':
            return tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
        elif optimizer == 'Momentum':
            return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
        elif optimizer == 'ftrl':
            return tf.train.FtrlOptimizer(learning_rate)
        elif optimizer == 'lazyAdam':
            return tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif optimizer == 'SGD':
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        logging.info('cannot find optimizer: ' + optimizer)
        return self.optimizer

    def train(self, session, worker_id=0, **kwargs):
        """执行训练步骤"""
        self.train_count += 1
        fetch = {
            'train_op': self.train_op,
            'loss': self.loss,
            'labels_pos_cvr_count': self.labels_pos_cvr_count,
            'global_step': self.global_step,
            'pred_mean': self['train_pred_mean'],
            'auc': self['train_auc'],
            'copc': self['train_copc'],
            'learning_rate': self.learning_rate,
        }

        res = session.run(fetch, options=self.timeout_options)

        if self.train_count % kwargs.get('train_log_step', 10) == 0:
            logging.info(f"----------------- train [{self.train_count}] ------------------------")
            logging.info(
                f"lstep: {self.train_count}, "
                f"gstep: {res['global_step']}, "
                f"loss: {res['loss']:.6f}, "
                f"auc: {res['auc']:.6f}, "
                f"copc: {res['copc']:.6f}, "
                f"pred_mean: {res['pred_mean']:.6f},"
                f"labels_pos_cvr_count: {res['labels_pos_cvr_count']},"
                f"learning_rate:  {res['learning_rate']},"
            )
            logging.info("-------------------------------------------------------------")

        if self.task_index == 0 and self.train_reset_interval > 0 \
                and self.train_count * self.num_worker > self.train_reset_interval * self.train_reset_count:
            self.train_reset_count += 1
            logging.info(" >>>> reset auc <<<< ")
            session.run([self['train_reset_auc_op']])
        return {'global_step': res['global_step'], 'train_reset_count': self.train_reset_count}

    def test(self, session, worker_id=0, prefix='test', **kwargs):
        self.train_init(session)
        FORMAT = '%(asctime)-15s [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s'
        file_handler = FileHandler('flood_worker_0.log')
        file_handler.setFormatter(Formatter(FORMAT))

        logger = getLogger(name='search_jarvis_logging')
        logger.addHandler(file_handler)

        test_cnt = 0
        session.run([self['test_init_op']])

        auc_accum = RocAucAccum(num_thresholds=2000)
        pr_auc_accum = PrAucAccum(num_thresholds=2000)
        copc_accum = COPCAccum()
        bucket_error = BucketErrorAccum(1000)
        sample_cnt_accum = SampleCntAccum()

        fetchs = {
            'sampleid': self['test_sampleid'],
            'test_search_id': self['test_search_id'],
            'test_example_id': self['test_example_id'],
            'labels': self['test_labels'],
            'pred': self['test_pred'],
            'auc': self['test_auc'],
            'copc': self['test_copc'],
        }

        if self.save_predict_result:
            local_path = 'predictions-{}.txt'.format(worker_id)
            if self.predict_path:
                hdfs_dir = os.path.join(self.predict_path, prefix)
            else:
                hdfs_dir = os.path.join(self.model_dir, prefix)
            hdfs_path = os.path.join(hdfs_dir, local_path)
            logging.info("predict res local path: %s", local_path)
            logging.info("predict res hdfs path: %s", hdfs_path)
            if worker_id == 0:
                mkdir_hdfs(hdfs_dir)
            cnt = 0
            with tf.gfile.Open(local_path, 'w') as f:
                f.write('')

        while True:
            try:
                res = session.run(fetchs, options=self.timeout_options)

                if self.save_predict_result:
                    with tf.gfile.Open(local_path, 'a') as f:
                        for search_id, example_id, label_cvr, pred in zip(res['test_search_id'],
                                                                          res['test_example_id'], res['labels'],
                                                                          res['pred']):
                            line = '\t'.join(
                                [search_id.decode(), example_id.decode(), str(label_cvr[0]), str(pred)]) + '\n'
                            f.write(line)
                            cnt += 1

                label_cvr, pred = res['labels'], res['pred']
                test_cnt += 1

                auc_accum.update(label_cvr, pred)
                pr_auc_accum.update(label_cvr, pred)
                copc_accum.update(label_cvr, pred)
                bucket_error.update(label_cvr, pred)
                sample_cnt_accum.update(label_cvr, pred)

                if 0 < self.test_batch_num < test_cnt:
                    logging.info(f"finish test by test_batch_num={self.test_batch_num}")
                    break

                if test_cnt % kwargs.get('test_log_step', 10) == 0:
                    logging.info("----------------- test_cnt [%s] ------------------------" % test_cnt)
                    logging.info(f"CVR AUC: {res['auc']:.6f}  CVR COPC: {res['copc']:.6f}")

            except tf.errors.OutOfRangeError as e:
                logging.info(f'all data set used. {e.message}')
                break
            except tf.errors.DeadlineExceededError as e:
                logging.error('===========test step timed out========== %s' % e.message)
                break
            except tf.errors.InvalidArgumentError as e:
                logging.warning('data error: %s' % e.message)
                continue
            except tf.errors.PermissionDeniedError as e:
                logging.error("PermissionDeniedError: %s" % str(e))
                break
            except tf.errors.FailedPreconditionError as e:
                logging.error("FailedPreconditionError: %s" % str(e))
                break
            except RuntimeError as e:
                logging.warning("runtime error:%s" % str(e))
                break

        accum_metrics = {'cvr-tower': {
            'roc_auc': auc_accum.dump(),
            'copc': copc_accum.dump(),
            'pr_auc': pr_auc_accum.dump(),
            'bucket_error': bucket_error.dump(),
            'sample_cnt': sample_cnt_accum.dump(),
        }}

        res = {'accum_metrics': accum_metrics,
               'title': f'lamb-feature-{self.random_feature}' if self.random_feature else 'base'}

        if self.save_predict_result:
            upload_hdfs(local_path, hdfs_path, True)
            logging.info("upload predict result into hdfs: %s", hdfs_path)

        if self.upload_log and self.save_predict_result and worker_id == 0:
            logging.info("set worker0 log file")
            log_hdfs_path = os.path.join(hdfs_dir, "flood_worker_0.log")
            upload_hdfs("flood_worker_0.log", log_hdfs_path, True)
            logging.info("worker0 log upload done")

        return res

    def predict(self, session, worker_id=0, **kwargs):
        prefix = 'predict'
        if self.random_feature:
            prefix = 'predict-%s' % self.random_feature

        ret = self.test(session, worker_id, prefix=prefix, **kwargs)

        if self.random_feature:
            logging.info("Run all predict data for Random Feature: %s" % self.random_feature)
        else:
            logging.info("Run all predict data.")

        if self.random_feature:
            if self.parallel_feature_analysis:
                ret.update({'merge_from_all_workers': False})
            else:
                ret.update({'merge_from_all_workers': True})

        return ret

    def _build_export(self, config=None):
        serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='example')
        features = tf.parse_example(serialized_tf_example,
                                    tf.feature_column.make_parse_example_spec(self.features.export_columns))

        fake_labels = tf.constant(value=[[1]], shape=[1, 1], dtype=tf.float32)
        pred_result = self.model_fn(features, fake_labels, mode="export", export=True)

        self.export_spec = {
            'input': {'example': serialized_tf_example},
            'output': {'cvr': pred_result['pred']}
        }

    def export(self):
        return self.export_spec

    def train_init(self, session):
        logging.info("reinitialize train_init_op.")
        session.run(self['train_init_op'])
        if self.is_chief:
            session.run(learning_rate_utils.get_or_create_milestone_step_reset_op())
            logging.info(
                "milestone step: %s",
                session.run(learning_rate_utils.get_or_create_milestone_step()),
            )

    def evaluate(self, session, **kwargs):
        self.eval_count += 1
        fetches = {
            'summary': self['eval_summary'],
            'global_step': self.global_step,
        }
        result = None
        try:
            timeout = 400000
            result = session.run(fetches, options=tf.RunOptions(timeout_in_ms=timeout))
        except tf.errors.DeadlineExceededError:
            logging.error('Error: evaluation timed out')
            return
        except tf.errors.OutOfRangeError:
            logging.info("Run out of evaluation data, reinitialize")
            self.train_init(session)

        result['summary'] = tf.Summary()
        return result

    def _post_process_sequence(self, features, feature_embed_map, mode="train"):
        """序列特征后处理函数"""
        sequence_embs_map = {}  # 序列嵌入映射
        sequence_mask_map = {}  # 填充掩码映射

        for key, v_map in self.fea_conf_obj.seq_fea_map.items():
            sparse_input = features[key]
            if v_map.get('padding_fea', False):
                sp_emb = feature_embed_map[key]
                max_len = int(v_map.get("max_len", self.default_sequence_len))
                dim = int(v_map.get("embedding_size", self.embedding_size))
                logging.info(f"key is {key}, v_map is {v_map}")

                if mode != 'export':
                    indices = sparse_input.indices
                    bz = sparse_input.dense_shape[0]
                    trunc_mask = tf.greater_equal(tf.constant(max_len - 1, dtype=tf.int64), indices[:, 1])
                    indices = tf.boolean_mask(indices, trunc_mask)
                    emb = tf.boolean_mask(sp_emb, trunc_mask)
                    emb = tf.scatter_nd(indices, emb, shape=[bz, max_len, dim])

                    ones = tf.ones(shape=[tf.shape(indices)[0]])
                    mask = tf.scatter_nd(indices, ones, shape=[bz, max_len])
                    sequence_mask_map[key] = tf.greater(mask, 0)
                else:
                    seq_len_feature = features[key + "009"]
                    mask = tf.greater(seq_len_feature, tf.range(0, max_len, dtype=tf.float32))
                    indices = tf.where(mask)
                    out_shape = tf.concat((tf.shape(mask, out_type=tf.int64), [dim]), axis=0)
                    emb = tf.scatter_nd(indices, sp_emb, out_shape)
                    sequence_mask_map[key] = mask

                sequence_embs_map[key] = emb

        return sequence_embs_map, sequence_mask_map

    def list_all_member(self):
        logging.info('-' * 30)
        logging.info('model args:')
        for name, value in vars(self).items():
            logging.info('%s=%s' % (name, value))
        logging.info('-' * 30)

    def get_hooks(self):
        hooks = []
        if self.enable_dense_warmup and (
                self.tf_config['task']['type'] == "master" or self.tf_config['task']['index'] == 0):
            hooks.append(Senet2NewWarmupHook(self.model_dir, model=self))
        return hooks

    # ============================ RankMixer 主塔 ============================
    @staticmethod
    def get_init(fan_in):
        # """RankMixer 投影/FFN 权重初始化：1/sqrt(fan_in) 正态。
        # 与 dcnm_fst 及本文件 rm_head（输出头）的初始化风格一致。"""
        return tf.random_normal_initializer(stddev=1.0 / math.sqrt(fan_in))

    def senet_layer(self, common_embedding, item_embedding, creative_embedding, is_train, export):
        """Field-wise hierarchical SENet gate retained from the strong base model."""
        common_field_num = len(self.fea_conf_obj.common_fea_map)
        item_field_num = len(self.fea_conf_obj.item_fea_map)
        creative_field_num = len(self.fea_conf_obj.creative_fea_map)

        with tf.variable_scope("senet", reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            weight_common_in = tf.get_variable(
                shape=[common_field_num, self.senet_hidden_size],
                initializer=tf.glorot_uniform_initializer(),
                name="common_weight_in",
            )
            weight_common_out = tf.get_variable(
                shape=[self.senet_hidden_size, common_field_num],
                initializer=tf.glorot_uniform_initializer(),
                name="common_weight_out",
            )
            weight_item_common_in = tf.get_variable(
                shape=[item_field_num + common_field_num, self.senet_hidden_size],
                initializer=tf.glorot_uniform_initializer(),
                name="common_item_weight_in",
            )
            weight_item_out = tf.get_variable(
                shape=[self.senet_hidden_size, item_field_num],
                initializer=tf.glorot_uniform_initializer(),
                name="item_weight_out",
            )
            weight_all_in = tf.get_variable(
                shape=[common_field_num + item_field_num + creative_field_num, self.senet_hidden_size],
                initializer=tf.glorot_uniform_initializer(),
                name="common_item_creative_weight_in",
            )
            weight_creative_out = tf.get_variable(
                shape=[self.senet_hidden_size, creative_field_num],
                initializer=tf.glorot_uniform_initializer(),
                name="creative_weight_out",
            )

            common_3d = tf.reshape(common_embedding, [-1, common_field_num, self.embedding_size])
            common_mean = tf.reduce_mean(common_3d, axis=-1)
            common_hidden = tf.matmul(common_mean, weight_common_in)
            if self.use_senet_bn:
                common_hidden = self.batch_norm_layer_v2(
                    x=common_hidden,
                    train_phase=is_train,
                    scope_bn="bn_input_common",
                    batch_norm_decay=self.batch_norm_decay,
                    use_riemann_bn=self.use_riemann_bn,
                    export=export,
                )
            common_gate = 2.0 * tf.nn.sigmoid(tf.matmul(tf.nn.tanh(common_hidden), weight_common_out))
            common_out = tf.reshape(
                common_3d * tf.expand_dims(common_gate, axis=2),
                [-1, common_embedding.shape[-1].value],
            )

            item_3d = tf.reshape(item_embedding, [-1, item_field_num, self.embedding_size])
            item_mean = tf.reduce_mean(item_3d, axis=-1)
            item_hidden = tf.matmul(tf.concat([common_mean, item_mean], axis=-1), weight_item_common_in)
            if self.use_senet_bn:
                item_hidden = self.batch_norm_layer_v2(
                    x=item_hidden,
                    train_phase=is_train,
                    scope_bn="bn_input_item",
                    batch_norm_decay=self.batch_norm_decay,
                    use_riemann_bn=self.use_riemann_bn,
                    export=export,
                )
            item_gate = 2.0 * tf.nn.sigmoid(tf.matmul(tf.nn.tanh(item_hidden), weight_item_out))
            item_out = tf.reshape(
                item_3d * tf.expand_dims(item_gate, axis=2),
                [-1, item_embedding.shape[-1].value],
            )

            creative_3d = tf.reshape(creative_embedding, [-1, creative_field_num, self.embedding_size])
            creative_mean = tf.reduce_mean(creative_3d, axis=-1)
            creative_hidden = tf.matmul(
                tf.concat([common_mean, item_mean, creative_mean], axis=-1),
                weight_all_in,
            )
            if self.use_senet_bn:
                creative_hidden = self.batch_norm_layer_v2(
                    x=creative_hidden,
                    train_phase=is_train,
                    scope_bn="bn_input_creative",
                    batch_norm_decay=self.batch_norm_decay,
                    use_riemann_bn=self.use_riemann_bn,
                    export=export,
                )
            creative_gate = 2.0 * tf.nn.sigmoid(
                tf.matmul(tf.nn.tanh(creative_hidden), weight_creative_out)
            )
            creative_out = tf.reshape(
                creative_3d * tf.expand_dims(creative_gate, axis=2),
                [-1, creative_embedding.shape[-1].value],
            )

        return common_out, item_out, creative_out

    def _rm_rms_norm(self, inputs, scope, per_token):
        """v6 RMSNorm: per-token gamma [T,D] or shared gamma [D]."""
        shape = inputs.get_shape().as_list()
        if len(shape) not in (2, 3):
            raise ValueError('RMSNorm expects rank-2 or rank-3 input, got {}'.format(shape))
        hidden_dim = shape[-1]
        if hidden_dim is None:
            raise ValueError('RMSNorm hidden dimension must be statically known')

        if per_token:
            if len(shape) != 3 or shape[1] is None:
                raise ValueError(
                    'per-token RMSNorm requires static [B,T,D], got {}'.format(shape)
                )
            gamma_shape = [shape[1], hidden_dim]
        else:
            gamma_shape = [hidden_dim]

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            gamma = tf.get_variable(
                'gamma',
                shape=gamma_shape,
                initializer=tf.ones_initializer(),
            )
        variance = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        return inputs * tf.rsqrt(variance + self.rm_rms_epsilon) * gamma

    def _rm_norm(self, inputs, scope, export, per_token):
        """Fixed v6 RMSNorm path; export is irrelevant for this pure TF op."""
        del export
        return self._rm_rms_norm(inputs, scope=scope, per_token=per_token)

    def _project_token_family(self, token_inputs, input_dim, family_name):
        """Project same-width local groups with one batched GEMM and independent weights."""
        if not token_inputs:
            raise ValueError('token projection family cannot be empty')
        for token_input in token_inputs:
            if token_input.shape[-1].value != input_dim:
                raise ValueError(
                    'projection family {} input dimension mismatch: {} vs {}'.format(
                        family_name,
                        token_input.shape[-1].value,
                        input_dim,
                    )
                )

        family_inputs = tf.stack(token_inputs, axis=1)          # [B,N,I]
        token_major = tf.transpose(family_inputs, [1, 0, 2])    # [N,B,I]
        family_size = len(token_inputs)
        with tf.variable_scope(
            family_name,
            reuse=tf.AUTO_REUSE,
            partitioner=self.partitioner,
        ):
            weight = tf.get_variable(
                'weight',
                shape=[family_size, input_dim, self.rm_hidden_dim],
                initializer=self.get_init(input_dim),
            )
            bias = tf.get_variable(
                'bias',
                shape=[family_size, 1, self.rm_hidden_dim],
                initializer=tf.zeros_initializer(),
            )
            projected = tf.matmul(token_major, weight) + bias
            if self.rm_token_proj_act not in (None, 'identity', 'linear'):
                projected = self.get_act_func(self.rm_token_proj_act)(projected)
        return tf.transpose(projected, [1, 0, 2])

    def _semantic_tokenize(self, bucket_field_maps, export):
        """Build 31 frozen balanced local tokens in their canonical hard-coded order."""
        token_specs = []
        for bucket_name in self._BUCKET_NAMES:
            for group_name, feature_ids in self.rm_semantic_feature_groups[bucket_name]:
                group = [bucket_field_maps[bucket_name][feature_id] for feature_id in feature_ids]
                token_input = group[0] if len(group) == 1 else tf.concat(group, axis=-1)
                input_dim = token_input.shape[-1].value
                if input_dim is None:
                    raise ValueError('local token input dimension must be statically known')
                token_specs.append((bucket_name, group_name, feature_ids, token_input, input_dim))

        if len(token_specs) != self.rm_local_token_num:
            raise ValueError(
                'local token count={} must equal rm_local_token_num={}'.format(
                    len(token_specs),
                    self.rm_local_token_num,
                )
            )

        families = {}
        for token_index, spec in enumerate(token_specs):
            families.setdefault(spec[4], []).append((token_index, spec[3]))

        tokens = [None] * self.rm_local_token_num
        with tf.variable_scope(
            'rm_local_tokenize',
            reuse=tf.AUTO_REUSE,
            partitioner=self.partitioner,
        ):
            for input_dim in sorted(families.keys()):
                family = families[input_dim]
                family_output = self._project_token_family(
                    [token_input for _, token_input in family],
                    input_dim,
                    'input_dim_{}'.format(input_dim),
                )
                if self.rm_optimize_tokenize:
                    # One Unpack gives one Pack in backward, instead of N full
                    # StridedSliceGrad tensors followed by their accumulation.
                    # Preserve the projection layout and its arithmetic graph:
                    # changing those can let Grappler reassociate GELU products.
                    family_tokens = tf.unstack(family_output, num=len(family), axis=1)
                    for (token_index, _), token in zip(family, family_tokens):
                        tokens[token_index] = token
                else:
                    for family_index, (token_index, _) in enumerate(family):
                        tokens[token_index] = family_output[:, family_index, :]

            if any(token is None for token in tokens):
                raise ValueError('RankMixer v6-E2-Small-2 failed to project every local token')
            local_tokens = tf.stack(tokens, axis=1)
            local_tokens = self._rm_norm(
                local_tokens,
                scope='local_token_rms_norm',
                export=export,
                per_token=True,
            )

        for token_index, (bucket_name, group_name, feature_ids, _, input_dim) in enumerate(token_specs):
            logging.info(
                'RankMixer v6-E2-Small-2 local token %d: bucket=%s, name=%s, fields=%d, input_dim=%d, D=%d',
                token_index,
                bucket_name,
                group_name,
                len(feature_ids),
                input_dim,
                self.rm_hidden_dim,
            )
        return local_tokens

    def _build_global_token(self, bucket_tensors, export):
        """Encode all post-BN/SENet fields into one global token."""
        global_input = tf.concat(
            [bucket_tensors[name] for name in self._BUCKET_NAMES],
            axis=-1,
        )
        input_dim = global_input.shape[-1].value
        if input_dim is None:
            raise ValueError('global token input dimension must be statically known')

        with tf.variable_scope(
            'rm_global_token',
            reuse=tf.AUTO_REUSE,
            partitioner=self.partitioner,
        ):
            hidden = tf.contrib.layers.fully_connected(
                inputs=global_input,
                num_outputs=self.rm_hidden_dim,
                activation_fn=self.get_act_func(self.rm_token_proj_act),
                weights_initializer=self.get_init(input_dim),
                biases_initializer=tf.zeros_initializer(),
                scope='fc1',
            )
            global_token = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=self.rm_hidden_dim,
                activation_fn=tf.identity,
                weights_initializer=self.get_init(self.rm_hidden_dim),
                biases_initializer=tf.zeros_initializer(),
                scope='fc2',
            )
            global_token = self._rm_norm(
                global_token,
                scope='rms_norm',
                export=export,
                per_token=False,
            )

        logging.info(
            'RankMixer v6-E2-Small-2 global token: input_dim=%d -> %d -> %d',
            input_dim,
            self.rm_hidden_dim,
            self.rm_hidden_dim,
        )
        return global_token

    def _rm_mix_tokens(self, inputs):
        """Parameter-free Mixing: [B,T,D] -> [B,H,T*D/H]."""
        token_num = inputs.shape[1].value
        hidden_dim = inputs.shape[2].value
        if token_num != self.rm_token_num or hidden_dim != self.rm_hidden_dim:
            raise ValueError(
                'Mixing expects [B,{},{}], got {}'.format(
                    self.rm_token_num,
                    self.rm_hidden_dim,
                    inputs.get_shape(),
                )
            )
        head_dim = hidden_dim // self.rm_head_num
        split = tf.reshape(
            inputs,
            [-1, token_num, self.rm_head_num, head_dim],
        )
        transposed = tf.transpose(split, [0, 2, 1, 3])
        mixed = tf.reshape(
            transposed,
            [-1, self.rm_head_num, token_num * head_dim],
        )
        if token_num * head_dim != self.rm_hidden_dim:
            raise ValueError('RankMixer v6-E2-Small-2 requires H=T so mixed dimension remains D')
        return mixed

    def _rm_revert_tokens(self, mixed):
        """Exact inverse of _rm_mix_tokens for the v6 H=T layout."""
        head_num = mixed.shape[1].value
        mixed_dim = mixed.shape[2].value
        head_dim = self.rm_hidden_dim // self.rm_head_num
        expected_mixed_dim = self.rm_token_num * head_dim
        if head_num != self.rm_head_num or mixed_dim != expected_mixed_dim:
            raise ValueError(
                'Reverting expects [B,{},{}], got {}'.format(
                    self.rm_head_num,
                    expected_mixed_dim,
                    mixed.get_shape(),
                )
            )
        split = tf.reshape(
            mixed,
            [-1, self.rm_head_num, self.rm_token_num, head_dim],
        )
        transposed = tf.transpose(split, [0, 2, 1, 3])
        return tf.reshape(
            transposed,
            [-1, self.rm_token_num, self.rm_hidden_dim],
        )

    def _rm_per_token_swiglu(self, inputs, scope):
        """Batched Per-token SwiGLU with independent [D,M,D] parameters per token."""
        token_num = inputs.shape[1].value
        hidden_dim = inputs.shape[2].value
        if token_num != self.rm_token_num or hidden_dim != self.rm_hidden_dim:
            raise ValueError(
                'Per-token SwiGLU expects [B,{},{}], got {}'.format(
                    self.rm_token_num,
                    self.rm_hidden_dim,
                    inputs.get_shape(),
                )
            )

        token_major = tf.transpose(inputs, [1, 0, 2])  # [T,B,D]
        if self.rm_down_init_scale == 0.0:
            down_initializer = tf.zeros_initializer()
        else:
            down_initializer = tf.random_normal_initializer(
                stddev=self.rm_down_init_scale / math.sqrt(self.rm_swiglu_hidden_dim)
            )

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            w_up = tf.get_variable(
                'w_up',
                shape=[token_num, hidden_dim, self.rm_swiglu_hidden_dim],
                initializer=self.get_init(hidden_dim),
            )
            b_up = tf.get_variable(
                'b_up',
                shape=[token_num, 1, self.rm_swiglu_hidden_dim],
                initializer=tf.zeros_initializer(),
            )
            w_gate = tf.get_variable(
                'w_gate',
                shape=[token_num, hidden_dim, self.rm_swiglu_hidden_dim],
                initializer=self.get_init(hidden_dim),
            )
            b_gate = tf.get_variable(
                'b_gate',
                shape=[token_num, 1, self.rm_swiglu_hidden_dim],
                initializer=tf.zeros_initializer(),
            )
            w_down = tf.get_variable(
                'w_down',
                shape=[token_num, self.rm_swiglu_hidden_dim, hidden_dim],
                initializer=down_initializer,
            )
            b_down = tf.get_variable(
                'b_down',
                shape=[token_num, 1, hidden_dim],
                initializer=tf.zeros_initializer(),
            )

            up = tf.matmul(token_major, w_up) + b_up
            gate = tf.matmul(token_major, w_gate) + b_gate
            hidden = up * (gate * tf.nn.sigmoid(gate))
            output = tf.matmul(hidden, w_down) + b_down
        return tf.transpose(output, [1, 0, 2])

    def _rm_block(self, inputs, block_idx, export):
        """PreNorm Mixing/Reverting block with two independent Per-token SwiGLUs."""
        with tf.variable_scope(
            'rm_block_{}'.format(block_idx),
            reuse=tf.AUTO_REUSE,
            partitioner=self.partitioner,
        ):
            mixed = self._rm_mix_tokens(inputs)
            mixed_norm = self._rm_norm(
                mixed,
                scope='mixed_rms_norm',
                export=export,
                per_token=True,
            )
            mixed_update = self._rm_per_token_swiglu(
                mixed_norm,
                scope='mixed_swiglu',
            )
            mixed_hidden = mixed + mixed_update

            reverted = self._rm_revert_tokens(mixed_hidden)
            original_norm = self._rm_norm(
                reverted,
                scope='original_rms_norm',
                export=export,
                per_token=True,
            )
            original_update = self._rm_per_token_swiglu(
                original_norm,
                scope='original_swiglu',
            )

            # The long residual is the block input. Reverted features condition
            # the original-space update without breaking the identity path.
            output = inputs + original_update
        logging.info(
            'RankMixer v6-E2-Small-2 block %d: input=%s, mixed=%s, reverted=%s, output=%s',
            block_idx,
            inputs.get_shape(),
            mixed.get_shape(),
            reverted.get_shape(),
            output.get_shape(),
        )
        return output

    def _rm_stack(self, inputs, export):
        output = inputs
        for block_idx in range(self.rm_layer_num):
            output = self._rm_block(output, block_idx, export)
        return output

    def _rm_final_layer_norm(self, inputs):
        """mature_v1 final LayerNorm: shared [D] gamma/beta over each token."""
        shape = inputs.get_shape().as_list()
        if len(shape) != 3 or shape[1:] != [self.rm_token_num, self.rm_hidden_dim]:
            raise ValueError(
                'RankMixer v6-E2-Small-2 final LayerNorm expects [B,{},{}], got {}'.format(
                    self.rm_token_num,
                    self.rm_hidden_dim,
                    shape,
                )
            )
        with tf.variable_scope(
            'rm_final_layer_norm',
            reuse=tf.AUTO_REUSE,
            partitioner=self.partitioner,
        ):
            gamma = tf.get_variable(
                'gamma',
                shape=[self.rm_hidden_dim],
                initializer=tf.ones_initializer(),
                trainable=True,
            )
            beta = tf.get_variable(
                'beta',
                shape=[self.rm_hidden_dim],
                initializer=tf.zeros_initializer(),
                trainable=True,
            )
            mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
            variance = tf.reduce_mean(
                tf.square(inputs - mean),
                axis=-1,
                keepdims=True,
            )
            normalized = (
                inputs - mean
            ) / tf.sqrt(variance + self.rm_final_ln_epsilon)
            return gamma * normalized + beta

    def _mature_batch_norm(self, inputs, is_train, export, scope):
        """Use the same task-tower BN configuration as mature_rankmixer_v1."""
        if not self.batch_norm:
            return inputs
        return ModelBase.batch_norm_layer_v2(
            x=inputs,
            train_phase=is_train,
            scope_bn=scope,
            batch_norm_decay=self.batch_norm_decay,
            use_riemann_bn=self.use_riemann_bn,
            renorm=self.embed_use_renorm,
            renorm_decay=self.embed_renorm_decay,
            export=export,
        )

    def _task_head(self, context, is_train, export):
        """mature_v1-style [256,128] single-task prediction tower."""
        hidden = context
        for layer_idx, layer_size in enumerate(self.cvr_layers):
            input_dim = hidden.shape[-1].value
            if input_dim is None:
                raise ValueError('task head input dimension must be statically known')
            hidden = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=layer_size,
                activation_fn=None,
                weights_initializer=self.get_init(input_dim),
                biases_initializer=tf.zeros_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_deep),
                scope='rm_v5_mlp{}'.format(layer_idx),
            )
            hidden = self._mature_batch_norm(
                hidden,
                is_train,
                export,
                scope='rm_v5_bn_{}'.format(layer_idx),
            )
            hidden = self.get_act_func(
                self.mlp_act_type,
                is_train=is_train,
                name='rm_v5_mlp_act_{}'.format(layer_idx),
            )(hidden)

        with tf.device('/job:ps/task:0'):
            output = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=1,
                activation_fn=tf.identity,
                weights_initializer=self.get_init(hidden.shape[-1].value),
                biases_initializer=tf.zeros_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_deep),
                scope='rm_v5_out',
            )
        return output

    def model_fn(self, features, labels, timestamps=None, mode="train", export=False):
        del timestamps
        variable_partitions = self.num_ps
        if self.max_partitions is not None:
            variable_partitions = min(variable_partitions, self.max_partitions)
        self.partitioner = tf.min_max_variable_partitioner(
            max_partitions=variable_partitions,
            min_slice_size=1024000,
        )
        is_train = mode == "train"
        ps_mode = "predict" if self.ps_stage == "join" and is_train else mode

        sparse_embeddings = lookup_utils.flood_lookup_psv2(
            features=features,
            non_seq_columns=self.features.lookup_nonseq_columns,
            seq_columns=self.features.seq_columns,
            batch_size=self.batch_size,
            mode=ps_mode,
            clicks=tf.cast(labels, tf.float32),
            no_update_fea_names=list(self.fea_conf_obj.const_fea_map.keys()),
        )

        bucket_embedding_maps = {name: {} for name in self._BUCKET_NAMES}
        for index, column in enumerate(self.features.lookup_nonseq_columns + self.features.seq_columns):
            feature_key = get_sparse_fc_key(column)
            bucket_name = None
            if feature_key in self.fea_conf_obj.common_fea_map:
                bucket_name = "common"
            elif feature_key in self.fea_conf_obj.item_fea_map:
                bucket_name = "item"
            elif feature_key in self.fea_conf_obj.creative_fea_map:
                bucket_name = "creative"
            if bucket_name is not None:
                if feature_key in bucket_embedding_maps[bucket_name]:
                    raise ValueError("duplicated lookup embedding for feature {}".format(feature_key))
                bucket_embedding_maps[bucket_name][feature_key] = sparse_embeddings[index]

        bucket_feature_ids = {
            "common": list(self.fea_conf_obj.common_fea_map.keys()),
            "item": list(self.fea_conf_obj.item_fea_map.keys()),
            "creative": list(self.fea_conf_obj.creative_fea_map.keys()),
        }
        bucket_embeddings = {}
        for bucket_name in self._BUCKET_NAMES:
            missing_ids = [
                feature_id for feature_id in bucket_feature_ids[bucket_name]
                if feature_id not in bucket_embedding_maps[bucket_name]
            ]
            if missing_ids:
                raise ValueError(
                    "lookup embeddings missing for {}: {}".format(bucket_name, missing_ids)
                )
            bucket_embeddings[bucket_name] = [
                bucket_embedding_maps[bucket_name][feature_id]
                for feature_id in bucket_feature_ids[bucket_name]
            ]

        self.dnn_input_map = {
            name: tf.concat(bucket_embeddings[name], axis=-1)
            for name in self._BUCKET_NAMES
        }

        with tf.variable_scope(
            "Cvr-task-part",
            reuse=tf.AUTO_REUSE,
            partitioner=self.partitioner,
        ) as cvr_scope:
            normalized_buckets = {}
            for name in self._BUCKET_NAMES:
                normalized_buckets[name] = self.batch_norm_layer_v2(
                    x=self.dnn_input_map[name],
                    train_phase=is_train,
                    scope_bn="bn_input_{}".format(name),
                    batch_norm_decay=self.batch_norm_decay,
                    use_riemann_bn=self.use_riemann_bn,
                    renorm=self.embed_use_renorm,
                    renorm_decay=self.embed_renorm_decay,
                    export=export,
                )

            if self.use_senet:
                gated_buckets = self.senet_layer(
                    normalized_buckets["common"],
                    normalized_buckets["item"],
                    normalized_buckets["creative"],
                    is_train,
                    export,
                )
                bucket_tensors = dict(zip(self._BUCKET_NAMES, gated_buckets))
            else:
                bucket_tensors = normalized_buckets

            bucket_field_maps = {}
            for name in self._BUCKET_NAMES:
                field_dims = [tensor.shape[-1].value for tensor in bucket_embeddings[name]]
                if any(dim is None for dim in field_dims):
                    raise ValueError("all field embedding dimensions must be statically known")
                if any(dim != self.embedding_size for dim in field_dims):
                    raise ValueError(
                        'RankMixer v6-E2-Small-2 expects every {} field embedding dim to equal {}, got {}'.format(
                            name,
                            self.embedding_size,
                            sorted(set(field_dims)),
                        )
                    )
                field_tensors = tf.split(bucket_tensors[name], field_dims, axis=-1)
                feature_ids = bucket_feature_ids[name]
                if len(field_tensors) != len(feature_ids):
                    raise ValueError("field tensor count mismatch for {}".format(name))
                bucket_field_maps[name] = dict(zip(feature_ids, field_tensors))

            local_tokens = self._semantic_tokenize(bucket_field_maps, export)
            global_token = self._build_global_token(bucket_tensors, export)
            input_tokens = tf.concat(
                [local_tokens, tf.expand_dims(global_token, axis=1)],
                axis=1,
            )
            input_tokens.set_shape([None, self.rm_token_num, self.rm_hidden_dim])

            hidden_tokens = self._rm_stack(input_tokens, export)
            final_tokens = self._rm_final_layer_norm(hidden_tokens)
            context = tf.reduce_mean(
                final_tokens,
                axis=1,
                name='rm_mean_pool',
            )
            context.set_shape([None, self.rm_hidden_dim])
            output = self._task_head(context, is_train, export)

            logits = tf.clip_by_value(tf.reshape(output, shape=[-1], name=mode), -self.clip_val, self.clip_val)
            predictions = tf.sigmoid(logits, name=mode)

        graph_dense_trainable_params = self._verify_graph_dense_trainable_params(
            cvr_scope.name
        )
        logging.info(
            "RankMixer v6-E2-Small-2 output: input_tokens=%s, hidden_tokens=%s, final_tokens=%s, "
            "context=%s, dense_params=%d, graph_dense_params=%d",
            input_tokens.get_shape(),
            hidden_tokens.get_shape(),
            final_tokens.get_shape(),
            context.get_shape(),
            self.rm_dense_trainable_param_count,
            graph_dense_trainable_params,
        )
        return {"logits": logits, "pred": predictions}
