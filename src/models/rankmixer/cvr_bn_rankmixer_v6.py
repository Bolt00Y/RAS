# -*- coding: utf-8 -*-
# RankMixer v6 与 v5 的区别：
# 1. Token hidden dimension D 从 1024 降为 512；在 H=T=32 时，head_dim 从 32 降为 16。
# 2. 其余模型结构和训练目标保持不变：31 个 Local Token + 1 个 Global Token、H=32、L=2、
#    Per-token SwiGLU hidden M=704、Mixing/Reverting、PreNorm + RMSNorm、SENet、增强读出、
#    [2048, 2048, 256] 任务头，以及单一 first-CVR BCE。
# 3. 在配套 v6 启动配置（除 D 外与 v5 一致）下，Dense 可训练参数量为 177,217,126
#    （约 177.22M）；不包含稀疏 Embedding 表、优化器状态、指标变量和 BN moving statistics。
# 4. 本文件是 v6 的完整独立实现，不导入、不继承 cvr_bn_rankmixer_v5。
# 5. 冻结字段分组 ID、变量 scope 与指标 tag 中保留 v5 字样，以保持分组校验、warm-up 配置和
#    checkpoint 命名兼容；这些仅是稳定名称，不是对 v5 Python 实现的引用。
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
    _GROUP_VERSION = 'rankmixer_v5_balanced_v1'
    _GROUP_CHECKSUMS = {
        'common': '888b2efe451c5c01d647bd6f5a7e91acc5e66818dba676504e6820fc82fef155',
        'item': '511fa8e8162d0ebde63c1d91c4d921183fae41a093596f9a32621eefdbcafadd',
        'creative': '6a893a087a17b08f063710a4edeeb07b532d5642ae560a2b78eaf67fd198ce85',
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
        self.use_mlp_rms_norm = _kwargs.get('use_mlp_rms_norm', False)
        self.use_input_rms_norm = _kwargs.get('use_input_rms_norm', False)

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
        self.cvr_layers = [int(value) for value in _kwargs.get('cvr_layers', [2048, 2048, 256])]
        if not self.cvr_layers or any(value <= 0 for value in self.cvr_layers):
            raise ValueError('cvr_layers must contain positive dimensions')
        self.opt_goal = _kwargs.get('opt_goal', 'first_cvr')
        self.export_name = _kwargs.get('export_name', 'first_cvr')
        self.cvr_label_name = _kwargs.get('cvr_label_name', 'fst_cvr_label')

        # RankMixer v6: 31 local tokens + 1 global token, H=T=32, D=512, L=2.
        self.use_rankmixer = _kwargs.get('use_rankmixer', True)
        self.rm_token_num = int(_kwargs.get('rm_token_num', 32))
        self.rm_local_token_num = int(_kwargs.get('rm_local_token_num', 31))
        self.rm_hidden_dim = int(_kwargs.get('rm_hidden_dim', 512))
        self.rm_layer_num = int(_kwargs.get('rm_layer_num', 2))
        self.rm_head_num = _kwargs.get('rm_head_num', self.rm_token_num)
        self.rm_head_num = int(self.rm_head_num)
        self.rm_swiglu_hidden_dim = int(_kwargs.get('rm_swiglu_hidden_dim', 704))
        self.rm_down_init_scale = float(_kwargs.get('rm_down_init_scale', 0.01))
        self.rm_rms_epsilon = float(_kwargs.get('rm_rms_epsilon', 1e-6))
        self.rm_token_proj_act = _kwargs.get('rm_token_proj_act', 'gelu_2')
        self.rm_pool_query_dim = int(_kwargs.get('rm_pool_query_dim', 128))
        self.rm_flatten_dim = int(_kwargs.get('rm_flatten_dim', 512))
        self.rm_flatten_gate_init = float(_kwargs.get('rm_flatten_gate_init', -2.0))

        if not self.use_rankmixer:
            raise ValueError('RankMixer v5 requires use_rankmixer=true')
        if self.embedding_size != 17:
            raise ValueError('RankMixer v5 requires embedding_size=17, got {}'.format(self.embedding_size))
        if self.rm_token_num != self.rm_local_token_num + 1:
            raise ValueError(
                'rm_token_num={} must equal rm_local_token_num+1={}'.format(
                    self.rm_token_num,
                    self.rm_local_token_num + 1,
                )
            )
        if self.rm_head_num != self.rm_token_num:
            raise ValueError('RankMixer v5 requires rm_head_num == rm_token_num')
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
        if self.rm_rms_epsilon <= 0.0:
            raise ValueError('rm_rms_epsilon must be positive')
        if self.rm_pool_query_dim <= 0 or self.rm_pool_query_dim > self.rm_hidden_dim:
            raise ValueError('rm_pool_query_dim must be in (0, rm_hidden_dim]')
        if self.rm_flatten_dim <= 0:
            raise ValueError('rm_flatten_dim must be positive')

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
                'RankMixer v5 accepts only common/item/creative; '
                'non-empty extra buckets: {}'.format(nonempty_unsupported)
            )

        field_counts = [
            len(self.fea_conf_obj.common_fea_map),
            len(self.fea_conf_obj.item_fea_map),
            len(self.fea_conf_obj.creative_fea_map),
        ]
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

        logging.info(
            'RankMixer v5: group_version=%s, fields=%s, local_bucket_tokens=%s, '
            'T=%d, H=%d, D=%d, L=%d, swiglu_hidden=%d, down_init_scale=%s, rms_eps=%s, '
            'token_proj_act=%s, senet=%s, pool_query_dim=%d, flatten_dim=%d',
            self._GROUP_VERSION,
            field_counts,
            self.rm_bucket_token_counts,
            self.rm_token_num,
            self.rm_head_num,
            self.rm_hidden_dim,
            self.rm_layer_num,
            self.rm_swiglu_hidden_dim,
            self.rm_down_init_scale,
            self.rm_rms_epsilon,
            self.rm_token_proj_act,
            self.use_senet,
            self.rm_pool_query_dim,
            self.rm_flatten_dim,
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
        """Return the frozen balanced field groups used by RankMixer v5."""
        # Generated once with the fixed v5 salt; runtime never hashes or reshuffles fields.
        return {
            'common': [
                ('common_v5_00', [
                    '868427', '231383', '866041', '882306', '16735', '25044', '201932', '215343',
                    '794208', '794210', '340367', '21750', '340125', '10232', '12209', '25001',
                    '204543', '1070', '866066', '231056', '25006', '1512', '21238', '031090',
                    '201756', '863018', '1035', '27516', '860034', '866023', '18214', '215312',
                    '866014', '3001', '16733', '866250', '881206', '7007741', '200306',
                ]),
                ('common_v5_01', [
                    '201930', '4418196', '200318', '865600', '340455', '2104', '24904004', '26107',
                    '25000', '340377', '870130', '21246', '3016', '2509', '794802', '4439006',
                    '231384', '2014601', '868404', '18094', '20518', '21404', '881842', '26017',
                    '33866903', '870277', '1001', '795012', '21033', '25003', '340109', '340451',
                    '870069', '340123', '866024', '881404', '21257', '201909', '866063',
                ]),
                ('common_v5_02', [
                    '870322', '1524', '24082402', '201939', '201757', '246003', '110151', '4504',
                    '25700', '340394', '6900', '19016', '210001', '7007737', '881663', '881834',
                    '202334', '211121', '881820', '18083', '21602', '863044', '19024', '794734',
                    '241125006', '204530', '790230', '1505', '794768', '861818', '881104', '10600',
                    '202223', '790222', '881816', '210015', '866068', '18105', '860045',
                ]),
                ('common_v5_03', [
                    '16727', '200124', '794214', '7007746', '881203', '881215', '3003', '867645',
                    '200764', '33600031', '16731', '202425', '2103', '7007755', '2022401', '790250',
                    '200300', '2504', '1501', '4418192', '866034', '200214', '860037', '16743',
                    '3004', '1527', '340160', '200305', '13037', '1034', '21663', '866027',
                    '3102', '4501', '201702', '1504', '795602', '215373', '202333',
                ]),
                ('common_v5_04', [
                    '863141', '202330', '24082413', '866073', '340093', '882304', '870311', '881818',
                    '24082412', '200752', '868023', '201704', '215401', '15002', '201906', '3015',
                    '21013', '2507', '863046', '202218', '201914', '21055', '866013', '25702',
                    '3008', '881691', '794015', '863712', '866251', '21030', '340037', '25002',
                    '160034', '1502', '131480', '1036', '863729', '21610', '16744',
                ]),
                ('common_v5_05', [
                    '2101', '110011', '201905', '200200', '1043', '866012', '12438', '200302',
                    '87560211', '201931', '870059', '340059', '200304', '21403', '600154', '300091',
                    '25703', '12402', '19013', '10522', '210000', '304322', '3006', '866029',
                    '881687', '3007', '201720', '10233', '2102', '18073', '200413', '866070',
                    '868407', '870025', '201937', '4503', '3020', '2015745',
                ]),
                ('common_v5_06', [
                    '862311', '20521', '882303', '15000', '21264', '1041', '794178', '868405',
                    '21260', '863014', '200758', '1065', '3103', '200715', '870324', '868413',
                    '1521', '201900', '21010', '866069', '306045', '26021', '231484', '881664',
                    '21355', '860023', '4502', '340364', '340092', '210042', '794030', '1042',
                    '868414', '21031', '2503', '4500', '866064', '202426',
                ]),
                ('common_v5_07', [
                    '2015703', '866082', '340001', '25138', '202144', '2123', '1014', '860031',
                    '862355', '1104', '870038', '110153', '215311', '795014', '6911', '2100',
                    '340374', '866103', '6912', '881402', '10442', '2015709', '25045', '2073',
                    '794031', '20517', '25049', '790220', '863030', '21032', '300000', '790221',
                    '866071', '18100', '1064', '1006', '1063', '867648',
                ]),
                ('common_v5_08', [
                    '16737', '790251', '6910', '3014', '310614', '25046', '21749', '2066',
                    '10601', '794215', '21307', '862376', '201915', '21340', '867603', '866065',
                    '26035', '21240', '2112', '200320', '231065', '21303', '1509', '24082411',
                    '200303', '881665', '881204', '18078', '2017702', '340122', '790249', '200714',
                    '10231', '794164', '21239', '794179', '200319', '881711',
                ]),
                ('common_v5_09', [
                    '21402', '340086', '340063', '25136', '2505', '1106', '340054', '12403',
                    '794209', '866054', '12235', '16739', '21258', '3009', '882305', '881817',
                    '881102', '13038', '21012', '21351', '26025', '13039', '2506', '340453',
                    '21359', '860042', '794200', '16725', '18098', '866072', '794014', '863024',
                    '866061', '18021', '1121', '21233', '200762', '340121',
                ]),
            ],
            'item': [
                ('item_v5_00', [
                    '200757', '33204187', '340827', '10210', '881331', '12113', '10062', '8501',
                    '27402', '17033', '28019', '160067', '33203306', '861060', '10520', '21037',
                    '600200', '25008', '340100', '211100', '21052', '621872', '131470', '27443',
                    '24115', '4009', '215337', '621414', '340483', '200316', '10388', '25751',
                    '17086', '10359', '25059', '600022', '341421', '882385', '24707', '864774',
                    '1111', '200640',
                ]),
                ('item_v5_01', [
                    '200106', '241215065', '6133', '12120', '160033', '212502', '206056', '868486',
                    '500134', '770592', '24701', '870177', '900647', '33866909', '862388', '201911',
                    '794169', '341265', '33205186', '27445', '131049', '206550', '863009', '18035',
                    '867638', '881265', '7007716', '25717', '206029', '862391', '794023', '770467',
                    '33795609', '6041', '794170', '770468', '27634', '863808', '870008', '25027',
                    '21726', '131482',
                ]),
                ('item_v5_02', [
                    '870313', '7502', '10216', '208014', '16759', '24711', '33758666', '882225',
                    '25741', '794211', '33204185', '12092', '600255', '302595', '26007', '27533',
                    '881733', '7001', '17111', '860076', '16728', '203797', '87560127', '341358',
                    '6008', '140700', '6811', '794021', '4007', '200324', '3401661', '7007711',
                    '870373', '882419', '302503', '24705', '206301', '881705', '868036', '17053',
                    '16742', '864738',
                ]),
                ('item_v5_03', [
                    '770470', '882371', '865342', '160065', '340296', '10016', '22101', '208013',
                    '770584', '201021', '341105', '160077', '900643', '621878', '864770', '341353',
                    '19047', '208015', '206389', '206077', '246005', '12112', '794212', '231374',
                    '10010', '212402', '794165', '21752', '21035', '864157', '131473', '600101',
                    '19042', '861540', '20505', '37617', '863056', '861504', '33203304', '212432',
                    '17136', '870012',
                ]),
                ('item_v5_04', [
                    '500302', '200313', '600233', '17058', '12138', '201916', '25014', '340321',
                    '27626', '21762', '5410', '131472', '200765', '17177', '24332', '206081',
                    '201910', '865341', '820061', '500158', '864132', '17062', '621416', '7501',
                    '909116', '820025', '865726', '18503', '869300', '863286', '770471', '882416',
                    '247030061', '310601', '25012', '4418001', '24709', '13022', '341102', '10013',
                    '7002', '21036',
                ]),
                ('item_v5_05', [
                    '200314', '863087', '868513', '25116', '870283', '10407', '500300', '21114',
                    '861534', '881284', '33203334', '200105', '200210', '160044', '770657', '865420',
                    '304913', '246006', '21242', '33203332', '33868978', '6892', '600112', '865618',
                    '24108', '25060', '870263', '600202', '200283', '25752', '33203333', '27525',
                    '25504', '881717', '200325', '24330', '12088', '770630', '794203', '6011',
                    '33204182', '600254',
                ]),
                ('item_v5_06', [
                    '868500', '304911', '881721', '33868973', '500151', '19041', '770571', '201716',
                    '5019', '131479', '5001', '770568', '820007', '3400731', '4014', '2115',
                    '276351', '24531', '215334', '6893', '33868965', '600249', '33203301', '19044',
                    '201735', '20504', '6870', '600001', '10413', '863054', '18010', '6012',
                    '24498', '881309', '33866912', '27311', '200310', '882223', '33204181', '206206',
                    '770590', '500103',
                ]),
                ('item_v5_07', [
                    '870340', '21708', '24496', '10213', '794206', '6007', '33203310', '882401',
                    '33203302', '3400141', '7704561', '21202', '10018', '882354', '131475', '863060',
                    '241215001', '7809', '770560', '13005', '6501', '6016', '200284', '24121',
                    '24231', '13021', '770461', '867665', '27308', '206563', '37616', '200756',
                    '870257', '33868953', '881237', '200768', '24704', '794005', '770607', '200727',
                    '200780', '870402',
                ]),
                ('item_v5_08', [
                    '25120', '17071', '302374', '4003', '28023', '202096', '22129', '18504',
                    '304451', '794202', '211130', '27632', '500159', '882235', '870166', '770570',
                    '33868977', '794201', '33866915', '861124', '25113', '310604', '500001', '10007',
                    '600102', '882417', '770460', '33204180', '864410', '22120', '341103', '33203312',
                    '302533', '1602601', '304952', '25721', '200315', '621412', '21053', '33203303',
                    '863047', '201912',
                ]),
                ('item_v5_09', [
                    '200311', '200585', '12094', '21054', '881267', '2022444', '25011', '274471',
                    '206201', '26003', '17027', '863133', '863062', '7007714', '25106', '7007713',
                    '18197', '10410', '24116', '863210', '206051', '131478', '6046', '212422',
                    '241215027', '27616', '27635', '8112', '770472', '203742', '27631', '3402761',
                    '10154', '3403491', '131483', '13006', '500120', '215350', '17178', '6004',
                    '203708', '21110',
                ]),
                ('item_v5_10', [
                    '27459', '37615', '21743', '882326', '865349', '201918', '6134', '820035',
                    '200317', '25506', '500000', '231333', '302502', '33795608', '206157', '302185',
                    '865421', '10022', '867685', '500301', '770591', '10008', '33204162', '1086',
                    '621877', '340028', '27606', '810103', '340756', '310588', '27102', '160063',
                    '33203321', '87560220', '131048', '33868943', '17088', '340859', '33203331', '882233',
                    '19035', '881226',
                ]),
                ('item_v5_11', [
                    '340076', '200754', '304395', '870195', '200107', '10021', '6871', '6013',
                    '310586', '208001', '33868970', '10207', '863811', '622555', '20501', '770626',
                    '622533', '21746', '12115', '810132', '200269', '13009', '33868961', '3044501',
                    '21201', '33203607', '24107', '864743', '87560214', '241215127', '21051', '870250',
                    '863132', '820001', '340096', '881108', '304946', '33795610', '22106', '870279',
                    '25073', '5014',
                ]),
                ('item_v5_12', [
                    '13002', '621856', '865118', '24246', '820027', '28003', '87580093', '861612',
                    '160049', '881353', '24530', '33868929', '21728', '770587', '12140', '865093',
                    '131052', '865275', '60119', '7007715', '770462', '160070', '304456', '780011',
                    '794204', '21115', '870357', '10219', '21669', '24708', '621415', '909043',
                    '500136', '24328', '247031681', '241215101', '160043', '87560133', '340761', '882353',
                    '302552', '280501',
                ]),
                ('item_v5_13', [
                    '33868954', '770583', '24710', '6131', '6852', '24541', '33203308', '241215038',
                    '200753', '17135', '4418073', '10528', '18088', '7007710', '20512', '24706',
                    '208000', '204202', '2022429', '302554', '770656', '12157', '6001', '861213',
                    '770469', '6802', '2015493', '4061', '870001', '770588', '7007708', '200181',
                    '24237', '6859', '20500', '11006', '880448', '25048', '212611', '6804',
                    '622496', '13010',
                ]),
                ('item_v5_14', [
                    '37618', '340044', '868291', '341104', '21034', '860066', '208030', '3401371',
                    '500150', '12118', '794007', '33205227', '10003', '231334', '208016', '33868969',
                    '206310', '24218', '12134', '200406', '794022', '12119', '10020', '215393',
                    '10012', '870264', '861201', '2015723', '304452', '131468', '201809', '12100',
                    '131485', '12205', '4418101', '208034', '21760', '18004', '864553', '500003',
                    '10014', '33866925',
                ]),
                ('item_v5_15', [
                    '33868950', '16746', '881221', '862616', '24082417', '7806', '25754', '27640',
                    '131474', '12204', '7704581', '231344', '863802', '16726', '882227', '310585',
                    '206585', '24021', '25010', '27303', '770473', '861219', '33866926', '881681',
                    '200104', '302302', '340317', '12117', '810109', '131476', '881634', '340116',
                    '110041', '500135', '881025', '820028', '33203311', '868029', '810107', '201856',
                    '21050',
                ]),
                ('item_v5_16', [
                    '3029611', '500015', '24082404', '18501', '10059', '2111', '200751', '280602',
                    '12104', '33868976', '302342', '8502', '27447', '25515', '208012', '794205',
                    '10387', '302190', '600024', '21702', '794207', '24497', '12101', '6206',
                    '12110', '246014', '864215', '600253', '622530', '862844', '18007', '1602631',
                    '280611', '28013', '870310', '14237', '770521', '860090', '208011', '33204186',
                    '21668',
                ]),
                ('item_v5_17', [
                    '28060', '6021', '25711', '27321', '24702', '6052', '204242', '600100',
                    '341320', '863780', '140707', '24703', '131466', '621842', '24082423', '12206',
                    '820004', '900017', '6914', '340856', '12122', '201705', '33866919', '794171',
                    '622316', '304394', '820003', '33866914', '820000', '341888', '22102', '131484',
                    '131467', '870128', '870270', '87560205', '201717', '4017', '500121', '865416',
                    '280502',
                ]),
                ('item_v5_18', [
                    '206510', '304393', '33203320', '500137', '770459', '870315', '33868952', '27316',
                    '33205180', '865682', '820029', '17107', '10152', '881709', '206082', '231494',
                    '10068', '340824', '6047', '865711', '882369', '33203330', '864386', '10310',
                    '864578', '28017', '620000', '201825', '881220', '868030', '10524', '12155',
                    '27367', '3401321', '22119', '900086', '10160', '25015', '302987', '770627',
                    '340335',
                ]),
                ('item_v5_19', [
                    '27507', '200729', '241125018', '6224', '25093', '17139', '870303', '865344',
                    '12111', '12137', '864744', '17137', '6888', '794213', '24808118', '13020',
                    '10002', '864219', '24242', '33204196', '6894', '1110', '340070', '4012',
                    '246004', '863069', '25501', '246007', '304383', '22131', '310602', '600201',
                    '215399', '17194', '867689', '881757', '10419', '1600912', '21729', '241215011',
                    '200615',
                ]),
            ],
            'creative': [
                ('creative_v5_00', [
                    '8001', '8310', '8203', '8207', '780111', '780117', '8002', '900137',
                    '780113', '780112', '780110', '8007', '8003', '500157',
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
                    'RankMixer v5 group sizes mismatch for {}: actual={}, expected={}'.format(
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
                    'RankMixer v5 group checksum mismatch for {}: actual={}, expected={}'.format(
                        bucket_name,
                        checksum,
                        expected_checksum,
                    )
                )
            logging.info(
                'RankMixer v5 frozen groups %s: version=%s, checksum=%s, groups=%s',
                bucket_name,
                self._GROUP_VERSION,
                checksum,
                [(name, len(ids)) for name, ids in bucket_groups],
            )

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
        pool_entropy_summary = tf.summary.scalar(
            f'{mode}/rm_v5/pool_entropy',
            self.rm_pool_entropy,
        )
        flatten_gate_summary = tf.summary.scalar(
            f'{mode}/rm_v5/flatten_gate',
            self.rm_flatten_gate,
        )

        self.eval_summary = tf.summary.merge(
            [
                loss_summary,
                auc_summary,
                copc_summary,
                pool_entropy_summary,
                flatten_gate_summary,
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
        """RMSNorm with either per-token gamma [T,D] or shared gamma [D]."""
        shape = inputs.get_shape().as_list()
        if len(shape) not in (2, 3):
            raise ValueError('RMSNorm expects rank-2 or rank-3 input, got {}'.format(shape))
        hidden_dim = shape[-1]
        if hidden_dim is None:
            raise ValueError('RMSNorm hidden dimension must be statically known')

        if per_token:
            if len(shape) != 3 or shape[1] is None:
                raise ValueError('per-token RMSNorm requires static [B,T,D], got {}'.format(shape))
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

    def _semantic_tokenize(self, bucket_field_maps):
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
                for family_index, (token_index, _) in enumerate(family):
                    tokens[token_index] = family_output[:, family_index, :]

            if any(token is None for token in tokens):
                raise ValueError('RankMixer v5 failed to project every local token')
            local_tokens = tf.stack(tokens, axis=1)
            local_tokens = self._rm_rms_norm(
                local_tokens,
                scope='local_token_rms_norm',
                per_token=True,
            )

        for token_index, (bucket_name, group_name, feature_ids, _, input_dim) in enumerate(token_specs):
            logging.info(
                'RankMixer v5 local token %d: bucket=%s, name=%s, fields=%d, input_dim=%d, D=%d',
                token_index,
                bucket_name,
                group_name,
                len(feature_ids),
                input_dim,
                self.rm_hidden_dim,
            )
        return local_tokens

    def _build_global_token(self, bucket_tensors):
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
            global_token = self._rm_rms_norm(
                global_token,
                scope='rms_norm',
                per_token=False,
            )

        logging.info(
            'RankMixer v5 global token: input_dim=%d -> %d -> %d',
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
            raise ValueError('RankMixer v5 requires H=T so mixed dimension remains D')
        return mixed

    def _rm_revert_tokens(self, mixed):
        """Exact inverse of _rm_mix_tokens for the v5 H=T layout."""
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

    def _rm_block(self, inputs, block_idx):
        """PreNorm Mixing/Reverting block with two independent Per-token SwiGLUs."""
        with tf.variable_scope(
            'rm_block_{}'.format(block_idx),
            reuse=tf.AUTO_REUSE,
            partitioner=self.partitioner,
        ):
            mixed = self._rm_mix_tokens(inputs)
            mixed_norm = self._rm_rms_norm(
                mixed,
                scope='mixed_rms_norm',
                per_token=True,
            )
            mixed_update = self._rm_per_token_swiglu(
                mixed_norm,
                scope='mixed_swiglu',
            )
            mixed_hidden = mixed + mixed_update

            reverted = self._rm_revert_tokens(mixed_hidden)
            original_norm = self._rm_rms_norm(
                reverted,
                scope='original_rms_norm',
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
            'RankMixer v5 block %d: input=%s, mixed=%s, reverted=%s, output=%s',
            block_idx,
            inputs.get_shape(),
            mixed.get_shape(),
            reverted.get_shape(),
            output.get_shape(),
        )
        return output

    def _rm_stack(self, inputs):
        output = inputs
        for block_idx in range(self.rm_layer_num):
            output = self._rm_block(output, block_idx)
        return output

    def _global_conditioned_pool(self, final_tokens):
        """Use the final global token to select information from 31 local tokens."""
        local_tokens = final_tokens[:, :self.rm_local_token_num, :]
        global_token = final_tokens[:, self.rm_local_token_num, :]

        with tf.variable_scope(
            'rm_global_conditioned_pool',
            reuse=tf.AUTO_REUSE,
            partitioner=self.partitioner,
        ):
            query = tf.contrib.layers.fully_connected(
                inputs=global_token,
                num_outputs=self.rm_pool_query_dim,
                activation_fn=tf.identity,
                weights_initializer=self.get_init(self.rm_hidden_dim),
                biases_initializer=tf.zeros_initializer(),
                scope='query',
            )
            keys = tf.contrib.layers.fully_connected(
                inputs=local_tokens,
                num_outputs=self.rm_pool_query_dim,
                activation_fn=tf.identity,
                weights_initializer=self.get_init(self.rm_hidden_dim),
                biases_initializer=tf.zeros_initializer(),
                scope='key',
            )
            scores = tf.reduce_sum(
                keys * tf.expand_dims(query, axis=1),
                axis=-1,
            ) / math.sqrt(self.rm_pool_query_dim)
            weights = tf.nn.softmax(scores, dim=1)
            pooled = tf.reduce_sum(
                local_tokens * tf.expand_dims(weights, axis=-1),
                axis=1,
            )
        return global_token, pooled, local_tokens, weights

    def _flatten_readout(self, local_tokens):
        """Keep token identity information that a weighted sum can discard."""
        flat_dim = self.rm_local_token_num * self.rm_hidden_dim
        flattened = tf.reshape(local_tokens, [-1, flat_dim])
        with tf.variable_scope(
            'rm_flatten_readout',
            reuse=tf.AUTO_REUSE,
            partitioner=self.partitioner,
        ):
            route = tf.contrib.layers.fully_connected(
                inputs=flattened,
                num_outputs=self.rm_flatten_dim,
                activation_fn=self.get_act_func(self.rm_token_proj_act),
                weights_initializer=self.get_init(flat_dim),
                biases_initializer=tf.zeros_initializer(),
                scope='projection',
            )
            route = self._rm_rms_norm(
                route,
                scope='rms_norm',
                per_token=False,
            )
            gate_logit = tf.get_variable(
                'gate_logit',
                shape=[],
                initializer=tf.constant_initializer(self.rm_flatten_gate_init),
            )
            gate = tf.nn.sigmoid(gate_logit)
        return gate * route, gate

    def _task_head(self, context, is_train, export):
        """Base-aligned [2048,2048,256] single-task prediction tower."""
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
                scope='rm_v5_mlp{}'.format(layer_idx),
            )
            if self.batch_norm:
                hidden = ModelBase.batch_norm_layer_v2(
                    x=hidden,
                    train_phase=is_train,
                    scope_bn='rm_v5_bn_{}'.format(layer_idx),
                    batch_norm_decay=self.batch_norm_decay,
                    use_riemann_bn=self.use_riemann_bn,
                    export=export,
                )
            hidden = self.get_act_func(self.mlp_act_type)(hidden)

        with tf.device('/job:ps/task:0'):
            output = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=1,
                activation_fn=tf.identity,
                weights_initializer=self.get_init(hidden.shape[-1].value),
                biases_initializer=tf.zeros_initializer(),
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

        with tf.variable_scope("Cvr-task-part", reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
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
                        'RankMixer v5 expects every {} field embedding dim to equal {}, got {}'.format(
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

            local_tokens = self._semantic_tokenize(bucket_field_maps)
            global_token = self._build_global_token(bucket_tensors)
            input_tokens = tf.concat(
                [local_tokens, tf.expand_dims(global_token, axis=1)],
                axis=1,
            )
            input_tokens.set_shape([None, self.rm_token_num, self.rm_hidden_dim])

            hidden_tokens = self._rm_stack(input_tokens)
            final_tokens = self._rm_rms_norm(
                hidden_tokens,
                scope='rm_final_rms_norm',
                per_token=True,
            )

            (
                global_context,
                pooled_context,
                final_local_tokens,
                pool_weights,
            ) = self._global_conditioned_pool(final_tokens)
            flatten_context, flatten_gate = self._flatten_readout(final_local_tokens)
            context = tf.concat(
                [global_context, pooled_context, flatten_context],
                axis=-1,
            )
            context.set_shape([
                None,
                2 * self.rm_hidden_dim + self.rm_flatten_dim,
            ])
            output = self._task_head(context, is_train, export)

            # Diagnostics are read-only tensors and never enter the BCE objective.
            self.rm_pool_entropy = tf.reduce_mean(
                -tf.reduce_sum(
                    pool_weights * tf.log(tf.maximum(pool_weights, 1e-8)),
                    axis=1,
                )
            )
            self.rm_flatten_gate = flatten_gate

            logits = tf.clip_by_value(tf.reshape(output, shape=[-1], name=mode), -self.clip_val, self.clip_val)
            predictions = tf.sigmoid(logits, name=mode)

        logging.info(
            "RankMixer v5 output: input_tokens=%s, hidden_tokens=%s, final_tokens=%s, context=%s",
            input_tokens.get_shape(),
            hidden_tokens.get_shape(),
            final_tokens.get_shape(),
            context.get_shape(),
        )
        return {"logits": logits, "pred": predictions}
