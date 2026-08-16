# -*- coding: utf-8 -*-
# rankmixer_v3版本：完整实现训练生命周期，按硬编码业务语义分组生成common/item/creative token
import os
import math
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

from .model_utils import layer_norm

try:
    from cayman.python import cal_dot_topk_indices_no_padding, layer_norm_for_train
except ImportError:
    logging.info('cal_dot_topk_indices_no_padding, layer_norm_for_train import error')


class MLPModel(ModelBase):
    _BUCKET_NAMES = ('common', 'item', 'creative')

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

        # senet conf (kept for compat, not used by rankmixer tower)
        self.senet_hidden_size = _kwargs.get('senet_hidden_size', 128)
        self.use_senet = _kwargs.get('use_senet', False)
        self.use_senet_bn = _kwargs.get('use_senet_bn', False)

        # cvr model conf
        self.cvr_layers = _kwargs.get('cvr_layers', [2048, 2048, 256])
        self.opt_goal = _kwargs.get('opt_goal', 'first_cvr')
        self.export_name = _kwargs.get('export_name', 'first_cvr')
        self.cvr_label_name = _kwargs.get('cvr_label_name', 'fst_cvr_label')

        # rankmixer conf (T=16, H=16, D=768, L=2)
        self.use_rankmixer = _kwargs.get('use_rankmixer', True)
        self.rm_token_num = _kwargs.get('rm_token_num', 16)        # T
        self.rm_hidden_dim = _kwargs.get('rm_hidden_dim', 768)     # D
        self.rm_layer_num = _kwargs.get('rm_layer_num', 2)         # L
        self.rm_ffn_expand = int(_kwargs.get('rm_ffn_expand', 2))  # k
        self.rm_head_num = _kwargs.get('rm_head_num', self.rm_token_num)
        self.rm_ffn_act = _kwargs.get('rm_ffn_act', 'gelu_2')
        self.rm_proj_ln = _kwargs.get('rm_proj_ln', False)
        self.rm_token_proj_act = _kwargs.get('rm_token_proj_act', 'gelu_2')
        self.rm_use_gated_pool = bool(_kwargs.get('rm_use_gated_pool', True))
        self.rm_use_bucket_cross = bool(_kwargs.get('rm_use_bucket_cross', True))
        self.rm_cross_gate_init = float(_kwargs.get('rm_cross_gate_init', -2.0))

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
                'Semantic-Cross RankMixer v3 accepts only common/item/creative; '
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

        if sum(self.rm_bucket_token_counts) != self.rm_token_num:
            raise ValueError(
                'hard-coded semantic token count={} must equal rm_token_num={}'.format(
                    sum(self.rm_bucket_token_counts), self.rm_token_num
                )
            )
        if self.rm_head_num != self.rm_token_num:
            raise ValueError('RankMixer residual layout requires rm_head_num == rm_token_num')

        logging.info(
            'Semantic-Cross RankMixer v3: fields=%s, bucket_tokens=%s, T=%d, H=%d, '
            'D=%d, L=%d, k=%d, token_proj_act=%s, senet=%s, gated_pool=%s, bucket_cross=%s',
            field_counts,
            self.rm_bucket_token_counts,
            self.rm_token_num,
            self.rm_head_num,
            self.rm_hidden_dim,
            self.rm_layer_num,
            self.rm_ffn_expand,
            self.rm_token_proj_act,
            self.use_senet,
            self.rm_use_gated_pool,
            self.rm_use_bucket_cross,
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
        """Hard-coded semantic field groups, following the in-model list style used by reference models."""
        # 用户静态画像、设备与基础环境（16 个字段）
        common_profile_device_features = [
            '1001', '1006', '1014', '1034', '1035', '1036', '1041', '1042',
            '1043', '1501', '1504', '1505', '1527', '25003', '866024', '868023',
        ]

        # 用户交易、购买与消费价值（90 个字段）
        common_purchase_value_features = [
            '10231', '10232', '10233', '10442', '10522', '10601', '1104', '1106',
            '131480', '1502', '16725', '16727', '16731', '16733', '16735', '16737',
            '16739', '19013', '19016', '200300', '200302', '200303', '200304', '200306',
            '200320', '201702', '201704', '201756', '201757', '201914', '201915', '201939',
            '202218', '20517', '20521', '2066', '210000', '210001', '21264', '21403',
            '21404', '21749', '21750', '231056', '231065', '24082411', '24082412', '24082413',
            '25000', '25001', '25002', '25006', '25044', '25700', '26017', '26021',
            '26025', '26107', '790249', '790250', '790251', '862355', '862376', '863712',
            '863729', '866012', '866014', '866023', '866027', '866029', '866034', '866054',
            '866063', '866064', '866065', '866069', '866070', '33866903', '870277', '870322',
            '870324', '795602', '340123', '340121', '340125', '340093', '110151', '110153',
            '110011', '340122',
        ]

        # 用户长期浏览、点击、收藏与兴趣历史（92 个字段）
        common_interest_history_features = [
            '10600', '1063', '1064', '1065', '1121', '12403', '12438', '1509',
            '1512', '1521', '1524', '18021', '18094', '18098', '18100', '18105',
            '18214', '19024', '200124', '200318', '200319', '200714', '200715', '200762',
            '200764', '201720', '201905', '201906', '201909', '202144', '2073', '210015',
            '210042', '21055', '21233', '21238', '21239', '21240', '21246', '21257',
            '21258', '21260', '21303', '21307', '21340', '21355', '21602', '21610',
            '231383', '231384', '231484', '25045', '25046', '2509', '25702', '25703',
            '26035', '4500', '4501', '4502', '4503', '4504', '33600031', '860031',
            '860034', '860037', '860042', '860045', '863014', '863018', '866041', '866066',
            '866068', '866071', '866072', '866082', '867603', '870311', '2017702', '4418192',
            '4418196', '204530', '204543', '241125006', '031090', '340059', '340063', '340092',
            '340086', '340037', '340054', '340001',
        ]

        # 查询意图、检索词与召回上下文（85 个字段）
        common_query_intent_retrieval_features = [
            '12209', '12402', '15000', '15002', '16743', '16744', '200200', '200214',
            '200758', '20518', '2100', '2101', '2102', '2103', '2104', '211121',
            '2112', '215401', '246003', '25136', '25138', '27516', '3001', '3006',
            '3007', '3008', '3009', '600154', '6910', '6911', '6912', '7007737',
            '7007741', '7007746', '7007755', '790220', '790221', '790222', '790230', '794014',
            '794015', '794030', '794031', '794164', '794178', '794179', '794200', '794208',
            '794209', '794210', '794214', '794215', '794734', '794768', '794802', '863044',
            '863046', '863141', '866013', '866250', '866251', '868413', '868414', '870025',
            '2014601', '795014', '795012', '2015703', '2015709', '2015745', '2022401', '300091',
            '160034', '24904004', '306045', '310614', '87560211', '340453', '340394', '340364',
            '340374', '340367', '340451', '340377', '340455',
        ]

        # 实时会话、曝光与短期漏斗（102 个字段）
        common_realtime_session_funnel_features = [
            '1070', '12235', '13037', '13038', '13039', '18073', '18078', '18083',
            '200305', '200413', '200752', '201900', '201930', '201931', '201932', '201937',
            '202223', '202330', '202333', '202334', '202425', '202426', '21010', '21012',
            '21013', '21030', '21031', '21032', '21033', '2123', '21351', '21359',
            '21402', '215311', '215312', '215343', '215373', '21663', '24082402', '2503',
            '2504', '25049', '2505', '2506', '2507', '300000', '3003', '3004',
            '3014', '3015', '3016', '3020', '3102', '3103', '6900', '860023',
            '861818', '862311', '863024', '863030', '865600', '866061', '866073', '866103',
            '867645', '867648', '868404', '868405', '868407', '868427', '870038', '870059',
            '870069', '870130', '881102', '881104', '881203', '881204', '881206', '881215',
            '881402', '881404', '881663', '881664', '881665', '881687', '881691', '881711',
            '882303', '882304', '882306', '882305', '881820', '881816', '881842', '881818',
            '881834', '881817', '4439006', '304322', '340109', '340160',
        ]

        # 候选身份、类目、静态属性与质量（98 个字段）
        item_static_identity_quality_features = [
            '10003', '10012', '10013', '10014', '10016', '10018', '10020', '10021',
            '10022', '10059', '10062', '10068', '10210', '10410', '1086', '13020',
            '13021', '13022', '17194', '19041', '19042', '19044', '19047', '200311',
            '200313', '200314', '200727', '200729', '201705', '201716', '201735', '206056',
            '24082417', '27631', '27632', '4017', '500000', '500001', '500003', '500015',
            '5001', '5014', '600022', '600024', '6001', '600100', '600101', '6004',
            '6007', '6008', '6012', '6013', '6016', '6021', '6041', '6206',
            '6501', '7001', '7002', '7007708', '7007710', '7007711', '7007713', '7501',
            '7502', '865726', '870310', '881226', '881237', '881709', '881721', '881733',
            '881757', '241215001', '302502', '302503', '304952', '14237', '302552', '302185',
            '160070', '302595', '341104', '304946', '304911', '241215101', '1600912', '820000',
            '820001', '820004', '820025', '820061', '340076', '500300', '500301', '500302',
            '770656', '770657',
        ]

        # 标题、查询词、NER 与文本相关性（71 个字段）
        item_text_relevance_features = [
            '10219', '10419', '13002', '13005', '13006', '13009', '13010', '18501',
            '18503', '18504', '211100', '211130', '2115', '25106', '25113', '25116',
            '25120', '28003', '28013', '28017', '28019', '28023', '4003', '4007',
            '4009', '4012', '5410', '6870', '6871', '6888', '6892', '6893',
            '6894', '6914', '7806', '7809', '8501', '8502', '862616', '862844',
            '864132', '864157', '864215', '864219', '864386', '864410', '33204162', '33204180',
            '33204182', '33204187', '204202', '204242', '24808118', '37616', '37617', '37618',
            '87560127', '8112', '87560133', '37615', '341105', '340100', '340044', '3400141',
            '87560214', '340483', '341320', '341353', '341358', '3402761', '770584',
        ]

        # 图像、视频、向量与多模态相似性（58 个字段）
        item_multimodal_features = [
            '200640', '200780', '201021', '33203301', '33203302', '33203303', '33203308', '33203320',
            '33203330', '33203332', '33203333', '33203334', '24021', '621856', '6802', '864743',
            '864744', '864770', '864774', '865118', '865275', '865416', '865421', '865618',
            '865682', '865711', '33866914', '868036', '33868929', '33868943', '33868950', '870001',
            '2015493', '882369', '882371', '882223', '882227', '882233', '882225', '882235',
            '203742', '203797', '4418073', '33205180', '33205227', '206201', '206301', '206389',
            '206563', '206585', '340116', '3401661', '160077', '212402', '212422', '212432',
            '212502', '341265',
        ]

        # 候选当前价格、优惠券与促销供给（60 个字段）
        item_price_offer_features = [
            '10524', '10528', '140707', '16726', '16728', '16742', '16746', '16759',
            '20512', '22102', '22119', '22120', '24530', '24541', '27102', '27303',
            '27308', '27311', '27316', '27321', '27443', '27445', '27447', '27459',
            '27606', '27616', '27626', '27634', '27635', '27640', '6046', '6131',
            '6133', '6134', '6852', '6859', '780011', '863060', '868029', '622316',
            '622533', '622530', '241215065', '274471', '276351', '500103', '500120', '500121',
            '500134', '500135', '500136', '500137', '500150', '500151', '500158', '500159',
            '622555', '868030', '302533', '770521',
        ]

        # 用户价格偏好、价格差与价格排序（126 个字段）
        item_price_preference_features = [
            '10359', '10520', '11006', '12122', '131466', '131467', '131468', '131470',
            '131472', '131473', '131474', '131475', '131476', '131478', '131479', '131482',
            '131483', '131484', '131485', '200181', '200315', '200316', '200317', '33203310',
            '33203311', '33203312', '33203321', '33203331', '206077', '206081', '206082', '215393',
            '21668', '21669', '21702', '21708', '21726', '21728', '21729', '21743',
            '21746', '21752', '21760', '21762', '22101', '22106', '22129', '22131',
            '24328', '24330', '24332', '24496', '24497', '24498', '246004', '246005',
            '246006', '246007', '246014', '27367', '27402', '27507', '600001', '794165',
            '794201', '794212', '794213', '33866909', '33866912', '33866915', '33866926', '867665',
            '867685', '868291', '33868952', '33868953', '33868954', '33868961', '33868965', '33868969',
            '33868970', '33868973', '33868976', '33868977', '33868978', '870177', '870313', '870315',
            '881108', '2022429', '33795609', '33795608', '33795610', '203708', '33204181', '33204185',
            '33204186', '33204196', '4418101', '160065', '33205186', '206310', '206510', '208000',
            '208011', '208012', '208013', '208014', '208015', '310601', '310602', '310604',
            '110041', '900086', '3401321', '2022444', '340824', '208001', '208016', '900643',
            '770460', '770471', '770470', '770461', '770462', '770469',
        ]

        # 商品、类目和店铺的全局漏斗统计（73 个字段）
        item_global_statistics_features = [
            '10152', '10154', '10160', '10207', '10213', '10310', '10407', '10413',
            '140700', '19035', '24107', '24108', '24115', '24116', '24121', '24218',
            '24231', '24237', '24242', '24246', '24531', '24701', '24702', '24703',
            '24704', '24705', '24706', '24707', '24708', '24709', '24710', '24711',
            '25008', '25010', '25011', '25012', '25014', '25015', '25501', '25506',
            '25717', '600102', '600112', '6011', '60119', '6047', '621412', '621414',
            '621415', '621416', '621872', '621877', '621878', '6224', '6804', '6811',
            '810103', '810107', '810109', '810132', '863056', '863069', '622496', '160067',
            '241215011', '241215038', '341102', '6052', '304913', '340028', '820003', '600233',
            '770568',
        ]

        # 用户购买、下单、收藏等正向偏好（46 个字段）
        item_positive_preference_features = [
            '10010', '10216', '10387', '10388', '1110', '1111', '12100', '12118',
            '12119', '12120', '12157', '131048', '131049', '17033', '200105', '200106',
            '200310', '200325', '201809', '201916', '202096', '206029', '21053', '21054',
            '2111', '21201', '21202', '25027', '25073', '25093', '25711', '25721',
            '25741', '26003', '26007', '4014', '5019', '870270', '870279', '302302',
            '302554', '304393', '341103', '3400731', '341421', '770560',
        ]

        # 用户曝光、浏览、点击与停留互动（134 个字段）
        item_exposure_engagement_features = [
            '10002', '10007', '10008', '12088', '12092', '12094', '12101', '12104',
            '12110', '12111', '12112', '12113', '12115', '12117', '12155', '12204',
            '12205', '12206', '17135', '17136', '17137', '17139', '18010', '200104',
            '200107', '200324', '200585', '200615', '200751', '200753', '200765', '201717',
            '201910', '201911', '33203304', '33203306', '20500', '20501', '20504', '20505',
            '206051', '206157', '21034', '21035', '21036', '21037', '21050', '21051',
            '21052', '21242', '215334', '215337', '215350', '215399', '231333', '231334',
            '231344', '231374', '231494', '25048', '25059', '25060', '25504', '25515',
            '25751', '25752', '25754', '28060', '621842', '7007714', '7007715', '7007716',
            '33758666', '860066', '861534', '863009', '863047', '863054', '863087', '863132',
            '863133', '863210', '863286', '33866919', '33866925', '868486', '868500', '868513',
            '869300', '870166', '870257', '870263', '870264', '882385', '882417', '882353',
            '882419', '882354', '882416', '882326', '33203607', '4418001', '208030', '208034',
            '302342', '3029611', '304394', '304395', '1602601', '1602631', '304451', '304452',
            '3401371', '340317', '340321', '3403491', '340335', '820027', '820028', '820029',
            '820035', '340070', '302374', '770583', '770626', '770627', '770570', '770630',
            '770571', '770459', '770468', '770472', '770473', '7704561',
        ]

        # 当前会话、页面位置与候选上下文（33 个字段）
        item_session_context_features = [
            '12134', '12137', '12138', '12140', '21110', '21114', '21115', '24082404',
            '863062', '881220', '881221', '881265', '881267', '881284', '881309', '881353',
            '881634', '881681', '881705', '881717', '160063', '241215127', '241215027', '341888',
            '340096', '600200', '600202', '600201', '600255', '600253', '600254', '600249',
            '212611',
        ]

        # 召回、i2i/u2i、图关系与排序（136 个字段）
        item_retrieval_graph_features = [
            '131052', '17027', '17053', '17058', '17062', '17071', '17086', '17088',
            '17107', '17111', '17177', '17178', '18004', '18007', '18035', '18088',
            '18197', '200210', '200269', '200283', '200284', '200406', '200754', '200756',
            '200757', '200768', '201825', '201856', '201912', '201918', '24082423', '27525',
            '27533', '280501', '280502', '280602', '280611', '620000', '794005', '794007',
            '794021', '794022', '794023', '794169', '794170', '794171', '794202', '794203',
            '794204', '794205', '794206', '794207', '794211', '860076', '860090', '861060',
            '861124', '861201', '861213', '861219', '861504', '861540', '861612', '862388',
            '862391', '863780', '863802', '863808', '863811', '864553', '864578', '864738',
            '865093', '865341', '865342', '865344', '865349', '865420', '867638', '867689',
            '870008', '870012', '870128', '870195', '870250', '870283', '870303', '870340',
            '870357', '870373', '870402', '881025', '881331', '882401', '880448', '160033',
            '160043', '160044', '160049', '206206', '206550', '3044501', '310585', '310586',
            '4061', '87560220', '87580093', '900017', '900647', '909043', '909116', '302987',
            '87560205', '310588', '304456', '302190', '241125018', '247030061', '340296', '340856',
            '340827', '340756', '340761', '820007', '304383', '247031681', '2015723', '340859',
            '770592', '770607', '770591', '770590', '770587', '770588', '7704581', '770467',
        ]

        # 创意展示、图片标识与促销表达（14 个字段）
        creative_offer_features = [
            '780110', '780111', '780112', '780113', '780117', '8001', '8002', '8003',
            '8007', '8310', '500157', '900137', '8203', '8207',
        ]

        return {
            'common': [
                ('common_profile_device', common_profile_device_features),
                ('common_purchase_value', common_purchase_value_features),
                ('common_interest_history', common_interest_history_features),
                ('common_query_intent_retrieval', common_query_intent_retrieval_features),
                ('common_realtime_session_funnel', common_realtime_session_funnel_features),
            ],
            'item': [
                ('item_static_identity_quality', item_static_identity_quality_features),
                ('item_text_relevance', item_text_relevance_features),
                ('item_multimodal', item_multimodal_features),
                ('item_price_offer', item_price_offer_features),
                ('item_price_preference', item_price_preference_features),
                ('item_global_statistics', item_global_statistics_features),
                ('item_positive_preference', item_positive_preference_features),
                ('item_exposure_engagement', item_exposure_engagement_features),
                ('item_session_context', item_session_context_features),
                ('item_retrieval_graph', item_retrieval_graph_features),
            ],
            'creative': [
                ('creative_offer', creative_offer_features),
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
            logging.info(
                'RankMixer v3 semantic groups %s: %s',
                bucket_name,
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

        self.eval_summary = tf.summary.merge([loss_summary, auc_summary, copc_summary], name='eval_summary')

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

    def _project_semantic_group(self, field_tensors, bucket_name, group_name, token_idx, export):
        token_input = field_tensors[0] if len(field_tensors) == 1 else tf.concat(field_tensors, axis=-1)
        input_dim = token_input.shape[-1].value
        if input_dim is None:
            raise ValueError("semantic token input dimension must be statically known")

        if self.rm_token_proj_act in (None, "identity", "linear"):
            activation_fn = tf.identity
        else:
            activation_fn = self.get_act_func(self.rm_token_proj_act)

        with tf.variable_scope(
            group_name,
            reuse=tf.AUTO_REUSE,
            partitioner=self.partitioner,
        ):
            projected = tf.contrib.layers.fully_connected(
                inputs=token_input,
                num_outputs=self.rm_hidden_dim,
                activation_fn=activation_fn,
                weights_initializer=self.get_init(input_dim),
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_deep),
                biases_initializer=tf.zeros_initializer(),
                scope="projection",
            )
            if self.rm_proj_ln:
                projected = layer_norm(projected, name="projection_ln", export=export)

        logging.info(
            "semantic token %s/%s/%d: fields=%d, input_dim=%d -> D=%d",
            bucket_name,
            group_name,
            token_idx,
            len(field_tensors),
            input_dim,
            self.rm_hidden_dim,
        )
        return projected

    def _semantic_tokenize(self, bucket_field_maps, export):
        tokens = []
        with tf.variable_scope("rm_semantic_tokenize", reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            for bucket_name in self._BUCKET_NAMES:
                semantic_groups = self.rm_semantic_feature_groups[bucket_name]
                for token_idx, (group_name, feature_ids) in enumerate(semantic_groups):
                    group = [bucket_field_maps[bucket_name][feature_id] for feature_id in feature_ids]
                    tokens.append(
                        self._project_semantic_group(
                            group,
                            bucket_name,
                            group_name,
                            token_idx,
                            export,
                        )
                    )
        return tf.stack(tokens, axis=1)

    def _rm_multi_head_token_mixing(self, x, H, export=False):
        """Multi-Head Token Mixing：纯张量重排（无参数、无 K-Q 内积），[B,T,D] -> [B,T,D]"""
        T = int(x.shape[1])
        D = int(x.shape[2])
        assert D % H == 0, f"D={D} must be divisible by H={H}"
        head_dim = D // H
        reshaped = tf.reshape(x, [-1, T, H, head_dim])              # [B,T,H,head_dim]
        transposed = tf.transpose(reshaped, [0, 2, 1, 3])              # [B,H,T,head_dim]
        mixed = tf.reshape(transposed, [-1, H, T * head_dim])          # [B,H,T*head_dim]
        out = tf.reshape(mixed, [-1, T, D])                            # [B,T,D]
        logging.info(
            f"rm_token_mixing: {x.get_shape()} -> {out.get_shape()} (head_dim={head_dim}, H={H}, T={T})")
        return out

    def _rm_per_token_ffn(self, inputs, expansion):
        """Fused batched matmul with independent parameters for every token."""
        token_num = inputs.shape[1].value
        hidden_dim = inputs.shape[2].value
        expanded_dim = expansion * hidden_dim
        token_major = tf.transpose(inputs, [1, 0, 2])

        with tf.variable_scope("rm_pffn_batched", reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            w1 = tf.get_variable(
                "w1",
                shape=[token_num, hidden_dim, expanded_dim],
                initializer=self.get_init(hidden_dim),
            )
            b1 = tf.get_variable(
                "b1",
                shape=[token_num, 1, expanded_dim],
                initializer=tf.zeros_initializer(),
            )
            w2 = tf.get_variable(
                "w2",
                shape=[token_num, expanded_dim, hidden_dim],
                initializer=self.get_init(expanded_dim),
            )
            b2 = tf.get_variable(
                "b2",
                shape=[token_num, 1, hidden_dim],
                initializer=tf.zeros_initializer(),
            )
            hidden = self.get_act_func(self.rm_ffn_act)(tf.matmul(token_major, w1) + b1)
            output = tf.matmul(hidden, w2) + b2
        return tf.transpose(output, [1, 0, 2])

    def _rm_block(self, inputs, block_idx, export, mode="forward"):
        """Paper-faithful post-norm block: exactly two Add&Norm operations."""
        del mode
        with tf.variable_scope(
            "rm_block_{}".format(block_idx),
            reuse=tf.AUTO_REUSE,
            partitioner=self.partitioner,
        ):
            mixed = self._rm_multi_head_token_mixing(inputs, self.rm_head_num, export)
            with tf.variable_scope("token_mix_norm", reuse=tf.AUTO_REUSE):
                hidden = layer_norm(mixed + inputs, name="ln", export=export)

            transformed = self._rm_per_token_ffn(hidden, self.rm_ffn_expand)
            with tf.variable_scope("pffn_norm", reuse=tf.AUTO_REUSE):
                output = layer_norm(transformed + hidden, name="ln", export=export)
        return output

    def _rm_stack(self, x, layers, export=False, mode='forward'):
        for layer_idx in range(layers):
            x = self._rm_block(x, layer_idx, export, mode)
        return x

    def _pool_tokens(self, tokens):
        if not self.rm_use_gated_pool:
            return tf.reduce_mean(tokens, axis=1)

        # Zero initialization starts exactly as mean pooling, then learns an
        # instance-conditioned token weighting without pairwise attention.
        with tf.variable_scope("rm_gated_pool", reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            scores = tf.contrib.layers.fully_connected(
                inputs=tokens,
                num_outputs=1,
                activation_fn=tf.identity,
                weights_initializer=tf.zeros_initializer(),
                biases_initializer=None,
                scope="score",
            )
            weights = tf.nn.softmax(scores, dim=1)
        return tf.reduce_sum(tokens * weights, axis=1)

    def _bucket_cross_residual(self, input_tokens, export):
        bucket_vectors = []
        cursor = 0
        for token_count in self.rm_bucket_token_counts:
            bucket_vectors.append(tf.reduce_mean(input_tokens[:, cursor:cursor + token_count, :], axis=1))
            cursor += token_count

        common_vec, item_vec, creative_vec = bucket_vectors
        cross_input = tf.concat(
            [
                common_vec,
                item_vec,
                creative_vec,
                common_vec * item_vec,
                common_vec * creative_vec,
                item_vec * creative_vec,
            ],
            axis=-1,
        )

        with tf.variable_scope("rm_bucket_cross", reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            cross = tf.contrib.layers.fully_connected(
                inputs=cross_input,
                num_outputs=self.rm_hidden_dim,
                activation_fn=self.get_act_func(self.rm_ffn_act),
                weights_initializer=self.get_init(6 * self.rm_hidden_dim),
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_deep),
                biases_initializer=tf.zeros_initializer(),
                scope="projection",
            )
            cross = layer_norm(cross, name="cross_ln", export=export)
            gate_logit = tf.get_variable(
                "gate_logit",
                shape=[],
                initializer=tf.constant_initializer(self.rm_cross_gate_init),
            )
            cross = tf.nn.sigmoid(gate_logit) * cross
        return cross

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
                field_tensors = tf.split(bucket_tensors[name], field_dims, axis=-1)
                feature_ids = bucket_feature_ids[name]
                if len(field_tensors) != len(feature_ids):
                    raise ValueError("field tensor count mismatch for {}".format(name))
                bucket_field_maps[name] = dict(zip(feature_ids, field_tensors))

            input_tokens = self._semantic_tokenize(bucket_field_maps, export)
            hidden_tokens = self._rm_stack(
                input_tokens,
                self.rm_layer_num,
                export=export,
                mode=mode,
            )
            context = self._pool_tokens(hidden_tokens)

            if self.rm_use_bucket_cross:
                cross_residual = self._bucket_cross_residual(input_tokens, export)
                with tf.variable_scope("rm_fusion_norm", reuse=tf.AUTO_REUSE):
                    context = layer_norm(context + cross_residual, name="ln", export=export)

            with tf.device("/job:ps/task:0"):
                # 保留现有输出 scope，兼容当前项目的 skip_tensors / warm_up_tensors 配置。
                output = tf.contrib.layers.fully_connected(
                    inputs=context,
                    num_outputs=1,
                    activation_fn=tf.identity,
                    weights_initializer=self.get_init(self.rm_hidden_dim),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_deep),
                    scope="rm_out_v2",
                )

            logits = tf.clip_by_value(tf.reshape(output, shape=[-1], name=mode), -self.clip_val, self.clip_val)
            predictions = tf.sigmoid(logits, name=mode)

        logging.info(
            "Semantic-Cross RankMixer output: input_tokens=%s, hidden_tokens=%s, context=%s",
            input_tokens.get_shape(),
            hidden_tokens.get_shape(),
            context.get_shape(),
        )
        return {"logits": logits, "pred": predictions}
