# -*- coding: utf-8 -*-
# rankmixer_v2版本：完整实现训练生命周期，按common/item/creative三桶保留字段边界生成语义token
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
                'Semantic-Cross RankMixer v2 accepts only common/item/creative; '
                'non-empty extra buckets: {}'.format(nonempty_unsupported)
            )

        field_counts = [
            len(self.fea_conf_obj.common_fea_map),
            len(self.fea_conf_obj.item_fea_map),
            len(self.fea_conf_obj.creative_fea_map),
        ]
        configured_counts = _kwargs.get('rm_bucket_token_counts')
        if configured_counts is None:
            configured_counts = self._allocate_bucket_tokens(self.rm_token_num, field_counts)
        self.rm_bucket_token_counts = [int(value) for value in configured_counts]

        if len(self.rm_bucket_token_counts) != len(self._BUCKET_NAMES):
            raise ValueError('rm_bucket_token_counts must contain [common, item, creative]')
        if sum(self.rm_bucket_token_counts) != self.rm_token_num:
            raise ValueError(
                'sum(rm_bucket_token_counts)={} must equal rm_token_num={}'.format(
                    sum(self.rm_bucket_token_counts), self.rm_token_num
                )
            )
        if any(token_count <= 0 for token_count in self.rm_bucket_token_counts):
            raise ValueError('each of the three buckets must receive at least one token')
        if any(token_count > field_count for token_count, field_count in
               zip(self.rm_bucket_token_counts, field_counts)):
            raise ValueError('a bucket cannot have more tokens than fields')
        if self.rm_head_num != self.rm_token_num:
            raise ValueError('RankMixer residual layout requires rm_head_num == rm_token_num')

        logging.info(
            'Semantic-Cross RankMixer v2: fields=%s, bucket_tokens=%s, T=%d, H=%d, '
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
    def _allocate_bucket_tokens(total_tokens, field_counts):
        """Allocate tokens proportionally while reserving one for every bucket."""
        if total_tokens < len(field_counts):
            raise ValueError("rm_token_num must be at least the number of buckets")
        if any(count <= 0 for count in field_counts):
            raise ValueError("all three feature buckets must be non-empty")

        counts = [1] * len(field_counts)
        remaining = total_tokens - len(field_counts)
        total_fields = float(sum(field_counts))
        quotas = [remaining * count / total_fields for count in field_counts]
        floors = [int(math.floor(quota)) for quota in quotas]
        counts = [count + floor for count, floor in zip(counts, floors)]

        leftover = total_tokens - sum(counts)
        order = sorted(
            range(len(field_counts)),
            key=lambda idx: (quotas[idx] - floors[idx], field_counts[idx]),
            reverse=True,
        )
        for idx in order[:leftover]:
            counts[idx] += 1
        return counts

    @staticmethod
    def _balanced_group_sizes(field_count, token_count):
        """Split whole fields into balanced, contiguous token groups."""
        base = field_count // token_count
        remainder = field_count % token_count
        return [base + (1 if idx < remainder else 0) for idx in range(token_count)]

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

    def _project_semantic_group(self, field_tensors, bucket_name, token_idx, export):
        token_input = field_tensors[0] if len(field_tensors) == 1 else tf.concat(field_tensors, axis=-1)
        input_dim = token_input.shape[-1].value
        if input_dim is None:
            raise ValueError("semantic token input dimension must be statically known")

        if self.rm_token_proj_act in (None, "identity", "linear"):
            activation_fn = tf.identity
        else:
            activation_fn = self.get_act_func(self.rm_token_proj_act)

        with tf.variable_scope(
            "{}_token_{}".format(bucket_name, token_idx),
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
            "semantic token %s/%d: fields=%d, input_dim=%d -> D=%d",
            bucket_name,
            token_idx,
            len(field_tensors),
            input_dim,
            self.rm_hidden_dim,
        )
        return projected

    def _semantic_tokenize(self, bucket_fields, export):
        tokens = []
        with tf.variable_scope("rm_semantic_tokenize", reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            for bucket_name, field_tensors, token_count in zip(
                    self._BUCKET_NAMES, bucket_fields, self.rm_bucket_token_counts):
                group_sizes = self._balanced_group_sizes(len(field_tensors), token_count)
                logging.info("%s semantic field groups: %s", bucket_name, group_sizes)
                cursor = 0
                for token_idx, group_size in enumerate(group_sizes):
                    group = field_tensors[cursor: cursor + group_size]
                    cursor += group_size
                    tokens.append(
                        self._project_semantic_group(group, bucket_name, token_idx, export)
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

        bucket_embeddings = {name: [] for name in self._BUCKET_NAMES}
        for index, column in enumerate(self.features.lookup_nonseq_columns + self.features.seq_columns):
            feature_key = get_sparse_fc_key(column)
            if feature_key in self.fea_conf_obj.common_fea_map:
                bucket_embeddings["common"].append(sparse_embeddings[index])
            elif feature_key in self.fea_conf_obj.item_fea_map:
                bucket_embeddings["item"].append(sparse_embeddings[index])
            elif feature_key in self.fea_conf_obj.creative_fea_map:
                bucket_embeddings["creative"].append(sparse_embeddings[index])

        if any(not bucket_embeddings[name] for name in self._BUCKET_NAMES):
            raise ValueError("Semantic-Cross RankMixer requires non-empty common/item/creative buckets")

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

            bucket_fields = []
            for name in self._BUCKET_NAMES:
                field_dims = [tensor.shape[-1].value for tensor in bucket_embeddings[name]]
                if any(dim is None for dim in field_dims):
                    raise ValueError("all field embedding dimensions must be statically known")
                bucket_fields.append(tf.split(bucket_tensors[name], field_dims, axis=-1))

            input_tokens = self._semantic_tokenize(bucket_fields, export)
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
