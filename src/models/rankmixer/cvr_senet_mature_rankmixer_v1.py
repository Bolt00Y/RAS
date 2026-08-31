# -*- coding: utf-8 -*-
"""Pure mature RankMixer for Base three-bucket single-head fst_CVR.

This is the production-callable implementation of the frozen D=256 design:

* Base-only sparse inputs: common_user(385), item(835), creative(14), E=17.
* Mature low-rank excitation2 SENet: user=256, item=128, creative=128.
* 31 local tokens + 1 user/item global token, T=32, D=256.
* Three mature mix_up + per-token SwiGLU blocks, expansion=3.5 (M=896).
* Mean pooling plus a proportionally scaled creative bypass of 32 dimensions.
* Mature compact task tower [256, 128] with Phalanx Robust SyncBN.
* One fst_cvr sigmoid/BCE only.

There is deliberately no sequence/gattr/dense-feature path, DIN, replay
correction, DCN-M/cross/shortcut, or auxiliary task head in this module.

The implementation targets the repository's existing Python 3 / TensorFlow 1.x
/ Flood runtime.  Common infrastructure (feature columns, Flood lookup/data,
metrics and optimizers) is imported from the project; all architecture-specific
RankMixer code lives in this file.
"""

import hashlib
import math
import os
from logging import FileHandler, Formatter, getLogger
from pydoc import locate

import logging
import numpy as np
import tensorflow as tf

import flood
from data.feature import FeatureColumnBuilder
from flood.python.data import data_util as flood_data_util
from flood.python.ops import parsing_ops
from flood.python.ops.auc import flood_auc
from flood.python.training.optimizer import FloodOptimizer
from flood.python.utils import lookup_utils
from utils import learning_rate as learning_rate_utils
from utils.accumulated_metrics import *
from utils.file_utils import mkdir_hdfs, upload_hdfs
from utils.odds import get_sparse_fc_key

from ..model_base import ModelBase


def _ids(value):
    """Build an immutable ordered feature-id tuple from a compact literal."""
    return tuple(value.split())


# These are the exact mature feature partitions.  They are embedded here so the
# model can be uploaded and called without any external reference directory on
# PYTHONPATH.
_USER_V1_IDS = _ids("""
1001 1006 1014 1034 1035 1036 866014 867603 1501 1502 1504 1505 1527 4500 4501 4502 4503 4504 16725
16727 16731 16733 16735 16737 16739 16743 16744 201702 201720 201756 201757 210015 210042 866250 866251
33866903 881203 881204 881206 881215 881402 881404 881663 881664 881665 881687 881691 881711 3001 3003
3004 3020 3102 3103 201939 300000 3006 3007 3008 3009 12209 12235 12402 12403 12438 15000 15002 21010
21012 21013 21030 21031 21032 21033 21055 21233 21238 21239 21240 21246 21257 21258 21260 21264 25136
25138 27516 794734 794768 794802 160034 24904004 304322 306045 310614 241125006 031090 340123 340121
340059 340063 340092
""")

_USER_V2_IDS = _ids("""
1121 1509 1512 1521 1524 2073 2100 2101 2102 2103 2104 2112 2123 2503 2504 2505 2506 2507 6900 10231
10232 10233 10522 10600 10601 13037 13038 13039 19013 19016 19024 20517 20518 20521 21749 21750 25000
25001 25002 25003 25006 25700 25702 25703 131480 200200 200214 200300 200302 200303 200304 200305 200306
200318 200319 200320 200762 200764 201704 202333 202334 202425 202426 211121 231056 231065 231383 231384
231484 600154 790220 790221 790222 790230 790249 790250 790251 794014 794015 794030 794031 794164 794178
794179 794200 794208 794209 794210 794214 794215 861818 866012 866013 866023 866024 866027 866029 866034
866041 866054 866061 866063 866064 866065 866066 866068 866069 866070 866071 866072 866073 866082 866103
867645 867648 868404 868405 868407 868413 868414 868427 870277 21303 21307 21340 21351 21355 21359 21402
21403 21404 21602 21610 21663 210000 210001 795602 4418192 4418196 340086 340125 340093 340037 340054
340109 340160 340001 110151 110153
""")

_USER_V3_IDS = _ids("""
1041 1042 1043 1063 1064 1065 1070 1104 1106 2066 2509 3014 3015 3016 6910 6911 6912 10442 18021 18073
18078 18083 18094 18098 18100 18105 18214 25044 25045 25046 25049 26017 26021 26025 26035 26107 200124
200413 200714 200715 200752 200758 201900 201905 201906 201909 201914 201915 201930 201931 201932 201937
202144 202218 202223 202330 215311 215312 215343 215373 215401 246003 33600031 860023 860031 860034
860037 860042 860045 862311 862355 862376 863014 863018 863024 863030 863044 863046 863141 863712 863729
865600 868023 870025 870038 870059 870069 870130 870311 870322 870324 881102 881104 7007737 7007741
7007746 7007755 24082402 24082411 24082412 24082413 795012 795014 881816 881817 881818 881820 881834
881842 882303 882304 882305 882306 2014601 2017702 204530 204543 4439006 2015703 2015709 2015745 2022401
300091 87560211 340453 340394 340364 340374 340367 340451 340377 110011 340455 340122
""")

_ITEM_V1_IDS = _ids("""
33203303 33866926 25741 204242 160043 810132 24702 204202 794204 27303 24082423 33203607 794206 864157
21202 201809 794205 280602 25711 200181 870313 33758666 201716 6894 794203 17136 33204162 8112 24711
870177 10216 863132 131468 4014 2015493 206206 25093 12092 882225 864132 10419 24531 864553 4007 600100
7007713 27316 33205227 201717 33204181 27321 860076 21052 160044 881705 860066 24710 794201 208013 4061
10310 310586 5014 203708 33203312 780011 200727 863286 33205186 861504 12104 4017 794212 864386 864410
870402 200316 10359 33203311 246007 17071 620000 33868978 37617 500136 24541 10016 864743 882401 621414
881353 17033 6007 12110 131470 206201 12118 27308 870279 4418073 7007710 215393 33204180 865349 28017
19044 12157 6016 16746 33203302 25515 13020 6047 17027 881309 881717 882385 20505 10219 25504 881757
794171 21051 10022 870264 12206 27311 500120 131049 33868929 22102 200284 310602 500137 6001 865421
865416 10207 12120 900017 131483 10407 200106 864219 500103 870303 27631 231494 27616 881709 310604
17137 864744 865726 865618 7007715 882233 600001 17135 870195 867665 794170 201856 12137 24082404 18088
17062 810107 12138 14237 909116 302552 302987 87560205 302185 87560133 1602601 160070 341102 900086
302595 1602631 6052 310588 304456 37615 341103 341105 304451 341104 304946 302190 304452 304913 304911
241125018 241215027 241215101 247030061 340116 3400731 3401321
""")

_ITEM_V2_IDS = _ids("""
10002 864578 208012 621872 201912 24708 33203330 10413 621878 6888 13009 882419 206056 202096 861201
33868969 882353 864770 2115 13021 600102 21114 215350 33868954 7806 24701 33868943 33203332 6802
33203306 5410 280501 7007714 37616 24116 24709 160067 33866912 25008 500015 25506 6134 12111 241215011
131474 882371 24246 863009 206051 131473 10387 208034 201021 863802 241215001 20501 21702 7002 25752
200754 870310 17177 33866925 28023 870008 206301 12205 200615 12155 13022 25120 8502 206157 33868965
33205180 622496 7502 621856 200325 19041 2111 6131 17178 10210 867685 865344 10007 21201 302342 500158
621412 12134 21668 882369 17088 794213 304393 600024 10003 276351 794165 206029 12117 200104 794211
10013 19042 863060 200107 33868950 33868976 870270 231344 863069 882326 25501 21726 860090 10012 870263
862844 24705 160033 21034 881721 200210 4418101 22119 870012 241215038 1110 215337 208011 241215065
10410 870357 861124 131475 25073 13005 882227 12094 10152 131479 160049 18035 131052 881226 868030
206585 6892 882235 7007716 140700 21050 865342 3044501 200729 18501 4418001 24242 6224 12204 881284 6013
28013 862616 881634 200317 25113 3401371 3401661 340100 340028 340044 3400141 340296 340317 340321
3403491 340335 160077 1600912 2022444 302533 87560214 340483 340856 340827 340824 340756 340761 820000
820001 820003 820004 820007 820025 820027 820028 820029 820035 820061
""")

_ITEM_V3_IDS = _ids("""
21035 200756 304952 21746 6206 881221 203742 20504 24707 21242 861540 200757 33203310 6011 500159 310601
87560220 33204182 33868952 19035 13002 201910 246005 10388 131484 500000 881025 33203301 33204185 12100
33868977 25014 868500 870315 861060 794023 6914 870340 870373 17139 131478 863133 4012 621877 208000
500003 24703 24498 33203321 206310 241215127 25717 794207 21762 870283 6859 304395 25027 12112 881108
302554 881331 10018 3029611 909043 500151 10068 881681 24082417 10010 21728 33203308 794202 200640 24530
881265 206389 881220 863210 7007711 868036 870128 200269 33203304 131476 19047 33866919 18504 864774
10524 200314 600112 10062 12122 215399 231334 865711 863062 13006 12113 206510 12115 10021 21110 600101
33866915 28003 10059 794021 864738 304394 131048 622530 12101 810109 868486 201825 868291 6852 280611
900647 500150 24706 25721 1111 11006 863054 200283 865420 24108 12119 33866914 2022429 12140 200780
794022 863780 794007 500121 861612 12088 13010 10160 865118 201918 18010 24237 302302 33203331 200768
208014 868513 881733 21752 33868961 21053 24496 881267 200313 302503 200753 6008 17086 622316 280502
20500 60119 21669 24218 863087 212402 212422 212432 212502 208001 341265 341320 341353 341358 341421
341888 304383 3402761 208016 247031681 900643 340096 2015723 340859 340076 340070 302374 500300 500301
500302 600200 600202 600201 600255 600233 600253 600254
""")

_ITEM_V4_PLUS_IDS = _ids("""
37618 10014 131466 24808118 201911 200751 17053 880448 25106 33203334 622533 201705 6870 33204196 865275
10008 231333 110041 869300 208015 867689 6804 10213 24704 206550 6046 25116 18007 7809 18197 201916
208030 8501 25048 17058 6012 21743 7501 862391 861219 622555 4009 200105 861213 200324 21760 18503
206563 621415 7001 24021 6041 870001 867638 863808 200585 17194 246004 33203333 600022 211100 246014
211130 17111 131482 310585 160063 24121 24497 33204187 864215 87580093 6133 24231 21036 870257 882223
33204186 21054 865682 302502 131467 21037 500135 26003 131472 863811 24115 5019 21708 27525 6893 868029
33868953 6811 27533 621416 87560127 200310 131485 6871 21115 7007708 10520 17107 33868970 33868973 25012
6004 794169 882354 274471 25010 160065 200406 863056 215334 21729 27507 33866909 25060 200315 4003
870250 28060 25754 246006 500001 27632 25059 25015 200765 25011 201735 24107 870166 865093 621842 862388
28019 231374 200311 26007 794005 861534 810103 18004 882417 6021 882416 25751 203797 881237 1086 865341
10154 863047 500134 33203320 10020 600249 212611 770583 770592 770626 770521 770607 770584 770591 770590
770627 770560 770587 770570 770568 770630 770571 770588 770459 7704581 770460 770471 770468 770470
770461 770462 770472 770473 7704561 770467 770469 770656 770657 10528 140707 16726 16728 16742 16759
20512 206077 206081 206082 22101 22106 22120 22129 22131 24328 24330 24332 27102 27367 27402 27443 27445
27447 27459 27606 27626 27634 27635 27640 5001 6501 33795609 33795608 33795610
""")

_CREATIVE_IDS = _ids("""
780110 780111 780112 780113 780117 8001 8002 8003 8007 8310 500157 900137 8203 8207
""")


class MLPModel(ModelBase):
    """Base-three-bucket, pure mature RankMixer D=256 model."""

    _EXPECTED_FIELD_COUNTS = (385, 835, 14)
    _EXPECTED_DENSE_TRAINABLE_PARAMS = 109976671
    _GROUP_VERSION = 'pure_mature_rankmixer_base_3bucket_d256_v1'
    _GROUP_CHECKSUMS = {
        'user_v1': 'bf6fed778c4286bd139cc89038be3cda1d1ff5160bbb87074c4f7eeb9773ae82',
        'user_v2': '3740fe428c796e2841d16abe9e66b5be6ff9b51feae37a97f03fd02504054636',
        'user_v3': 'a5e4c9cdfb67d2a97decf411cfcb174ca919db7518139304f1ae2397f6ce997f',
        'item_v1': '00a2e5f67d270c2cbcb5d19303225148f1cba3ad3f5a417548ae382874790d45',
        'item_v2': '0feb0107e4a98e4039fdb21185424c35063d1635a6f39fc6f20d9597719f183d',
        'item_v3': '7f75c1fe98a1614cd3a09bd3ad5af47a702f155a80af57bd92eea250418037da',
        'item_v4_plus': 'fb48c17fe1b2b3e0ed9630b3164c18706c8a01d15d7008ed0b64d11ce1527830',
        'creative': 'eb010d042c3750ca4739847484fff75be5307ed4c2c29d94cc17f9f956a56702',
    }
    _ALL_GROUPS_CHECKSUM = '187d557afabac51240f98663eba2f39f130be1261412f182f9481c6aad5fb6cc'
    _USER_GROUPS = (
        ('user_v1', _USER_V1_IDS, 3),
        ('user_v2', _USER_V2_IDS, 3),
        ('user_v3', _USER_V3_IDS, 4),
    )
    _ITEM_GROUPS = (
        ('item_v1', _ITEM_V1_IDS, 5),
        ('item_v2', _ITEM_V2_IDS, 5),
        ('item_v3', _ITEM_V3_IDS, 5),
        ('item_v4_plus', _ITEM_V4_PLUS_IDS, 6),
    )

    def __init__(self, **_kwargs):
        for key, value in _kwargs.items():
            setattr(self, key, value)

        # Runtime / framework configuration.
        self.batch_size = int(_kwargs.get('batch_size', 2048))
        self.eval_batch_size = int(_kwargs.get('eval_batch_size', 2048))
        self.l2_deep = float(_kwargs.get('l2_deep', 0.000001))
        self.grad_clip_value = float(_kwargs.get('grad_clip_value', 15))
        self.max_partitions = _kwargs.get('max_partitions', None)
        self.embedding_size = int(_kwargs.get('embedding_size', 17))
        self.log_nn_vars = _kwargs.get('log_nn_vars', False)

        self.tf_config = _kwargs.get('tf_config', None)
        self.worker_id = self.tf_config['task']['index'] if self.tf_config else 0
        self.is_chief = self.worker_id == 0
        self.task_index = self.worker_id

        self.model_dir = _kwargs.get('model_dir', None)
        self.predict_path = _kwargs.get('predict_path', None)
        self.timeout = int(_kwargs.get('timeout', 60 * 20) * 1000)
        self.upload_log = _kwargs.get('upload_log', False)
        self.save_predict_result = _kwargs.get('save_predict_result', False)
        self.ps_stage = _kwargs.get('ps_stage', 'update')

        # Normalization and optimization follow the mature/Base boundary.
        self.enable_phalanx = _kwargs.get('enable_phalanx', True)
        self.batch_norm = _kwargs.get('batch_norm', True)
        self.batch_norm_decay = float(_kwargs.get('batch_norm_decay', 0.9))
        self.use_riemann_bn = _kwargs.get('use_riemann_bn', True)
        self.embed_use_renorm = _kwargs.get('embed_use_renorm', False)
        self.embed_renorm_decay = float(_kwargs.get('embed_renorm_decay', 0.99))
        self.use_senet = _kwargs.get('use_senet', True)
        self.use_senet_bn = _kwargs.get('use_senet_bn', True)
        self.senet_act_type = _kwargs.get('senet_act_type', 'sigmoid')
        self.mlp_act_type = _kwargs.get('mlp_act_type', 'gelu_2')
        self.clip_val = float(_kwargs.get('clip_val', 50))

        self.optimizer = _kwargs.get('optimizer', 'flood_adam')
        self.learning_rate = _kwargs.get('learning_rate', 0.00001)
        self.decay = _kwargs.get('decay', '')
        self.schedule_config = _kwargs.get(
            'schedule_config',
            {'type': 'gauss_decay', 'warmup_steps': 60000,
             'decay_steps': 40000, 'min_rate': 0.5},
        )

        # Frozen architecture.  Parameter names mirror the mature model args.
        self.mixup_token_num = int(_kwargs.get('mixup_token_num', 32))
        self.mixup_token_dim = int(_kwargs.get('mixup_token_dim', 256))
        self.mlp_mixer_layers = int(_kwargs.get('mlp_mixer_layers', 3))
        self.mixer_expand_ratio = float(_kwargs.get('mixer_expand_ratio', 3.5))
        self.mixer_hidden_dim = int(self.mixup_token_dim * self.mixer_expand_ratio)
        self.global_token_hidden_dim = int(_kwargs.get('global_token_hidden_dim', 512))
        self.creative_output_dim = int(_kwargs.get('creative_output_dim', 32))
        self.creative_hidden_dim = int(_kwargs.get('creative_hidden_dim', 256))
        self.cvr_layers = [int(value) for value in _kwargs.get('cvr_layers', [256, 128])]
        self.cvr_label_name = _kwargs.get('cvr_label_name', 'fst_cvr_label')
        self.opt_goal = _kwargs.get('opt_goal', 'first_cvr')
        self.export_name = _kwargs.get('export_name', 'first_cvr')

        # Base-only feature configuration.
        self.feature_version = _kwargs.get('feature_version', 'data.cvr.cvr_fea_v10_base_cold')
        self.feature_version_old = _kwargs.get('feature_version_old', self.feature_version)
        module = locate(self.feature_version)
        module_old = locate(self.feature_version_old)
        if module is None or module_old is None:
            raise ValueError('invalid feature_version: {} / {}'.format(
                self.feature_version, self.feature_version_old))
        self.fea_conf_obj = module.FeatureConfig()
        self.fea_conf_obj_old = module_old.FeatureConfig()
        self.features = FeatureColumnBuilder(
            feature_config=self.fea_conf_obj,
            default_embedding_size=self.embedding_size,
        )
        self.features_old = FeatureColumnBuilder(
            feature_config=self.fea_conf_obj_old,
            default_embedding_size=self.embedding_size,
        )

        # Data path remains the Base path; no replay-specific correction exists.
        self.epochs = _kwargs.get('epochs', None)
        self.prefetch_num = int(_kwargs.get('prefetch_num', 100))
        self.interleave = int(_kwargs.get('interleave', 8))
        self.test_interleave = int(_kwargs.get('test_interleave', 8))
        self.sampler_stat = _kwargs.get('sampler_stat', False)
        self.async_pull = _kwargs.get('async_pull', False)
        self.test_async_pull = _kwargs.get('test_async_pull', True)
        self.max_prefetched_pull = int(_kwargs.get('max_prefetched_pull', -1))
        self.test_batch_num = int(_kwargs.get('test_batch_num', 4000 * 10000))
        self.drop_last_files = int(_kwargs.get('drop_last_files', 2))
        self.slow_worker_timeout = int(_kwargs.get('slow_worker_timeout', 3600000))
        self.slow_worker_num_limit = int(_kwargs.get('slow_worker_num_limit', 0))
        self.sampler_label_name = _kwargs.get('sampler_label_name', '')
        self.sampler_positive_rate = float(_kwargs.get('sampler_positive_rate', 1.0))
        self.sampler_negative_rate = float(_kwargs.get('sampler_negative_rate', 1.0))
        self.filter_pass_empty = _kwargs.get('filter_pass_empty', True)

        self.strict_test_date = _kwargs.get('strict_test_date', False)
        self.order_by_date = _kwargs.get('order_by_date', False)
        self.random_feature = _kwargs.get('random_feature', None)
        self.parallel_feature_analysis = _kwargs.get('parallel_feature_analysis', False)
        self.train_reset_interval = int(_kwargs.get('train_reset_interval', 10000))
        self.train_reset_count = 0
        self.train_count = 0
        self.eval_count = 0
        self.num_ps = len(self.tf_config['cluster']['ps']) if self.tf_config else 1
        self.num_worker = len(self.tf_config['cluster']['worker']) if self.tf_config else 1
        self.fq_table_config = _kwargs.get('fq_table_config', 'shrink_only_config')
        self.dir2_all_tensor = _kwargs.get('dir2_all_tensor', 'None')
        self.second_epoch_ckpt_import_dir = _kwargs.get('second_epoch_ckpt_import_dir', '')
        self.enable_dense_warmup = _kwargs.get('enable_dense_warmup', False)

        self._validate_architecture_args(_kwargs)
        self._validate_feature_contract()
        self.rm_parameter_breakdown = self._calculate_dense_trainable_params()
        if self.rm_parameter_breakdown['total'] != self._EXPECTED_DENSE_TRAINABLE_PARAMS:
            raise ValueError(
                'dense parameter ledger mismatch: actual={}, expected={}'.format(
                    self.rm_parameter_breakdown['total'],
                    self._EXPECTED_DENSE_TRAINABLE_PARAMS,
                )
            )

        logging.info(
            'Pure mature RankMixer D256: fields=%s, tokens=31+1, T=%d, D=%d, '
            'L=%d, M=%d, creative=%d, head=%s, dense_params=%d',
            self._EXPECTED_FIELD_COUNTS,
            self.mixup_token_num,
            self.mixup_token_dim,
            self.mlp_mixer_layers,
            self.mixer_hidden_dim,
            self.creative_output_dim,
            self.cvr_layers,
            self.rm_parameter_breakdown['total'],
        )

        if _kwargs.get('log_gflags', True) and self.random_feature is None:
            self.list_all_member()

        super().__init__()

    def _validate_architecture_args(self, kwargs):
        required = {
            'embedding_size': (self.embedding_size, 17),
            'mixup_token_num': (self.mixup_token_num, 32),
            'mixup_token_dim': (self.mixup_token_dim, 256),
            'mlp_mixer_layers': (self.mlp_mixer_layers, 3),
            'mixer_hidden_dim': (self.mixer_hidden_dim, 896),
            'global_token_hidden_dim': (self.global_token_hidden_dim, 512),
            'creative_output_dim': (self.creative_output_dim, 32),
            'creative_hidden_dim': (self.creative_hidden_dim, 256),
        }
        for name, pair in required.items():
            actual, expected = pair
            if actual != expected:
                raise ValueError('{} must be {}, got {}'.format(name, expected, actual))
        if self.cvr_layers != [256, 128]:
            raise ValueError('cvr_layers must be [256, 128]')
        if self.mixup_token_dim % self.mixup_token_num != 0:
            raise ValueError('mixup_token_dim must be divisible by mixup_token_num')
        if not self.use_senet or not self.use_senet_bn or not self.batch_norm:
            raise ValueError('use_senet/use_senet_bn/batch_norm must all be true')
        if not self.enable_phalanx:
            raise ValueError('the frozen D256 design requires enable_phalanx=true')
        if self.senet_act_type != 'sigmoid' or self.mlp_act_type != 'gelu_2':
            raise ValueError('senet_act_type=\'sigmoid\' and mlp_act_type=\'gelu_2\' are required')
        if self.enable_dense_warmup:
            raise ValueError('the frozen D256 design requires dense cold start')

        forbidden_flags = (
            'enable_rpy_neg_sampler', 'enable_last_cvr', 'enable_wide_cvr',
            'enable_delay_train_mode', 'enable_mlt_loss',
            'enable_aux_distill_head', 'enable_distill_loss',
        )
        enabled = [name for name in forbidden_flags if bool(kwargs.get(name, False))]
        if enabled:
            raise ValueError('unsupported extra paths enabled: {}'.format(enabled))
        if self.sampler_label_name or self.sampler_positive_rate != 1.0 \
                or self.sampler_negative_rate != 1.0:
            raise ValueError('re-sampling is forbidden; use Base samples unchanged')

    def _validate_feature_contract(self):
        unsupported = {
            'coupon': getattr(self.fea_conf_obj, 'coupon_fea_map', {}),
            'dense': getattr(self.fea_conf_obj, 'dense_fea_map', {}),
            'sequence': getattr(self.fea_conf_obj, 'seq_fea_map', {}),
            'gattr': getattr(self.fea_conf_obj, 'gattr_fea_map', {}),
            'din': getattr(self.fea_conf_obj, 'din_fea_map', {}),
        }
        nonempty = {name: len(value) for name, value in unsupported.items() if value}
        if nonempty:
            raise ValueError('only common/item/creative are allowed: {}'.format(nonempty))

        expected_sets = {
            'common': set(self.fea_conf_obj.common_fea_map.keys()),
            'item': set(self.fea_conf_obj.item_fea_map.keys()),
            'creative': set(self.fea_conf_obj.creative_fea_map.keys()),
        }
        group_sets = {
            'common': set(_USER_V1_IDS + _USER_V2_IDS + _USER_V3_IDS),
            'item': set(_ITEM_V1_IDS + _ITEM_V2_IDS + _ITEM_V3_IDS + _ITEM_V4_PLUS_IDS),
            'creative': set(_CREATIVE_IDS),
        }
        field_counts = tuple(len(expected_sets[name]) for name in ('common', 'item', 'creative'))
        if field_counts != self._EXPECTED_FIELD_COUNTS:
            raise ValueError('Base field counts must be {}, got {}'.format(
                self._EXPECTED_FIELD_COUNTS, field_counts))
        for bucket_name in ('common', 'item', 'creative'):
            if expected_sets[bucket_name] != group_sets[bucket_name]:
                missing = expected_sets[bucket_name] - group_sets[bucket_name]
                unknown = group_sets[bucket_name] - expected_sets[bucket_name]
                raise ValueError('{} routing mismatch: missing={}, unknown={}'.format(
                    bucket_name, sorted(missing), sorted(unknown)))

        ordered_groups = list(self._USER_GROUPS) + list(self._ITEM_GROUPS)
        all_ids = []
        for group_name, feature_ids, _ in ordered_groups:
            if len(feature_ids) != len(set(feature_ids)):
                raise ValueError('{} contains duplicate IDs'.format(group_name))
            checksum = hashlib.sha256('\n'.join(feature_ids).encode('utf-8')).hexdigest()
            if checksum != self._GROUP_CHECKSUMS[group_name]:
                raise ValueError('{} checksum mismatch: {}'.format(group_name, checksum))
            all_ids.extend(feature_ids)
        creative_checksum = hashlib.sha256('\n'.join(_CREATIVE_IDS).encode('utf-8')).hexdigest()
        if creative_checksum != self._GROUP_CHECKSUMS['creative']:
            raise ValueError('creative checksum mismatch: {}'.format(creative_checksum))
        all_ids.extend(_CREATIVE_IDS)
        if len(all_ids) != 1234 or len(set(all_ids)) != 1234:
            raise ValueError('internal routing must cover 1234 unique fields')
        all_checksum = hashlib.sha256('\n'.join(all_ids).encode('utf-8')).hexdigest()
        if all_checksum != self._ALL_GROUPS_CHECKSUM:
            raise ValueError('complete routing checksum mismatch: {}'.format(all_checksum))

        local_tokens = sum(token_count for _, _, token_count in ordered_groups)
        if local_tokens != 31 or local_tokens + 1 != self.mixup_token_num:
            raise ValueError('token allocation must be 31 local + 1 global')

    def _calculate_dense_trainable_params(self):
        user_width = 385 * self.embedding_size
        item_width = 835 * self.embedding_size
        creative_width = 14 * self.embedding_size
        token_dim = self.mixup_token_dim
        token_num = self.mixup_token_num
        mixer_hidden = self.mixer_hidden_dim

        input_bn = 2 * (user_width + item_width + creative_width)
        senet = (
            user_width * 256 + 256 + 2 * 256 + 256 * user_width + user_width
            + (user_width + item_width) * 128 + 128 + 2 * 128
            + 128 * item_width + item_width
            + creative_width * 128 + 128 + 2 * 128
            + 128 * creative_width + creative_width
        )
        local_tokens = 0
        for _, feature_ids, token_count in self._USER_GROUPS + self._ITEM_GROUPS:
            output_width = token_count * token_dim
            local_tokens += len(feature_ids) * self.embedding_size * output_width
            local_tokens += 3 * output_width  # dense bias + BN gamma/beta

        global_input = user_width + item_width
        global_token = (
            2 * global_input
            + global_input * self.global_token_hidden_dim + self.global_token_hidden_dim
            + self.global_token_hidden_dim * token_dim + token_dim
            + 2 * token_dim
        )
        per_mixer_layer = (
            2 * (token_num * token_dim * mixer_hidden + token_num * mixer_hidden)
            + token_num * mixer_hidden * token_dim + token_num * token_dim
            + 2 * token_dim + mixer_hidden + token_dim
        )
        mixer = self.mlp_mixer_layers * per_mixer_layer + 2 * token_dim

        creative = (
            creative_width * self.creative_hidden_dim + self.creative_hidden_dim
            + 2 * self.creative_hidden_dim + self.creative_hidden_dim
            + self.creative_hidden_dim * self.creative_output_dim + self.creative_output_dim
            + 2 * self.creative_output_dim + self.creative_output_dim
        )
        head_input = token_dim + self.creative_output_dim
        task_head = 0
        for layer_size in self.cvr_layers:
            task_head += head_input * layer_size + layer_size + 2 * layer_size
            head_input = layer_size
        task_head += head_input + 1

        result = {
            'input_bn': input_bn,
            'senet': senet,
            'local_tokens': local_tokens,
            'global_token': global_token,
            'mixer': mixer,
            'creative_bypass': creative,
            'task_head': task_head,
        }
        result['total'] = sum(result.values())
        return result

    def _verify_graph_dense_trainable_params(self, dense_scope):
        """Verify actual graph variables, including partitioned variable shards."""
        dense_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=dense_scope,
        )
        actual_total = 0
        log_manifest = not getattr(self, '_rm_logged_dense_manifest', False)
        for variable in dense_variables:
            shape = variable.shape.as_list()
            if any(dimension is None for dimension in shape):
                raise ValueError('unknown dense variable shape: {} {}'.format(
                    variable.op.name, shape))
            variable_total = 1
            for dimension in shape:
                variable_total *= dimension
            actual_total += variable_total
            if log_manifest:
                logging.info('D256 dense variable: %s shape=%s params=%d',
                             variable.op.name, shape, variable_total)
        if actual_total != self._EXPECTED_DENSE_TRAINABLE_PARAMS:
            raise ValueError('graph dense params={}, expected={}'.format(
                actual_total, self._EXPECTED_DENSE_TRAINABLE_PARAMS))
        self._rm_logged_dense_manifest = True
        return actual_total

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @classmethod
    def get_features_conf(cls, **kwargs):
        features_conf = {}
        feature_version = kwargs.get(
            'feature_version', 'data.cvr.cvr_fea_v10_base_cold')
        module = locate(feature_version)
        if module is None:
            raise ValueError('invalid feature_version: {}'.format(feature_version))
        fea_conf_obj = module.FeatureConfig()
        embedding_size = int(kwargs.get('embedding_size', 17))

        for key, value_map in fea_conf_obj.feature_details.items():
            if bool(int(value_map.get('model_ignore', 0))):
                logging.info('feature %s will not save', key)
                continue
            if value_map.get('fea_class', 'common') in ('dense', 'label', 'extra'):
                logging.info('skip feature %s', key)
                continue
            conf = {
                'embedding_size': int(value_map.get('embedding_size', embedding_size)),
                'pooling_type': value_map.get('pooling_type', 'SUM_POOLING'),
                'feature_parameter_args': {
                    'accessor': {
                        'stats_param': {
                            'constant_feature': bool(int(
                                value_map.get('constant_feature', 0)))
                        }
                    }
                },
            }
            stats_param = conf['feature_parameter_args']['accessor']['stats_param']
            if 'delete_threshold' in value_map:
                stats_param['delete_threshold'] = value_map['delete_threshold']
            if 'create_nonclk_prob' in value_map:
                stats_param['create_nonclk_prob'] = value_map['create_nonclk_prob']
            if 'create_click_prob' in value_map:
                # Keep the established repository behavior for checkpoint config.
                stats_param['create_nonclk_prob'] = value_map['create_click_prob']
            features_conf[key] = conf
        logging.info('features_conf size=%d', len(features_conf))
        return features_conf

    @classmethod
    def get_share_embedding_conf(cls, **kwargs):
        feature_version = kwargs.get(
            'feature_version', 'data.cvr.cvr_fea_v10_base_cold')
        module = locate(feature_version)
        if module is None:
            raise ValueError('invalid feature_version: {}'.format(feature_version))
        return module.FeatureConfig().features_share_map

    def get_dataset(self, data_paths, mode, use_dynamic_file=True, take_batch_num=0):
        parquet_columns = self.features.parquet_reader_columns
        features_spec = tf.feature_column.make_parse_example_spec(parquet_columns)
        visible_features = self.fea_conf_obj.visible_fea_map.keys()
        return {
            'dataset': flood_data_util.get_parquet_data(
                features=features_spec,
                data_paths=data_paths,
                batch_size=self.batch_size if mode == 'train' else self.eval_batch_size,
                size_limits_map=self.fea_conf_obj.feature_size_limit_map,
                feature_name_map=self.fea_conf_obj.features_multi_map,
                sparse_features_to_tensor=list(visible_features),
                sampler_label_name=self.sampler_label_name,
                sampler_positive_rate=self.sampler_positive_rate,
                sampler_negative_rate=self.sampler_negative_rate,
                filter_pass_empty=self.filter_pass_empty,
                shuffle=(mode == 'train'),
                use_dynamic_files=use_dynamic_file if mode != 'predict' else False,
                take_batch_num=0 if mode == 'train' else take_batch_num,
                random_feature='' if mode == 'train' else self.random_feature,
                join_key_name='pk',
                epochs=1,
                prefetch_num=self.prefetch_num,
                sampler_stat=self.sampler_stat,
                drop_last_files=self.drop_last_files if mode == 'train' else 0,
                async_pull=self.async_pull,
                max_prefetched_pull=self.max_prefetched_pull,
                drop_remainder=(mode == 'train'),
                interleave=self.test_interleave if mode in ('test', 'predict') else self.interleave,
                slow_worker_timeout=self.slow_worker_timeout,
                slow_worker_num_limit=self.slow_worker_num_limit,
                range_size_limit=100 * 1024 * 1024,
                hole_size_limit=10 * 1024 * 1024,
            )
        }

    def build(self, input_paths, test_paths, mode='train', config=None,
              use_dynamic=True, **kwargs):
        self.global_step = tf.train.get_or_create_global_step()
        self.global_step_op = tf.assign_add(self.global_step, 1)
        for graph_mode in ('train', 'test'):
            logging.info('********** %s **********', graph_mode)
            data_paths = test_paths if graph_mode == 'test' else input_paths
            self.build_dataset_op(data_paths, mode=graph_mode, flood_mode=mode)
            self.build_pred_results_op(mode=graph_mode, flood_mode=mode)
            self.build_auc_copc_op(mode=graph_mode)
            if graph_mode == 'train':
                self.build_loss_op(mode=graph_mode)
                self.build_summary(mode=graph_mode)
                self.build_optimizer_op()
        self._build_export(config=config)
        self.run_metadata = tf.RunMetadata()
        self.run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE,
            timeout_in_ms=self.timeout,
        )
        self.timeout_options = tf.RunOptions(timeout_in_ms=self.timeout)

        if self.log_nn_vars:
            for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                logging.info('global variable: %s', variable)

    def build_dataset_op(self, data_paths, mode, flood_mode):
        if mode == 'train':
            use_dynamic_files = (flood_mode == 'train')
        else:
            use_dynamic_files = self.strict_test_date and self.order_by_date
        logging.info('flood_mode=%s, graph_mode=%s, use_dynamic_files=%s',
                     flood_mode, mode, use_dynamic_files)
        dataset_op = self.get_dataset(
            data_paths,
            mode,
            use_dynamic_file=use_dynamic_files,
            take_batch_num=self.test_batch_num if mode == 'test' else 0,
        )
        dataset = dataset_op['dataset'].map(self.parse_examples, num_parallel_calls=None)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_initializable_iterator()
        self['{}_iterator'.format(mode)] = iterator
        self['{}_init_op'.format(mode)] = iterator.initializer
        result = iterator.get_next()
        for key, value in result.items():
            self['{}_{}'.format(mode, key)] = value

    def parse_examples(self, *example_batch):
        columns = self.features.parquet_reader_columns
        features = parsing_ops.parse_parquet(
            example_batch,
            tf.feature_column.make_parse_example_spec(columns),
            reserved_keys=self.fea_conf_obj.visible_fea_map,
            unique=False,
            share_embedding_conf=self.fea_conf_obj.features_share_map,
            global_hash=False,
            psv2=True,
        )
        features['sampleid'] = flood.generate_sample_id(
            search_ids=features['search_id'].values,
            example_ids=features['example_ids'].values,
        )
        labels = tf.cast(features.pop(self.cvr_label_name), tf.float32)
        sample_id = tf.cast(features.pop('sampleid'), tf.float32)
        search_id = features['search_id'].values
        example_id = features['example_ids'].values
        return {
            'features': features,
            'labels': labels,
            'sampleid': sample_id,
            'search_id': search_id,
            'example_id': example_id,
        }

    def build_pred_results_op(self, mode, flood_mode=None):
        function_mode = mode if mode == 'test' else flood_mode
        results = self.model_fn(
            self['{}_features'.format(mode)],
            self['{}_labels'.format(mode)],
            mode=function_mode,
        )
        for key, value in results.items():
            self['{}_{}'.format(mode, key)] = value

    def build_loss_op(self, mode):
        labels = tf.reshape(self['{}_labels'.format(mode)], shape=[-1])
        predictions = tf.reshape(self['{}_pred'.format(mode)], shape=[-1])
        self.loss = tf.reduce_mean(tf.losses.log_loss(
            predictions=predictions,
            labels=labels,
        ))
        self.loss_first = self.loss
        self.labels_pos_cvr_count = tf.reduce_sum(labels)

    def build_auc_copc_op(self, mode):
        labels = self['{}_labels'.format(mode)]
        predictions = self['{}_pred'.format(mode)]
        self['{}_auc'.format(mode)] = flood_auc(
            labels,
            predictions,
            name='auc/cvr',
            num_thresholds=2000,
        )
        self['{}_copc'.format(mode)] = (
            tf.reduce_sum(predictions) / (tf.reduce_sum(labels) + 1e-8)
        )
        self['{}_auc_values'.format(mode)] = tf.get_collection(
            tf.GraphKeys.METRIC_VARIABLES,
            scope='auc',
        )
        self['{}_reset_auc_op'.format(mode)] = tf.variables_initializer(
            var_list=self['{}_auc_values'.format(mode)])
        self['{}_pred_mean'.format(mode)] = tf.reduce_mean(predictions)

    def build_summary(self, mode):
        summaries = [
            tf.summary.scalar('{}/loss'.format(mode), self.loss),
            tf.summary.scalar('{}/auc'.format(mode), self['{}_auc'.format(mode)]),
            tf.summary.scalar('{}/copc'.format(mode), self['{}_copc'.format(mode)]),
        ]
        self.eval_summary = tf.summary.merge(summaries, name='eval_summary')

    def build_optimizer_op(self):
        if 'circle_restart' in self.decay:
            self.learning_rate = tf.train.cosine_decay_restarts(
                learning_rate=self.learning_rate,
                global_step=tf.train.get_global_step(),
                first_decay_steps=800000,
                t_mul=2.0,
                m_mul=1.0,
                alpha=0.000005,
            )
        elif 'exp' in self.decay:
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.learning_rate,
                global_step=tf.train.get_global_step(),
                decay_steps=500000,
                decay_rate=0.98,
                staircase=False,
            )
        else:
            self._build_lr_schedule()

        optimizer = self.get_optimizer(self.optimizer, self.learning_rate)
        self.optimizer = FloodOptimizer(optimizer)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        for gradient, variable in grads_and_vars:
            logging.info('[gradient] %s %s', gradient, variable)
            if gradient is not None:
                tf.summary.histogram(
                    'train/{}/gradients'.format(variable.op.name),
                    gradient,
                )
        self.train_op = [self.optimizer.apply_gradients(
            grads_and_vars,
            global_step=tf.train.get_global_step(),
        )]

    def _build_lr_schedule(self):
        self.learning_rate = self._schedule_lr(
            self.learning_rate,
            self.schedule_config,
        )

    def _schedule_lr(self, learning_rate, schedule_config):
        learning_rate = tf.convert_to_tensor(learning_rate)
        if 'type' in schedule_config:
            logging.info('use learning-rate schedule: %s', schedule_config)
            learning_rate_utils.get_or_create_milestone_step_reset_op()
            learning_rate = learning_rate_utils.learning_rate_schedule(
                learning_rate,
                schedule_config['type'],
                **schedule_config
            )
        return learning_rate

    def get_optimizer(self, optimizer='Adagrad', learning_rate=0.001):
        optimizer = optimizer.strip()
        logging.info('use optimizer: %s', optimizer)
        if optimizer == 'Adam':
            return tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
            )
        if optimizer == 'flood_adam':
            from flood.python.training.adam_optimizer import AdamOptimizer
            return AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
            )
        if optimizer == 'Adagrad':
            return tf.train.AdagradOptimizer(
                learning_rate=learning_rate,
                initial_accumulator_value=1e-8,
            )
        if optimizer == 'Momentum':
            return tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=0.95,
            )
        if optimizer == 'ftrl':
            return tf.train.FtrlOptimizer(learning_rate)
        if optimizer == 'lazyAdam':
            return tf.contrib.opt.LazyAdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
            )
        if optimizer == 'SGD':
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        raise ValueError('unsupported optimizer: {}'.format(optimizer))

    def train(self, session, worker_id=0, **kwargs):
        self.train_count += 1
        fetches = {
            'train_op': self.train_op,
            'loss': self.loss,
            'labels_pos_cvr_count': self.labels_pos_cvr_count,
            'global_step': self.global_step,
            'pred_mean': self['train_pred_mean'],
            'auc': self['train_auc'],
            'copc': self['train_copc'],
            'learning_rate': self.learning_rate,
        }
        result = session.run(fetches, options=self.timeout_options)
        if self.train_count % kwargs.get('train_log_step', 10) == 0:
            logging.info('---------------- train [%d] ----------------', self.train_count)
            logging.info(
                'gstep=%s loss=%.6f auc=%.6f copc=%.6f pred_mean=%.6f '
                'positive=%s lr=%s',
                result['global_step'],
                result['loss'],
                result['auc'],
                result['copc'],
                result['pred_mean'],
                result['labels_pos_cvr_count'],
                result['learning_rate'],
            )
        if self.task_index == 0 and self.train_reset_interval > 0 \
                and self.train_count * self.num_worker \
                > self.train_reset_interval * self.train_reset_count:
            self.train_reset_count += 1
            logging.info('reset train AUC')
            session.run([self['train_reset_auc_op']])
        return {
            'global_step': result['global_step'],
            'train_reset_count': self.train_reset_count,
        }

    def test(self, session, worker_id=0, prefix='test', **kwargs):
        self.train_init(session)
        log_format = '%(asctime)-15s [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s'
        file_handler = FileHandler('flood_worker_0.log')
        file_handler.setFormatter(Formatter(log_format))
        getLogger(name='search_jarvis_logging').addHandler(file_handler)

        test_count = 0
        session.run([self['test_init_op']])
        auc_accum = RocAucAccum(num_thresholds=2000)
        pr_auc_accum = PrAucAccum(num_thresholds=2000)
        copc_accum = COPCAccum()
        bucket_error = BucketErrorAccum(1000)
        sample_count_accum = SampleCntAccum()
        fetches = {
            'sampleid': self['test_sampleid'],
            'search_id': self['test_search_id'],
            'example_id': self['test_example_id'],
            'labels': self['test_labels'],
            'pred': self['test_pred'],
            'auc': self['test_auc'],
            'copc': self['test_copc'],
        }

        local_path = None
        hdfs_path = None
        if self.save_predict_result:
            local_path = 'predictions-{}.txt'.format(worker_id)
            hdfs_dir = os.path.join(
                self.predict_path if self.predict_path else self.model_dir,
                prefix,
            )
            hdfs_path = os.path.join(hdfs_dir, local_path)
            if worker_id == 0:
                mkdir_hdfs(hdfs_dir)
            with tf.gfile.Open(local_path, 'w') as output_file:
                output_file.write('')

        while True:
            try:
                result = session.run(fetches, options=self.timeout_options)
                if self.save_predict_result:
                    with tf.gfile.Open(local_path, 'a') as output_file:
                        for search_id, example_id, label, prediction in zip(
                                result['search_id'], result['example_id'],
                                result['labels'], result['pred']):
                            line = '\t'.join([
                                search_id.decode(),
                                example_id.decode(),
                                str(label[0]),
                                str(prediction),
                            ]) + '\n'
                            output_file.write(line)

                labels = result['labels']
                predictions = result['pred']
                test_count += 1
                auc_accum.update(labels, predictions)
                pr_auc_accum.update(labels, predictions)
                copc_accum.update(labels, predictions)
                bucket_error.update(labels, predictions)
                sample_count_accum.update(labels, predictions)

                if 0 < self.test_batch_num < test_count:
                    logging.info('finish test at test_batch_num=%d', self.test_batch_num)
                    break
                if test_count % kwargs.get('test_log_step', 10) == 0:
                    logging.info('test_count=%d auc=%.6f copc=%.6f',
                                 test_count, result['auc'], result['copc'])
            except tf.errors.OutOfRangeError as error:
                logging.info('all test data consumed: %s', str(error))
                break
            except tf.errors.DeadlineExceededError as error:
                logging.error('test step timed out: %s', str(error))
                break
            except tf.errors.InvalidArgumentError as error:
                logging.warning('skip invalid test data: %s', str(error))
                continue
            except (tf.errors.PermissionDeniedError,
                    tf.errors.FailedPreconditionError,
                    RuntimeError) as error:
                logging.error('test aborted: %s', str(error))
                break

        accumulated_metrics = {
            'cvr-tower': {
                'roc_auc': auc_accum.dump(),
                'copc': copc_accum.dump(),
                'pr_auc': pr_auc_accum.dump(),
                'bucket_error': bucket_error.dump(),
                'sample_cnt': sample_count_accum.dump(),
            }
        }
        result = {
            'accum_metrics': accumulated_metrics,
            'title': 'lamb-feature-{}'.format(self.random_feature)
            if self.random_feature else 'base',
        }
        if self.save_predict_result:
            upload_hdfs(local_path, hdfs_path, True)
            if self.upload_log and worker_id == 0:
                upload_hdfs(
                    'flood_worker_0.log',
                    os.path.join(os.path.dirname(hdfs_path), 'flood_worker_0.log'),
                    True,
                )
        return result

    def predict(self, session, worker_id=0, **kwargs):
        prefix = 'predict-{}'.format(self.random_feature) \
            if self.random_feature else 'predict'
        result = self.test(session, worker_id, prefix=prefix, **kwargs)
        if self.random_feature:
            result['merge_from_all_workers'] = not self.parallel_feature_analysis
        return result

    def _build_export(self, config=None):
        serialized_example = tf.placeholder(
            dtype=tf.string,
            shape=[None],
            name='example',
        )
        features = tf.parse_example(
            serialized_example,
            tf.feature_column.make_parse_example_spec(self.features.export_columns),
        )
        fake_labels = tf.constant(value=[[1]], shape=[1, 1], dtype=tf.float32)
        prediction_result = self.model_fn(
            features,
            fake_labels,
            mode='export',
            export=True,
        )
        self.export_spec = {
            'input': {'example': serialized_example},
            'output': {'cvr': prediction_result['pred']},
        }

    def export(self):
        return self.export_spec

    def train_init(self, session):
        logging.info('reinitialize train dataset')
        session.run(self['train_init_op'])
        if self.is_chief:
            session.run(learning_rate_utils.get_or_create_milestone_step_reset_op())
            logging.info(
                'milestone step: %s',
                session.run(learning_rate_utils.get_or_create_milestone_step()),
            )

    def evaluate(self, session, **kwargs):
        self.eval_count += 1
        try:
            result = session.run(
                {
                    'summary': self['eval_summary'],
                    'global_step': self.global_step,
                },
                options=tf.RunOptions(timeout_in_ms=400000),
            )
        except tf.errors.DeadlineExceededError:
            logging.error('evaluation timed out')
            return None
        except tf.errors.OutOfRangeError:
            logging.info('evaluation data exhausted; reinitialize train data')
            self.train_init(session)
            return None
        result['summary'] = tf.Summary()
        return result

    def list_all_member(self):
        logging.info('-' * 30)
        logging.info('model args:')
        for name, value in vars(self).items():
            logging.info('%s=%s', name, value)
        logging.info('-' * 30)

    def get_hooks(self):
        # The frozen design requires dense cold start.
        return []

    # ===================== Mature RankMixer architecture =====================

    @staticmethod
    def _gelu(inputs):
        return 0.5 * inputs * (
            1.0 + tf.tanh(
                0.7978845608028654
                * (inputs + 0.044715 * tf.pow(inputs, 3))
            )
        )

    @staticmethod
    def _rms_norm(inputs, epsilon=1e-8, scope='rms_norm', ndims=1):
        dimension = inputs.shape[-1].value
        if dimension is None:
            raise ValueError('RMSNorm requires a static final dimension')
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            scale = tf.get_variable(
                'scale',
                [1] * ndims + [dimension],
                initializer=tf.ones_initializer(),
            )
        squared_mean = tf.reduce_mean(
            tf.pow(inputs, 2.0),
            axis=-1,
            keepdims=True,
        )
        return inputs * tf.rsqrt(tf.add(squared_mean, epsilon)) * scale

    @staticmethod
    def _layer_norm(inputs, epsilon=1e-8, scope='layer_norm'):
        dimension = inputs.get_shape().as_list()[-1]
        if dimension is None:
            raise ValueError('LayerNorm requires a static final dimension')
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            gamma = tf.get_variable(
                'gamma',
                shape=[dimension],
                initializer=tf.ones_initializer(),
                trainable=True,
            )
            beta = tf.get_variable(
                'beta',
                shape=[dimension],
                initializer=tf.zeros_initializer(),
                trainable=True,
            )
            mean, variance = tf.nn.moments(
                inputs,
                axes=-1,
                keep_dims=True,
            )
            normalized = (inputs - mean) / tf.sqrt(variance + epsilon)
            return gamma * normalized + beta

    @staticmethod
    def _mix_up(inputs, new_token_num, scope):
        _, token_num, dimension = inputs.get_shape().as_list()
        if token_num is None or dimension is None:
            raise ValueError('mix_up requires static token and channel dimensions')
        if dimension % new_token_num != 0:
            raise ValueError('{} is not divisible by {}'.format(
                dimension, new_token_num))
        new_dimension = dimension // new_token_num * token_num
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            output = tf.reshape(
                inputs,
                [-1, token_num, new_token_num, dimension // new_token_num],
            )
            output = tf.transpose(output, [0, 2, 1, 3])
            return tf.reshape(output, [-1, new_token_num, new_dimension])

    @staticmethod
    def _add_weight(name, shape, initializer=None, dtype=tf.float32,
                    trainable=True, regularizer=None):
        # Scope layout intentionally matches the mature fused/non-fused paths.
        with tf.variable_scope('mlp_mixer', reuse=tf.AUTO_REUSE):
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                return tf.get_variable(
                    name=name,
                    shape=shape,
                    initializer=initializer,
                    dtype=dtype,
                    trainable=trainable,
                    regularizer=regularizer,
                )

    def _matmul_dense(self, name, inputs, units, regularizer):
        token_num = inputs.get_shape().as_list()[0]
        token_dimension = inputs.get_shape().as_list()[-1]
        if token_num is None or token_dimension is None:
            raise ValueError('per-token Dense requires static T and D')
        kernel_shape = (token_num, token_dimension, units)
        bias_shape = (token_num, 1, units)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            scale = 1.0 / max(1.0, float(token_dimension + units) / 2.0)
            stddev = np.sqrt(scale) / 0.87962566103423978
            kernel = self._add_weight(
                name=name + 'kernel',
                shape=kernel_shape,
                initializer=tf.truncated_normal_initializer(stddev=stddev),
                regularizer=regularizer,
            )
            bias = self._add_weight(
                name=name + 'bias',
                shape=bias_shape,
                initializer=tf.zeros_initializer(),
            )
            return tf.matmul(inputs, kernel) + bias

    def _per_token_swiglu_non_fused(self, inputs, layer_index, mode,
                                    regularizer, residual_scale=1.0):
        """Mature export/test path mathematically aligned with the fused op."""
        normalized = self._layer_norm(
            inputs,
            scope='pre_ln_{}'.format(layer_index),
        )
        shortcut = inputs
        transposed = tf.transpose(normalized, perm=[1, 0, 2])
        hidden_dimension = self.mixer_hidden_dim

        gate = self._matmul_dense(
            'pwff_fc1_{}'.format(layer_index),
            transposed,
            hidden_dimension,
            regularizer,
        )
        if mode == 'export':
            gate = gate * tf.sigmoid(gate)
        else:
            gate = tf.nn.swish(gate)
        value = self._matmul_dense(
            'pwff_fc2_{}'.format(layer_index),
            transposed,
            hidden_dimension,
            regularizer,
        )
        hidden = self._rms_norm(
            gate * value,
            scope='hidden_rms_norm_{}'.format(layer_index),
            ndims=2,
        )

        token_num = transposed.get_shape().as_list()[0]
        with tf.variable_scope(
                'pwff_fc3_{}'.format(layer_index), reuse=tf.AUTO_REUSE):
            # The initializer is ignored when restoring the trained fused path;
            # the variable name and shape are deliberately identical.
            kernel = tf.get_variable(
                'kernel',
                shape=(token_num, hidden_dimension, self.mixup_token_dim),
                initializer=tf.zeros_initializer(),
                regularizer=regularizer,
            )
            bias = tf.get_variable(
                'bias',
                shape=(token_num, 1, self.mixup_token_dim),
                initializer=tf.zeros_initializer(),
            )
            output = tf.matmul(hidden, kernel) + bias
        output = tf.transpose(output, perm=[1, 0, 2])
        output = self._rms_norm(
            output,
            epsilon=1e-8,
            scope='w3_output_rms_norm{}'.format(layer_index),
        )
        return shortcut + output * residual_scale

    def _per_token_swiglu_fused(self, inputs, layer_index, regularizer,
                                residual_scale=1.0):
        """Exact mature Phalanx/Cayman training path."""
        from cayman.python.custom_train_ops import swiglu

        normalized = self._layer_norm(
            inputs,
            scope='pre_ln_{}'.format(layer_index),
        )
        shortcut = inputs
        transposed = tf.transpose(normalized, perm=[1, 0, 2])
        token_num = transposed.get_shape().as_list()[0]
        token_dimension = self.mixup_token_dim
        hidden_dimension = self.mixer_hidden_dim

        scale = 1.0 / max(
            1.0,
            float(token_dimension + hidden_dimension) / 2.0,
        )
        stddev = np.sqrt(scale) / 0.87962566103423978
        with tf.variable_scope(
                'pwff_fc1_{}'.format(layer_index), reuse=tf.AUTO_REUSE):
            gate_kernel = self._add_weight(
                name='pwff_fc1_{}kernel'.format(layer_index),
                shape=(token_num, token_dimension, hidden_dimension),
                initializer=tf.truncated_normal_initializer(stddev=stddev),
                regularizer=regularizer,
            )
            gate_bias = self._add_weight(
                name='pwff_fc1_{}bias'.format(layer_index),
                shape=(token_num, 1, hidden_dimension),
                initializer=tf.zeros_initializer(),
            )
        with tf.variable_scope(
                'pwff_fc2_{}'.format(layer_index), reuse=tf.AUTO_REUSE):
            value_kernel = self._add_weight(
                name='pwff_fc2_{}kernel'.format(layer_index),
                shape=(token_num, token_dimension, hidden_dimension),
                initializer=tf.truncated_normal_initializer(stddev=stddev),
                regularizer=regularizer,
            )
            value_bias = self._add_weight(
                name='pwff_fc2_{}bias'.format(layer_index),
                shape=(token_num, 1, hidden_dimension),
                initializer=tf.zeros_initializer(),
            )
        with tf.variable_scope(
                'pwff_fc3_{}'.format(layer_index), reuse=tf.AUTO_REUSE):
            down_kernel = tf.get_variable(
                'kernel',
                shape=(token_num, hidden_dimension, token_dimension),
                initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / math.sqrt(float(hidden_dimension))),
                regularizer=regularizer,
            )
            down_bias = tf.get_variable(
                'bias',
                shape=(token_num, 1, token_dimension),
                initializer=tf.zeros_initializer(),
            )
        with tf.variable_scope(
                'hidden_rms_norm_{}'.format(layer_index), reuse=tf.AUTO_REUSE):
            rms_scale = tf.get_variable(
                'scale',
                shape=[1, 1, hidden_dimension],
                initializer=tf.ones_initializer(),
            )

        output = swiglu(
            transposed,
            gate_kernel,
            gate_bias,
            value_kernel,
            value_bias,
            down_kernel,
            down_bias,
            rms_scale=rms_scale,
            rms_epsilon=1e-8,
        )
        output = tf.transpose(output, perm=[1, 0, 2])
        output = self._rms_norm(
            output,
            epsilon=1e-8,
            scope='w3_output_rms_norm{}'.format(layer_index),
        )
        return shortcut + output * residual_scale

    def _mlp_mixer_swiglu(self, inputs, mode, regularizer):
        with tf.variable_scope(
                'mlp_mixer', reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            current = inputs
            for index in range(self.mlp_mixer_layers):
                layer_index = index + 1
                suffix = '' if index == 0 else str(layer_index)
                mixed = self._mix_up(
                    current,
                    self.mixup_token_num,
                    'mix_up{}'.format(suffix),
                )
                if mode == 'train':
                    current = self._per_token_swiglu_fused(
                        mixed,
                        layer_index,
                        regularizer,
                        residual_scale=1.0,
                    )
                else:
                    current = self._per_token_swiglu_non_fused(
                        mixed,
                        layer_index,
                        mode,
                        regularizer,
                        residual_scale=1.0,
                    )
            return self._layer_norm(
                current,
                scope='final_layer_norm',
            )

    def _mature_batch_norm(self, inputs, is_train, export, scope, reuse):
        if self.enable_phalanx:
            from phalanx.tensorflow.sync_bn import batch_norm
            return batch_norm(
                inputs,
                decay=self.batch_norm_decay,
                center=True,
                scale=True,
                updates_collections=None,
                is_training=is_train,
                reuse=reuse,
                scope=scope,
                require_robust_algo=True,
            )
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

    def _excitation2(self, input_layer, target_feature, is_train, export,
                     lowrank, scope, regularizer):
        output_size = target_feature.shape[1].value
        if output_size is None:
            raise ValueError('excitation2 requires a static target width')
        with tf.variable_scope(
                'senet16_{}'.format(scope),
                partitioner=self.partitioner,
                reuse=tf.AUTO_REUSE):
            squeeze = tf.layers.dense(
                input_layer,
                units=lowrank,
                activation=None,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name='squeeze',
            )
            reuse = tf.AUTO_REUSE if is_train else None
            squeeze = self._mature_batch_norm(
                squeeze,
                is_train,
                export,
                scope='bn_input_se',
                reuse=reuse,
            )
            squeeze = tf.nn.relu(squeeze)
            gate = tf.layers.dense(
                squeeze,
                units=output_size,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.zeros_initializer(),
                kernel_regularizer=regularizer,
                bias_initializer=tf.zeros_initializer(),
                name='excitation',
            )
            return target_feature * gate

    def _embedding_to_tokens(self, inputs, token_count, is_train, export,
                             scope, regularizer):
        output_width = token_count * self.mixup_token_dim
        with tf.variable_scope(
                scope, partitioner=self.partitioner, reuse=tf.AUTO_REUSE):
            projected = tf.layers.dense(
                inputs=inputs,
                units=output_width,
                activation=self._gelu,
                kernel_regularizer=regularizer,
                name='mlp_to_tokens',
            )
            projected = self._mature_batch_norm(
                projected,
                is_train,
                export,
                scope='bn_{}'.format(scope),
                reuse=tf.AUTO_REUSE,
            )
            return projected

    def _bottom_embedding_to_global_token(self, bottom_embeddings, regularizer):
        with tf.variable_scope(
                'direct_global_token',
                partitioner=self.partitioner,
                reuse=tf.AUTO_REUSE):
            global_input = tf.concat(
                bottom_embeddings,
                axis=1,
                name='bottom_embedding_concat',
            )
            global_input = self._layer_norm(
                global_input,
                scope='bottom_embedding_ln',
            )
            hidden = tf.layers.dense(
                inputs=global_input,
                units=self.global_token_hidden_dim,
                activation=self._gelu,
                kernel_regularizer=regularizer,
                name='bottom_mlp_hidden',
            )
            token = tf.layers.dense(
                inputs=hidden,
                units=self.mixup_token_dim,
                activation=None,
                kernel_regularizer=regularizer,
                name='bottom_mlp_projection',
            )
            return self._layer_norm(token, scope='global_token_ln')

    def _creative_converter(self, inputs, is_train, export):
        def apply_bn(tensor, suffix):
            reuse = tf.AUTO_REUSE if is_train else None
            return self._mature_batch_norm(
                tensor,
                is_train,
                export,
                scope='bn_{}'.format(suffix),
                reuse=reuse,
            )

        with tf.variable_scope(
                'mlp_mixer', reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            with tf.variable_scope(
                    'transform_creative', reuse=tf.AUTO_REUSE):
                hidden = tf.layers.dense(
                    inputs=inputs,
                    units=self.creative_hidden_dim,
                    activation=None,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='hidden_dense',
                )
                hidden = apply_bn(hidden, 'hidden')
                hidden_beta = tf.get_variable(
                    name='swish_beta_hiddentransform_creative',
                    dtype=tf.float32,
                    shape=hidden.shape[1],
                    initializer=tf.constant_initializer(1.702),
                )
                hidden = hidden * tf.nn.sigmoid(hidden_beta * hidden)

                output = tf.layers.dense(
                    inputs=hidden,
                    units=self.creative_output_dim,
                    activation=None,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    name='projection',
                )
                output = apply_bn(output, 'input')
                output_beta = tf.get_variable(
                    name='swish_beta_transform_creative',
                    dtype=tf.float32,
                    shape=output.shape[1],
                    initializer=tf.constant_initializer(1.702),
                )
                return output * tf.nn.sigmoid(output_beta * output)

    def _task_head(self, context, is_train, export):
        hidden = context
        for index, layer_size in enumerate(self.cvr_layers):
            hidden = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=layer_size,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(
                    stddev=1.0 / math.sqrt(hidden.shape[1].value)),
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_deep),
                scope='mlp{}'.format(index),
            )
            hidden = self._mature_batch_norm(
                hidden,
                is_train,
                export,
                scope='bn_{}'.format(index),
                reuse=tf.AUTO_REUSE if is_train else None,
            )
            hidden = self.get_act_func(self.mlp_act_type)(hidden)

        with tf.device('/job:ps/task:0'):
            output = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=1,
                activation_fn=tf.identity,
                weights_initializer=tf.random_normal_initializer(
                    stddev=1.0 / math.sqrt(hidden.shape[1].value)),
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_deep),
                scope='deep_out',
            )
        return output

    @staticmethod
    def _concat_feature_ids(feature_embedding_map, feature_ids, group_name):
        missing = [feature_id for feature_id in feature_ids
                   if feature_id not in feature_embedding_map]
        if missing:
            raise ValueError('{} missing lookup embeddings: {}'.format(
                group_name, missing))
        return tf.concat(
            [feature_embedding_map[feature_id] for feature_id in feature_ids],
            axis=1,
            name='{}_ordered_concat'.format(group_name),
        )

    def model_fn(self, features, labels, timestamps=None, mode='train', export=False):
        variable_partitions = self.num_ps
        if self.max_partitions is not None:
            variable_partitions = min(variable_partitions, self.max_partitions)
        self.partitioner = tf.min_max_variable_partitioner(
            max_partitions=variable_partitions,
            min_slice_size=1024000,
        )
        is_train = (mode == 'train')
        logging.info(
            'build Pure Mature RankMixer D256: mode=%s, is_train=%s, partitions=%s',
            mode, is_train, variable_partitions,
        )

        ps_mode = 'predict' if self.ps_stage == 'join' and is_train else mode
        sparse_embeddings = lookup_utils.flood_lookup_psv2(
            features=features,
            non_seq_columns=self.features.lookup_nonseq_columns,
            seq_columns=self.features.seq_columns,
            batch_size=self.batch_size,
            mode=ps_mode,
            clicks=tf.cast(labels, tf.float32),
            no_update_fea_names=list(self.fea_conf_obj.const_fea_map.keys()),
        )
        lookup_columns = (
            self.features.lookup_nonseq_columns + self.features.seq_columns
        )
        if len(sparse_embeddings) != len(lookup_columns):
            raise ValueError('lookup result/column length mismatch: {} vs {}'.format(
                len(sparse_embeddings), len(lookup_columns)))

        feature_embedding_map = {}
        for index, column in enumerate(lookup_columns):
            feature_id = get_sparse_fc_key(column)
            if feature_id in feature_embedding_map:
                raise ValueError('duplicate lookup feature ID: {}'.format(feature_id))
            feature_embedding_map[feature_id] = sparse_embeddings[index]

        user_group_tensors = []
        for group_name, feature_ids, _ in self._USER_GROUPS:
            user_group_tensors.append(self._concat_feature_ids(
                feature_embedding_map, feature_ids, group_name))
        item_group_tensors = []
        for group_name, feature_ids, _ in self._ITEM_GROUPS:
            item_group_tensors.append(self._concat_feature_ids(
                feature_embedding_map, feature_ids, group_name))
        creative_raw = self._concat_feature_ids(
            feature_embedding_map,
            _CREATIVE_IDS,
            'creative',
        )
        user_raw = tf.concat(user_group_tensors, axis=1, name='user_part')
        item_raw = tf.concat(item_group_tensors, axis=1, name='item_part')

        expected_widths = (385 * 17, 835 * 17, 14 * 17)
        actual_widths = (
            user_raw.shape[1].value,
            item_raw.shape[1].value,
            creative_raw.shape[1].value,
        )
        if actual_widths != expected_widths:
            raise ValueError('three-bucket widths={}, expected={}'.format(
                actual_widths, expected_widths))

        with tf.variable_scope(
                'Cvr-task-part',
                reuse=tf.AUTO_REUSE,
                partitioner=self.partitioner) as dense_scope:
            reuse = tf.AUTO_REUSE if is_train else None
            user_bn = self._mature_batch_norm(
                user_raw, is_train, export,
                scope='embed_bn_input_user_part', reuse=reuse)
            item_bn = self._mature_batch_norm(
                item_raw, is_train, export,
                scope='embed_bn_input_item_part', reuse=reuse)
            creative_bn = self._mature_batch_norm(
                creative_raw, is_train, export,
                scope='embed_bn_input_creative_embeds', reuse=reuse)

            regularizer = tf.contrib.layers.l2_regularizer(self.l2_deep)
            user_senet = self._excitation2(
                user_bn, user_bn, is_train, export,
                lowrank=256, scope='user', regularizer=regularizer)
            item_senet = self._excitation2(
                tf.concat([user_bn, item_bn], axis=1),
                item_bn,
                is_train,
                export,
                lowrank=128,
                scope='item',
                regularizer=regularizer,
            )
            creative_senet = self._excitation2(
                creative_bn, creative_bn, is_train, export,
                lowrank=128, scope='creative', regularizer=regularizer)

            user_widths = [
                len(feature_ids) * self.embedding_size
                for _, feature_ids, _ in self._USER_GROUPS
            ]
            item_widths = [
                len(feature_ids) * self.embedding_size
                for _, feature_ids, _ in self._ITEM_GROUPS
            ]
            user_senet_parts = tf.split(user_senet, user_widths, axis=1)
            item_senet_parts = tf.split(item_senet, item_widths, axis=1)

            processed_tokens = []
            for part, group_config in zip(user_senet_parts, self._USER_GROUPS):
                group_name, _, token_count = group_config
                processed_tokens.append(self._embedding_to_tokens(
                    part, token_count, is_train, export,
                    scope='tokens_{}'.format(group_name),
                    regularizer=regularizer,
                ))
            for part, group_config in zip(item_senet_parts, self._ITEM_GROUPS):
                group_name, _, token_count = group_config
                processed_tokens.append(self._embedding_to_tokens(
                    part, token_count, is_train, export,
                    scope='tokens_{}'.format(group_name),
                    regularizer=regularizer,
                ))

            global_token = self._bottom_embedding_to_global_token(
                [user_bn, item_bn],
                regularizer=regularizer,
            )
            processed_tokens.append(global_token)
            flattened_tokens = tf.concat(
                processed_tokens,
                axis=1,
                name='all_31_local_plus_global',
            )
            expected_flat_width = self.mixup_token_num * self.mixup_token_dim
            actual_flat_width = flattened_tokens.get_shape().as_list()[1]
            if actual_flat_width != expected_flat_width:
                raise ValueError('token width={}, expected={}'.format(
                    actual_flat_width, expected_flat_width))
            tokens = tf.reshape(
                flattened_tokens,
                [-1, self.mixup_token_num, self.mixup_token_dim],
                name='rankmixer_input_tokens',
            )

            mixed_tokens = self._mlp_mixer_swiglu(
                tokens,
                mode=mode,
                regularizer=regularizer,
            )
            mixer_context = tf.reduce_mean(
                mixed_tokens,
                axis=1,
                name='rankmixer_mean_pool',
            )
            creative_context = self._creative_converter(
                creative_senet,
                is_train,
                export,
            )
            context = tf.concat(
                [mixer_context, creative_context],
                axis=1,
                name='rankmixer_plus_creative',
            )
            if context.shape[1].value != 288:
                raise ValueError('task context must be 256+32=288, got {}'.format(
                    context.shape[1].value))
            output = self._task_head(context, is_train, export)

        tensor_name = mode if mode else 'predict'
        logits = tf.reshape(output, shape=[-1], name=tensor_name)
        logits = tf.clip_by_value(logits, -self.clip_val, self.clip_val)
        predictions = tf.sigmoid(logits, name=tensor_name)

        self._verify_graph_dense_trainable_params(dense_scope.name)
        return {
            'logits': logits,
            'pred': predictions,
        }
