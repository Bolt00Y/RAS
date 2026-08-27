# -*- coding: utf-8 -*-
"""32×512 语义 Token、仅预测 fst_CVR 的 UniMixer v1 单文件实现。

本文件直接包含特征配置、数据集、训练/评估/导出生命周期与 UniMixer 主干；
运行时不继承任何 seq_model 或 rankmixer 模型实现。
"""
import os
import io as _io
import subprocess as _subprocess

# ---- Workaround: schedule_incr_mode resubmit UnicodeDecodeError ----
# flood platform_access.py:48 calls os.popen(cmd).read(); the returned file
# object decodes bytes with locale.getpreferredencoding() which is ASCII in
# this container, so UTF-8 Chinese text from the pokemon API crashes.
# Rebuild only the read-mode path with explicit utf-8 decoding, keeping the
# returned object type identical to the original (os._wrap_close wrapping a
# TextIOWrapper). Write mode and edge cases fall through unchanged, so this
# has no effect on any other module that calls os.popen.
_original_popen = os.popen


def _utf8_popen(cmd, mode='r', buffering=-1):
    if mode != 'r' or buffering == 0 or buffering is None:
        return _original_popen(cmd, mode, buffering)
    proc = _subprocess.Popen(cmd, shell=True, stdout=_subprocess.PIPE, bufsize=buffering)
    return os._wrap_close(_io.TextIOWrapper(proc.stdout, encoding='utf-8', errors='replace'), proc)


os.popen = _utf8_popen
# ---- End workaround ----
from pydoc import locate

import math

import numpy as np
import tensorflow as tf
import logging
from logging import Formatter, getLogger, FileHandler

import flood
from data.feature import FeatureColumnBuilder
from flood.python.training.optimizer import FloodOptimizer
from flood.python.ops import parsing_ops
from framework.hooks.new_branch_warmup_hook import Senet2NewWarmupHook
#from framework.hooks.two_model_warmup_multi_target import TwoModelMultiWarmupHook
#from framework.hooks.warm_senet_2_epoch import WarmSecondEpochHook

from utils.accumulated_metrics import *
from flood.python.utils import lookup_utils
from utils.file_utils import upload_hdfs, mkdir_hdfs
from flood.python.ops.auc import flood_auc
from ..model_base import ModelBase

from utils.odds import get_sparse_fc_key
from flood.python.data import data_util as flood_data_util
from utils import learning_rate as learning_rate_utils

try:
    from cayman.python import cal_dot_topk_indices_no_padding, layer_norm_for_train
except ImportError:
    logging.info('cal_dot_topk_indices_no_padding, layer_norm_for_train import error')

BUCKET_NAMES = ("common", "item", "creative")
EXPECTED_BUCKET_TOKEN_COUNTS = (10, 21, 1)
EXPECTED_TOKEN_NUM = sum(EXPECTED_BUCKET_TOKEN_COUNTS)


def build_semantic_feature_groups():
    """返回硬编码的 32 个细粒度语义组；列表顺序就是最终 token 顺序。"""
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
            # 当前价格、SKU 供给与商品价值（35 个字段）
            ('item_current_price_supply', [
                '16759', '27303', '6046', '863060', '27308', '22102', '22119', '302533',
                '500150', '500151', '131485', '21762', '27443', '27445', '27459', '27606',
                '770521', '500103', '16728', '16742', '16746', '20512', '500158', '6133',
                '6134', '6859', '241215065', '4017', '500003', '500015', '6041', '12204',
                '12205', '12206', '201716',
            ]),
            # 优惠券、促销、折扣与活动供给（35 个字段）
            ('item_coupon_promotion_discount', [
                '22120', '27635', '276351', '500121', '500136', '500137', '868029', '868030',
                '10524', '140707', '27447', '27626', '27634', '27640', '500120', '500134',
                '500135', '868291', '10528', '24530', '27311', '27316', '27321', '274471',
                '622316', '622555', '6852', '780011', '2022429', '2022444', '16726', '27102',
                '27616', '500159', '622530',
            ]),
            # 用户购买价格与消费偏好（35 个字段）
            ('item_user_purchase_price_preference', [
                '131474', '131475', '131476', '131478', '131479', '131482', '203708', '208000',
                '208001', '770461', '770462', '770470', '770471', '10359', '11006', '131466',
                '131467', '131468', '131470', '131472', '131473', '131483', '131484', '160065',
                '206081', '206082', '21702', '21743', '21746', '21752', '22106', '33204181',
                '900086', '206510', '22101',
            ]),
            # 用户浏览点击价格偏好（35 个字段）
            ('item_user_view_click_price_preference', [
                '200181', '21708', '21728', '22129', '22131', '24330', '24332', '27402',
                '3401321', '4418101', '770460', '770469', '867665', '867685', '870313', '870315',
                '206077', '215393', '21668', '21669', '21726', '21729', '246004', '246005',
                '246006', '246007', '246014', '33203310', '33203311', '33203312', '33203321', '33203331',
                '33866909', '33866912', '33866915',
            ]),
            # 价格差、价格排序与竞争力（35 个字段）
            ('item_price_gap_rank_competitiveness', [
                '340824', '24541', '206310', '208011', '208012', '208013', '208014', '208015',
                '21760', '24328', '24496', '24497', '24498', '27367', '27507', '33204185',
                '33204186', '33204196', '33205186', '33795608', '33795609', '33795610', '33868952', '33868953',
                '33868961', '33868965', '33868969', '33868973', '33868976', '33868977', '794165', '794201',
                '794212', '794213', '900643',
            ]),
            # 价格、促销、购买力、活动与召回源的交叉上下文（35 个字段）
            ('item_price_promotion_buypower_context', [
                '201735', '206550', '302502', '861219', '870008', '870012', '870303',
                '622533', '10520', '110041', '140700', '500001', '864219', '870310',
                '600001', '881108', '10216', '10387', '10388', '25027', '5019',
                '33866926', '33868954', '33868970', '33868978', '870177', '208030', '208034',
                '208016', '310601', '310602', '310604', '6011', '60119', '6047',
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
            # 创意图片、展示形态、券与促销表达（14 个字段）
            ('creative_display_offer', [
                '780110', '780111', '780112', '780113', '780117', '8001', '8002', '8003',
                '8007', '8310', '500157', '900137', '8203', '8207',
            ]),
        ],
    }


def validate_semantic_feature_groups(groups, expected_bucket_ids):
    """校验硬编码分组与特征配置严格一一覆盖，并返回每桶 token 数。"""
    if set(groups.keys()) != set(BUCKET_NAMES):
        raise ValueError(
            "semantic buckets={} must be {}".format(
                sorted(groups.keys()), sorted(BUCKET_NAMES)
            )
        )

    all_seen_ids = set()
    bucket_token_counts = []
    for bucket_name in BUCKET_NAMES:
        bucket_groups = groups.get(bucket_name, [])
        if not bucket_groups:
            raise ValueError(
                "semantic feature groups are empty for {}".format(bucket_name)
            )

        bucket_seen_ids = set()
        group_names = set()
        for group_name, feature_ids in bucket_groups:
            if group_name in group_names:
                raise ValueError(
                    "duplicated semantic group name: {}".format(group_name)
                )
            group_names.add(group_name)
            if not feature_ids:
                raise ValueError(
                    "semantic group {} is empty".format(group_name)
                )

            feature_id_set = set(feature_ids)
            if len(feature_id_set) != len(feature_ids):
                raise ValueError(
                    "semantic group {} contains duplicated feature ids".format(
                        group_name
                    )
                )
            duplicate_ids = bucket_seen_ids.intersection(feature_id_set)
            if duplicate_ids:
                raise ValueError(
                    "features assigned to multiple semantic groups: {}".format(
                        sorted(duplicate_ids)
                    )
                )
            bucket_seen_ids.update(feature_id_set)

        expected_ids = set(expected_bucket_ids[bucket_name])
        missing_ids = expected_ids - bucket_seen_ids
        unknown_ids = bucket_seen_ids - expected_ids
        if missing_ids or unknown_ids:
            raise ValueError(
                "semantic mapping mismatch for {}: missing={}, unknown={}".format(
                    bucket_name, sorted(missing_ids), sorted(unknown_ids)
                )
            )

        cross_bucket_ids = all_seen_ids.intersection(bucket_seen_ids)
        if cross_bucket_ids:
            raise ValueError(
                "semantic features cross buckets: {}".format(
                    sorted(cross_bucket_ids)
                )
            )
        all_seen_ids.update(bucket_seen_ids)
        bucket_token_counts.append(len(bucket_groups))

    bucket_token_counts = tuple(bucket_token_counts)
    if bucket_token_counts != EXPECTED_BUCKET_TOKEN_COUNTS:
        raise ValueError(
            "semantic bucket token counts={} must be frozen as {}".format(
                bucket_token_counts,
                EXPECTED_BUCKET_TOKEN_COUNTS,
            )
        )
    return bucket_token_counts


def flatten_semantic_group_names(groups):
    """按模型 token 顺序返回语义组名，供测试和文档核对。"""
    return tuple(
        group_name
        for bucket_name in BUCKET_NAMES
        for group_name, _ in groups[bucket_name]
    )


def pertoken_swiglu(deep_inputs, expansion_factor=2):
    """
    Per-token SwiGLU (每个 token 拥有独立的 gate/up/down 权重与偏置):
        pSwiGLU(o_i) = W_down^i ( (W_up^i o_i + b_up^i) ⊙ Swish(W_gate^i o_i + b_gate^i) ) + b_down^i
    - 3D 输入 [B, T, D]: 直接按 T 个 token 处理;

    Args:
        deep_inputs: tensor, [B, T, D] 或 [B, L]
        expansion_factor: int, 中间隐藏维 d_ff = D * expansion_factor
    """
    with tf.variable_scope("pertoken_swiglu", reuse=tf.AUTO_REUSE):
        input_shape = deep_inputs.get_shape().as_list()

        T, D = input_shape[1], input_shape[2]
        x = deep_inputs  # [B, T, D]

        d_ff = D * expansion_factor

        # 每个 token 一套独立的权重与偏置
        W_gate = tf.get_variable("W_gate", shape=[T, D, d_ff],
                                initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / D)))
        b_gate = tf.get_variable("b_gate", shape=[T, d_ff], initializer=tf.zeros_initializer())
        W_up = tf.get_variable("W_up", shape=[T, D, d_ff],
                              initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / D)))
        b_up = tf.get_variable("b_up", shape=[T, d_ff], initializer=tf.zeros_initializer())
        W_down = tf.get_variable("W_down", shape=[T, d_ff, D],
                                initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / d_ff)))
        b_down = tf.get_variable("b_down", shape=[T, D], initializer=tf.zeros_initializer())

        gate = tf.einsum('btd,tde->bte', x, W_gate) + tf.reshape(b_gate, [1, T, d_ff])  # [B, T, d_ff]
        gate = gate * tf.nn.sigmoid(gate)                     # Swish(W_gate o + b_gate)
        up = tf.einsum('btd,tde->bte', x, W_up) + tf.reshape(b_up, [1, T, d_ff])      # [B, T, d_ff]
        gated = tf.multiply(up, gate)                        # [B, T, d_ff]
        out = tf.einsum('bte,ted->btd', gated, W_down) + tf.reshape(b_down, [1, T, D])  # [B, T, D]

        return out


def anneal_tau(global_step, tau_max=1.0, tau_min=0.05,
               decay_steps=100000, decay_rate=0.98, schedule='linear',
               global_step_base=0):
    """
    依据训练步数 global_step 生成退火的温度系数 tau(标量 tensor)。
    tau 是依赖 global_step 的 tensor, 训练中每步自动变化:
        - 初期 tau 大 -> 混合矩阵接近均匀(充分混合/探索);
        - 后期 tau 小 -> 矩阵趋于稀疏/尖锐(聚焦关键交互)。

    注意: 当从 checkpoint 恢复训练时, global_step 起始值不为 0,
          需要传入 global_step_base (基准步数) 进行偏移,
          使退火从 0 开始计算, 而非从原始训练的累计步数计算。

    Args:
        global_step: 标量 tensor, 训练步数(通常 tf.train.get_global_step())
        tau_max: float, 初始(最大)温度
        tau_min: float, 退火下限, 防止 tau 过小导致 exp 溢出
        decay_steps: int, 退火步长
        decay_rate: float, 指数退火衰减率
        schedule: str, 'exponential' / 'cosine' / 'linear'
        global_step_base: int, 基准步数, 从 checkpoint 恢复时传入恢复点的 global_step
    Returns:
        标量 tensor, 当前步的 tau
    """
    with tf.name_scope('anneal_tau'):
        gs = tf.cast(global_step - global_step_base, tf.float32)
        gs = tf.maximum(gs, 0.0)  # 防止基准值设置过大导致负数
        if schedule == 'exponential':
            # tau_max * decay_rate^(gs/decay_steps), 不低于 tau_min
            decayed = tf.train.exponential_decay(
                tau_max, gs, decay_steps, decay_rate, staircase=False)
            tau = tf.maximum(tau_min, decayed)
        elif schedule == 'cosine':
            tau = tau_min + 0.5 * (tau_max - tau_min) * \
                (1.0 + tf.cos(np.pi * gs / tf.cast(decay_steps, tf.float32)))
            tau = tf.clip_by_value(tau, tau_min, tau_max)
        elif schedule == 'linear':
            tau = tau_max - (tau_max - tau_min) * gs / tf.cast(decay_steps, tf.float32)
            tau = tf.maximum(tau_min, tau)
        else:
            tau = tf.constant(tau_max, dtype=tf.float32)
        return tau


def to_doubly_stochastic(W, tau=1.0, num_iters=10, epsilon=1e-8):
    """
    将原始参数矩阵变换为对称、稀疏、非负且双随机的混合矩阵。

    变换链路:
        W --(对称: (W+W^T)/2)--> W̃ --(稀疏: /τ)--> W̃/τ
           --(正性: exp + Sinkhorn-Knopp 交替行列归一化)--> W̄

    Args:
        W: tensor, 矩阵位于最后两维, 支持 [G, G] 或 [G, B, B] 等
        tau: float, 温度系数, 控制稀疏度(越小越尖锐/稀疏)
        num_iters: int, Sinkhorn-Knopp 交替行列归一化迭代次数
        epsilon: float, 数值稳定项
        name: str, name_scope 名称
    Returns:
        tensor, 与 W 同形, 非负、对称、双随机
    """
    with tf.name_scope('to_doubly_stochastic'):
        # 1) 对称性: 求转置均值 (W + W^T) / 2
        rank = len(W.shape.as_list())
        perm = list(range(rank))
        perm[-1], perm[-2] = perm[-2], perm[-1]  # 交换最后两维, 即矩阵转置
        W_sym = (W + tf.transpose(W, perm)) / 2.0

        # 2) 稀疏性: 除以温度系数 τ
        W_scaled = W_sym / tau

        # 3) 正性 + 双随机性: 指数化(保证非负) + Sinkhorn-Knopp 交替行列归一化
        # 减全局最大值防止 exp 溢出；整体平移不改变 Sinkhorn 结果。
        W_scaled = W_scaled - tf.reduce_max(W_scaled)
        M = tf.exp(W_scaled)
        for _ in range(num_iters):
            # 行归一化: 每行(最后一维)和为1
            M = M / (tf.reduce_sum(M, axis=-1, keepdims=True) + epsilon)
            # 列归一化: 每列(倒数第二维)和为1
            M = M / (tf.reduce_sum(M, axis=-2, keepdims=True) + epsilon)
        # 收敛后再次对称化, 保证严格对称(对已近似双随机的矩阵近似无副作用)
        M = (M + tf.transpose(M, perm)) / 2.0
        return M


def unimixing_lite(deep_inputs, block_size=16, rank=128, num_bases=8,
                   global_step=None, tau_max=1.0, tau_min=0.05,
                   decay_steps=100000, decay_rate=0.98, schedule='linear',
                   sinkhorn_iters=10, global_step_base=0):
    """
    通过参数分解大幅降低混合矩阵参数量, 同时保持对称/稀疏/双随机性质。
      1) 全局变换矩阵 W_G 低秩分解: W_G = U @ V^T  (U,V: [G, rank])
         参数量 G^2 -> 2*G*rank
      2) 块变换矩阵 W_B 用 base 矩阵: W_B_g = sum_k coef[g,k] * Base[k]
         (Base: [num_bases, B, B], coef: [G, num_bases])
         每个块只维护一组关于基的权重向量, 参数量 G*B^2 -> num_bases*B^2 + G*num_bases

    Args:
        deep_inputs: tensor, [B, T, D]
        block_size: int, 每个块的维度 B
        rank: int, W_G 低秩分解的秩 r (r < G 时省参数)
        num_bases: int, W_B 的 base 矩阵个数 K
        global_step/tau_*/decay_*/schedule/sinkhorn_iters: 同 unimixing
    Returns:
        tensor, [B, T, D]
    """
    with tf.variable_scope("unimixing_lite", reuse=tf.AUTO_REUSE):
        input_shape = deep_inputs.get_shape().as_list()
        T, D = input_shape[1], input_shape[2]
        L = T * D
        assert L % block_size == 0, \
            "deep_inputs dim L(%d) must be divisible by block_size(%d)" % (L, block_size)
        num_blocks = L // block_size  # G = L / B, 块的个数

        deep_inputs_flat = tf.reshape(deep_inputs, [-1, L])  # [b, L]

        # ---- 全局变换矩阵 W_G: 低秩分解 W_G = U @ V^T ----
        W_G_U = tf.get_variable("W_G_U", shape=[num_blocks, rank],
                                initializer=tf.glorot_uniform_initializer())
        W_G_V = tf.get_variable("W_G_V", shape=[num_blocks, rank],
                                initializer=tf.glorot_uniform_initializer())
        W_G = tf.matmul(W_G_U, W_G_V, transpose_b=True)  # [G, G]

        # ---- 块变换矩阵 W_B: base 矩阵 + 每块权重向量 ----
        # Base: [K, B, B], coef: [G, K], W_B_g = sum_k coef[g,k] * Base[k] -> [G, B, B]
        W_B_base = tf.get_variable("W_B_base", shape=[num_bases, block_size, block_size],
                                   initializer=tf.glorot_uniform_initializer())
        W_B_coef = tf.get_variable("W_B_coef", shape=[num_blocks, num_bases],
                                   initializer=tf.glorot_uniform_initializer())
        W_B = tf.einsum("gk,kij->gij", W_B_coef, W_B_base)  # [G, B, B]

        # 退火温度 + 双随机变换(保持对称/稀疏/双随机性质)
        tau = anneal_tau(global_step, tau_max=tau_max, tau_min=tau_min,
                         decay_steps=decay_steps, decay_rate=decay_rate,
                         schedule=schedule, global_step_base=global_step_base)
        tf.add_to_collection('unimixer_tau', tau)
        W_G = to_doubly_stochastic(W_G, tau=tau, num_iters=sinkhorn_iters)
        W_B = to_doubly_stochastic(W_B, tau=tau, num_iters=sinkhorn_iters)

        logging.info("W_G shape: %s, W_B shape: %s" % (W_G.get_shape().as_list(), W_B.get_shape().as_list()))

        # ---- 混合 ----
        deep_inputs_reshaped = tf.reshape(deep_inputs_flat, [-1, num_blocks, block_size])  # [b, G, B]
        logging.info("deep_inputs_reshaped shape: %s" % deep_inputs_reshaped.get_shape().as_list())

        # 块内交互: [G, b, block_size] x [G, block_size, block_size] -> [G, b, block_size]
        intra_out = tf.matmul(tf.transpose(deep_inputs_reshaped, [1, 0, 2]), W_B)
        # 块间交互: [b, block_size, G] x [G, G] -> [b, block_size, G]
        intra_out_t = tf.transpose(intra_out, [1, 2, 0])  # [b, block_size, G]
        inter_out = tf.einsum('blg,gh->blh', intra_out_t, W_G)
        x = tf.transpose(inter_out, [0, 2, 1])  # [b, G, block_size]
        logging.info("intra_out shape: %s" % intra_out.get_shape().as_list())

        output = tf.reshape(x, [-1, T, D])  # [b, T, D]
        return output




def semantic_unimixer_stack(
        input_tokens,
        num_blocks,
        partitioner,
        global_step,
        block_size=32,
        rank=128,
        num_bases=8,
        swiglu_expansion=2,
        tau_max=1.0,
        tau_min=0.05,
        tau_decay_steps=120000,
        tau_decay_rate=0.98,
        tau_schedule="linear",
        sinkhorn_iters=10,
        global_step_base=0,
        rms_epsilon=1e-8):
    """以论文的 SiameseNorm 拓扑堆叠 UniMixing-Lite 与 Pertoken SwiGLU。

    Args:
        input_tokens: [batch, token_num, token_dim] 的语义 token。
        num_blocks: UniMixer block 数量。
        partitioner: 当前模型的参数服务器 partitioner。
        global_step: 用于温度退火的 TensorFlow global step。
        其余参数与本文件内联的 ``unimixing_lite`` 一致。

    Returns:
        [batch, token_num, token_dim] 的最终 SiameseNorm 表示。
    """
    shape = input_tokens.get_shape().as_list()
    if len(shape) != 3 or shape[1] is None or shape[2] is None:
        raise ValueError(
            "semantic UniMixer input must have static [B,T,D], got {}".format(
                shape
            )
        )
    if global_step is None:
        raise ValueError("global_step is required for UniMixer temperature annealing")

    token_num, token_dim = shape[1], shape[2]
    flat_dim = token_num * token_dim
    if flat_dim % block_size != 0:
        raise ValueError(
            "T*D={} must be divisible by block_size={}".format(
                flat_dim, block_size
            )
        )

    with tf.variable_scope(
            "semantic_unimixer",
            reuse=tf.AUTO_REUSE,
            partitioner=partitioner):
        input_tokens = tf.identity(input_tokens, name="input_tokens")
        tf.add_to_collection("unimixer_v1_input_tokens", input_tokens)

        stream_x = input_tokens
        stream_y = input_tokens
        for block_idx in range(num_blocks):
            with tf.variable_scope(
                    "unimixer_block_{}".format(block_idx),
                    reuse=tf.AUTO_REUSE):
                normalized_y = ModelBase.rms_norm(
                    stream_y,
                    scope="siamese_y_norm",
                    epsilon=rms_epsilon,
                )
                block_input = stream_x + normalized_y

                # 论文 Eq.(18): 先进行可学习的局部/全局混合。
                mixed = unimixing_lite(
                    block_input,
                    block_size=block_size,
                    rank=rank,
                    num_bases=num_bases,
                    global_step=global_step,
                    tau_max=tau_max,
                    tau_min=tau_min,
                    decay_steps=tau_decay_steps,
                    decay_rate=tau_decay_rate,
                    schedule=tau_schedule,
                    sinkhorn_iters=sinkhorn_iters,
                    global_step_base=global_step_base,
                )

                # 论文 Eq.(16)/(18): RMSNorm(X + UniMixing-Lite(X))。
                mixed_residual = ModelBase.rms_norm(
                    block_input + mixed,
                    scope="mix_residual_norm",
                    epsilon=rms_epsilon,
                )
                block_output = pertoken_swiglu(
                    mixed_residual,
                    expansion_factor=swiglu_expansion,
                )

                # 论文 Eq.(20) 前的双流更新。
                stream_x = ModelBase.rms_norm(
                    stream_x + block_output,
                    scope="siamese_x_norm",
                    epsilon=rms_epsilon,
                )
                stream_y = stream_y + block_output

                logging.info(
                    "UniMixer v1 block %d: input=%s, mixed=%s, output=%s",
                    block_idx,
                    block_input.get_shape(),
                    mixed.get_shape(),
                    block_output.get_shape(),
                )

        output_tokens = stream_x + ModelBase.rms_norm(
            stream_y,
            scope="siamese_final_y_norm",
            epsilon=rms_epsilon,
        )
        output_tokens = tf.identity(output_tokens, name="output_tokens")
        tf.add_to_collection("unimixer_v1_output_tokens", output_tokens)

    return output_tokens


class MLPModel(ModelBase):
    """仅支持 common/item/creative 与 fst_CVR 的 UniMixer v1。"""

    _BUCKET_NAMES = BUCKET_NAMES
    _TOKEN_NUM = EXPECTED_TOKEN_NUM
    _TOKEN_DIM = 512
    _DEFAULT_BLOCK_SIZE = 32
    _DISABLED_TASK_FLAGS = (
        "enable_last_cvr",
        "enable_wide_cvr",
        "enable_mlt_loss",
        "enable_delay_train_mode",
    )

    def __init__(self, **_kwargs):
        requested_extra_tasks = [
            flag for flag in self._DISABLED_TASK_FLAGS
            if bool(_kwargs.get(flag, False))
        ]
        if requested_extra_tasks:
            raise ValueError(
                "UniMixer v1 predicts fst_CVR only; unsupported flags: {}".format(
                    requested_extra_tasks
                )
            )

        # 原始 base 的 wide/last/mlt 默认开启。这里在本类内部先覆盖为 False，
        # 使同文件内的 loss/metric/train/export 生命周期只处理 fst_CVR。
        _kwargs = dict(_kwargs)
        _kwargs.update({flag: False for flag in self._DISABLED_TASK_FLAGS})
        _kwargs["enable_bound_loss"] = False
        # 原始实现读取历史拼写 enable_mtl_warmup；两个别名都显式关闭。
        _kwargs["enable_mtl_warmup"] = False
        _kwargs["enable_mlt_warmup"] = False
        _kwargs["enable_wide_cvr_fst_warmup"] = False

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
        self.enable_mlt_warmup = _kwargs.get("enable_mtl_warmup", False)
        self.hooks = _kwargs.get('hooks', [])
        self.skip_tensors = _kwargs.get("skip_tensors", "")
        self.warm_up_tensors = _kwargs.get("warm_up_tensors", "dcnm-cross;mlp0;bn_input")
        self.warmup_type = _kwargs.get('warmup_type', 'default')
        self.warm_mlp_layer = _kwargs.get("warm_mlp_layer", [])
        self.use_mlp_gate = _kwargs.get('use_mlp_gate', False)
        self.old_epoch_ckpt_import_dir = _kwargs.get("old_epoch_ckpt_import_dir", None)
        self.ckpt_import_dir1 = _kwargs.get("ckpt_import_dir1", None)
        self.ckpt_import_dir2 = _kwargs.get("ckpt_import_dir2", None)
        self.warm_up_tensors1 = _kwargs.get("warm_up_tensors1", "mlp0;mlp1;mlp2;deep_out")
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

        # senet conf
        self.senet_hidden_size = _kwargs.get('senet_hidden_size', 128)
        self.use_senet = _kwargs.get('use_senet', False)
        self.use_senet_bn = _kwargs.get('use_senet_bn', False)

        # cvr model conf
        self.cvr_layers = _kwargs.get('cvr_layers', [2048,2048,256])
        self.mlt_cvr_layers = _kwargs.get('mlt_cvr_layers', [512, 256, 128])
        self.opt_goal = _kwargs.get('opt_goal', 'first_cvr')
        self.export_name = _kwargs.get('export_name', 'first_cvr')
        self.cvr_label_name = _kwargs.get('cvr_label_name', 'fst_cvr_label')

        # dense 相关
        self.dense_scale = _kwargs.get("dense_scale", 0.01)
        self.dense_global_norm = _kwargs.get("dense_global_norm", True)  # do global norm for dense feature or not
        self.dense_clip_threshold = _kwargs.get("dense_clip_threshold", [-2000000.0,2000000.0])  # clip dense fea value with delete_threshold

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
        # replay conf
        self.train_stage_param = _kwargs.get('train_stage_param', 'replay##dist2')
        self.sampler_label_name = _kwargs.get('sampler_label_name', '')
        self.sampler_positive_rate = _kwargs.get('sampler_positive_rate', 1.0)
        self.sampler_negative_rate = _kwargs.get('sampler_negative_rate', 1.0)
        self.enable_neg_sampler = _kwargs.get('enable_neg_sampler', True)
        self.filter_pass_values = _kwargs.get('filter_pass_values', '')
        self.filter_label_names = _kwargs.get('filter_label_names', '')
        self.filter_drop_values = _kwargs.get('filter_drop_values', '')
        self.filter_pass_empty = _kwargs.get('filter_pass_empty', True)

        # dcnm conf
        self.dcnm_layer = _kwargs.get('dcnm_layer', 500)
        self.cross_num = _kwargs.get('cross_num', 2)
        self.use_cross_act = _kwargs.get('use_cross_act', False)
        self.use_dcnm_ln = _kwargs.get('use_dcnm_ln', True)
        self.use_dcnm_bn = _kwargs.get('use_dcnm_bn', False)
        self.layer_norm_opt = _kwargs.get('layer_norm_opt', True)

        # wide conf
        self.enable_wide_cvr = _kwargs.get('enable_wide_cvr', True)
        self.wide_stop_gradient = _kwargs.get('wide_stop_gradient', False)
        self.last_stop_gradient = _kwargs.get('last_stop_gradient', True)
        # last conf
        self.enable_last_cvr = _kwargs.get('enable_last_cvr', True)
        self.enable_wide_cvr_fst_warmup = _kwargs.get('enable_wide_cvr_fst_warmup', False)
        self.wide_sim_fst_label = _kwargs.get('wide_sim_fst_label', 'wide_sim_noquery_fst_label')
        self.wide_loss_weight = _kwargs.get('wide_loss_weight', 0.3)
        self.last_loss_weight = _kwargs.get('last_loss_weight', 0.5)


        # delay conf
        self.enable_delay_train_mode = _kwargs.get('enable_delay_train_mode', False)
        self.fst_cvr_delay_label = _kwargs.get('fst_cvr_delay_label', 'delay_2d_fst_cvr_label')
        self.delay_checkpoint_import_dir = _kwargs.get('delay_checkpoint_import_dir', 'hdfs://pdd-data-ns/a/b/c')
        self.enable_export_delay_model = False  # 标志位

        # 优化监控相关
        self.enable_bound_loss = _kwargs.get('enable_bound_loss', False)

        # 多目标相关
        self.enable_mlt_loss = _kwargs.get('enable_mlt_loss', True)
        self.mlt_loss_weight = _kwargs.get('mlt_loss_weight', 0.2)
        self.time_loss_weight = _kwargs.get('time_loss_weight', 0.0001)
        self.ce_loss_weight = _kwargs.get('ce_loss_weight', 0.01)
        self.mlt_tuning_v2 = _kwargs.get('mlt_tuning_v2', False)
        self.stop_dcnm_gradient = _kwargs.get('stop_dcnm_gradient', False)
        self.mlt_tasks = _kwargs.get('mlt_tasks', 3)
        self.mlt_tuning = _kwargs.get('mlt_tuning', False)

        self.eval_count = 0
        self.num_ps = 1
        self.num_worker = 1
        if self.tf_config:
            self.num_ps = len(self.tf_config["cluster"]["ps"])
            self.num_worker = len(self.tf_config["cluster"]["worker"])

        self.task_index = self.tf_config['task']['index']

        self.train_reset_interval = _kwargs.get('train_reset_interval', 10000)
        # train reset count
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
        self.dir2_all_tensor = _kwargs.get('dir2_all_tensor',"None")
        self.second_epoch_ckpt_import_dir = _kwargs.get('second_epoch_ckpt_import_dir', '')

        super().__init__()

        if self.opt_goal != "first_cvr":
            raise ValueError(
                "UniMixer v1 requires opt_goal=first_cvr, got {}".format(
                    self.opt_goal
                )
            )
        if self.cvr_label_name != "fst_cvr_label":
            raise ValueError(
                "UniMixer v1 requires cvr_label_name=fst_cvr_label, got {}".format(
                    self.cvr_label_name
                )
            )

        self.um_token_num = int(_kwargs.get("um_token_num", self._TOKEN_NUM))
        self.um_token_dim = int(_kwargs.get("um_token_dim", self._TOKEN_DIM))
        self.um_num_blocks = int(_kwargs.get("um_num_blocks", 2))
        self.um_block_size = int(
            _kwargs.get("um_block_size", self._DEFAULT_BLOCK_SIZE)
        )
        self.um_rank = int(_kwargs.get("um_rank", 128))
        self.um_num_bases = int(_kwargs.get("um_num_bases", 8))
        self.um_swiglu_expansion = int(
            _kwargs.get("um_swiglu_expansion", 2)
        )
        self.um_tau_max = float(_kwargs.get("um_tau_max", 1.0))
        self.um_tau_min = float(_kwargs.get("um_tau_min", 0.05))
        self.um_tau_decay_steps = int(
            _kwargs.get("um_tau_decay_steps", 120000)
        )
        self.um_tau_decay_rate = float(
            _kwargs.get("um_tau_decay_rate", 0.98)
        )
        self.um_tau_schedule = _kwargs.get("um_tau_schedule", "linear")
        self.um_sinkhorn_iters = int(
            _kwargs.get("um_sinkhorn_iters", 10)
        )
        self.um_global_step_base = int(
            _kwargs.get("um_global_step_base", 0)
        )
        self.um_rms_epsilon = float(
            _kwargs.get("um_rms_epsilon", 1e-8)
        )
        self.um_use_token_bn = bool(
            _kwargs.get("um_use_token_bn", True)
        )

        self._validate_architecture_config(_kwargs)
        self.um_semantic_feature_groups = build_semantic_feature_groups()
        expected_bucket_ids = {
            "common": self.fea_conf_obj.common_fea_map.keys(),
            "item": self.fea_conf_obj.item_fea_map.keys(),
            "creative": self.fea_conf_obj.creative_fea_map.keys(),
        }
        self.um_bucket_token_counts = validate_semantic_feature_groups(
            self.um_semantic_feature_groups,
            expected_bucket_ids,
        )
        if self.um_bucket_token_counts != EXPECTED_BUCKET_TOKEN_COUNTS:
            raise ValueError(
                "semantic bucket token counts={} must equal {}".format(
                    self.um_bucket_token_counts,
                    EXPECTED_BUCKET_TOKEN_COUNTS,
                )
            )
        configured_bucket_counts = _kwargs.get("um_bucket_token_counts")
        if configured_bucket_counts is not None:
            configured_bucket_counts = tuple(
                int(value) for value in configured_bucket_counts
            )
            if configured_bucket_counts != self.um_bucket_token_counts:
                raise ValueError(
                    "um_bucket_token_counts={} must match hard-coded groups={}".format(
                        configured_bucket_counts,
                        self.um_bucket_token_counts,
                    )
                )
        if sum(self.um_bucket_token_counts) != self.um_token_num:
            raise ValueError(
                "hard-coded semantic token count={} must equal um_token_num={}".format(
                    sum(self.um_bucket_token_counts),
                    self.um_token_num,
                )
            )

        self.um_token_order = flatten_semantic_group_names(
            self.um_semantic_feature_groups
        )
        logging.info(
            "UniMixer v1: bucket_tokens=%s, T=%d, D=%d, L=%d, blocks=%d, "
            "block_size=%d, rank=%d, bases=%d, pSwiGLU_expand=%d, "
            "token_bn=%s, token_order=%s",
            self.um_bucket_token_counts,
            self.um_token_num,
            self.um_token_dim,
            self.um_token_num * self.um_token_dim,
            self.um_num_blocks,
            self.um_block_size,
            self.um_rank,
            self.um_num_bases,
            self.um_swiglu_expansion,
            self.um_use_token_bn,
            self.um_token_order,
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
        """获取数据集
        Args:
            data_paths: 数据路径列表
            mode: 模式，支持 "train", "test", "predict"
            use_dynamic_file: 是否使用动态文件
            take_batch_num: 要获取的批次数量

        Returns:
            包含数据集的字典
        """

        # 1. 确定特征列和解析规范
        parquet_cols = self.features.parquet_reader_columns
        features_spec = tf.feature_column.make_parse_example_spec(parquet_cols)
        # 2. 配置序列特征的长度限制，性能优化的参数
        size_limits_map = self.fea_conf_obj.feature_size_limit_map
        # 3. 多版本embedding 配置
        feature_name_map = self.fea_conf_obj.features_multi_map
        # 4. sparse_features_to_tensor 配置
        visible_feature_lst = self.fea_conf_obj.visible_fea_map.keys()

        # 5. 创建并返回数据集
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
        """构建完整的模型计算图
        Args:
            input_paths: 训练数据路径
            test_paths: 测试数据路径
            mode: 运行模式，默认为'train'
            config: 配置字典
            use_dynamic: 是否使用动态文件
            **kwargs: 其他参数
        """
        # 1. 初始化基础配置
        self.global_step = tf.train.get_or_create_global_step()
        self.global_step_op = tf.assign_add(self.global_step, 1)
        # 2. 为每个模式构建计算图
        for tmp_mode in ['train', 'test']:
            # 2.1 打印当前构建阶段信息
            logging.info(f"{'*' * 10} {tmp_mode} {'*' * 10}")
            # 2.2 构建数据集操作
            data_paths = test_paths if tmp_mode == 'test' else input_paths
            self.build_dataset_op(data_paths, mode=tmp_mode, flood_mode=mode)
            # 2.3 构建预测结果操作
            self.build_pred_results_op(mode=tmp_mode, flood_mode=mode)
            # 2.4 构建评估指标操作
            self.build_auc_copc_op(mode=tmp_mode)
            # 2.5 训练模式下构建损失操作 & 构建优化器
            if tmp_mode == 'train':
                self.build_loss_op(mode=tmp_mode)
                self.build_summary(mode=tmp_mode)
                self.build_optimizer_op()
        # 3.  构建导出相关操作
        self._build_export(config=config)
        # 4. 配置运行时选项
        self.run_metadata = tf.RunMetadata()
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, timeout_in_ms=self.timeout)
        self.timeout_options = tf.RunOptions(timeout_in_ms=self.timeout)

        if self.log_nn_vars:
            global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            logging.info('global_vars:')
            for var in global_vars:
                logging.info('{}'.format(var))
        if self.enable_delay_train_mode:
            for k, v in config.items():
                logging.info(f"[flood config] {k} {v}")
            if config["mode"] == "export":
                self.enable_export_delay_model = True

    def build_dataset_op(self, data_paths, mode, flood_mode):
        """构建数据集操作
        Args:
            data_paths: 数据路径列表
            mode: 模式，支持 "train", "test", "predict"
            flood_mode: flood_mode
        """

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
        for key, value in res.items():  # [features, label, sampleid, search_id, example_id]
            self[f'{mode}_{key}'] = value

    def parse_examples(self, *example_batch):
        """解析输入数据批次

        Args:
            *example_batch: 输入的数据批次

        Returns:
            包含特征、标签和指示符的字典 {
                'features': 解析后的特征字典,
                'labels': 点击标签(tf.float32),
            }
        """
        # 1. 配置解析参数
        columns = self.features.parquet_reader_columns
        # 2. 解析特征数据
        features = parsing_ops.parse_parquet(
            example_batch,
            tf.feature_column.make_parse_example_spec(columns),
            reserved_keys=self.fea_conf_obj.visible_fea_map,
            unique=False,
            share_embedding_conf=self.fea_conf_obj.features_share_map,
            global_hash=False,
            psv2=True
        )
        # 3. 生成样本ID并处理标签
        features["sampleid"] = flood.generate_sample_id(
            search_ids=features["search_id"].values,
            example_ids=features["example_ids"].values)
        label_cvr_first = tf.cast(features.pop('fst_cvr_label'), tf.float32)
        label_cvr_last = tf.cast(features.pop('last_cvr_label'), tf.float32)
        sampleid = tf.cast(features.pop('sampleid'), tf.float32)
        search_id = features["search_id"].values
        example_id = features["example_ids"].values

        # 4. 返回结构化结果
        res = {
            'features': features,
            'labels': label_cvr_first,
            'labels_last': label_cvr_last,
            'sampleid': sampleid,
            'search_id': search_id,
            'example_id': example_id
        }

        if self.enable_wide_cvr:
            label_cvr_first_wide = tf.cast(features.pop(self.wide_sim_fst_label), tf.float32)
            res.update({'labels_wide': label_cvr_first_wide})

        if self.enable_delay_train_mode:
            label_cvr_first_delay = tf.cast(features.pop(self.fst_cvr_delay_label), tf.float32)
            res.update({'labels_delay': label_cvr_first_delay})

        return res

    def build_pred_results_op(self, mode, flood_mode=None):
        """构建预测结果操作
        Args:
            mode: 模式，可选 "train", "test", "predict"
            flood_mode: flood 框架透传的mode
        """
        fn_mode = mode if mode == 'test' else flood_mode
        results = self.model_fn(self[f'{mode}_features'], self[f'{mode}_labels'], mode=fn_mode)

        for key, value in results.items():
            self[f'{mode}_{key}'] = value

    def build_loss_op(self, mode):
        """构建损失函数操作，计算模型预测与真实标签之间的对数损失
        Args:
            mode: 模式，可选 "train", "test", "predict"
        """

        # fst loss
        labels = tf.reshape(self[f'{mode}_labels'], shape=[-1])
        logits = tf.reshape(self[f'{mode}_logits'], shape=[-1])
        pred = tf.reshape(self[f'{mode}_pred'], shape=[-1])
        self.loss_first = tf.reduce_mean(tf.losses.log_loss(predictions=pred, labels=labels))

        # last loss
        loss = self.loss_first
        if self.enable_last_cvr:
            labels_last = tf.reshape(self[f'{mode}_labels_last'], shape=[-1])
            pred_last = tf.reshape(self[f'{mode}_pred_last'], shape=[-1])
            self.labels_last_pos_count = tf.reduce_sum(labels_last)
            self.loss_last = tf.reduce_mean(
                tf.losses.log_loss(predictions=pred_last, labels=labels_last)) * self.last_loss_weight

            loss = self.loss_first + self.loss_last

        if self.enable_mlt_loss:
            features = self[f'{mode}_features']
            mlt_labels = tf.concat(
                [features['label_is_clk_buy_all'], features['label_is_cnslt'], features['label_is_clk_rev']], axis=1)
            mlt_logits = self[f'{mode}_mlt_logits']
            mlt_loss = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mlt_logits,
                                                                                             labels=tf.cast(mlt_labels,
                                                                                                            dtype=tf.float32)),
                                                     axis=0))

            self.buckets = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000, 20000, 30000, 40000,
                            60000, 80000, 120000, 180000, 300000, 480000]
            stay_time_list = [tf.cast(features['label_gpage_stay_time'] >= value, tf.int64) for value in
                              self.buckets[1:]]
            self.stay_time_labels = tf.concat(stay_time_list, axis=1)
            self.label_gpage_stay_time_labels = tf.clip_by_value(
                tf.cast(features['label_gpage_stay_time'], tf.float32) / 1000, 0, 480)

            buckets1 = [1, 5, 10, 15, 30, 60, 120, 180]
            stat_time_list = [tf.cast(self.label_gpage_stay_time_labels >= value, tf.int64) for value in buckets1]
            self.new_ce_labels1 = tf.reduce_sum(tf.concat(stat_time_list, axis=1), axis=-1)

            self.label_gpage_clk_cnt_labels = tf.clip_by_value(tf.cast(features['label_gpage_clk_cnt'], tf.float32), 0,
                                                               100)
            buckets2 = [1, 3, 5, 10, 25]
            clk_cnt_list = [tf.cast(self.label_gpage_clk_cnt_labels >= value, tf.int64) for value in buckets2]
            self.new_ce_labels2 = tf.reduce_sum(tf.concat(clk_cnt_list, axis=1), axis=-1)

            def _masked_loss(logits, labels):
                mask = tf.reshape(tf.cast((self.label_gpage_stay_time_labels > 0), dtype=tf.float32), [-1])
                ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
                cnt = tf.reduce_sum(mask)
                return tf.cond(cnt > 0, lambda: tf.reduce_sum(ce * mask) / cnt, lambda: 0.0)

            ce_loss_list = []
            res_pred_list = []
            or_loss_list = []
            for i in range(len(self.buckets) - 1):
                logits_i = self[f'{mode}_stay_time_logits'][:, i]
                stay_time_preds = self[f'{mode}_stay_time_preds']
                ce_loss_i = _masked_loss(logits_i, tf.cast(self.stay_time_labels[:, i], dtype=tf.float32))
                ce_loss_list.append(ce_loss_i)
                res_pred_i = tf.expand_dims(
                    ((self.buckets[i + 1] - self.buckets[i]) / 1000) * tf.cast(stay_time_preds[:, i], dtype=tf.float32),
                    axis=-1)
                res_pred_list.append(res_pred_i)
                if i > 0:
                    or_loss_i = tf.expand_dims(
                        tf.maximum(tf.cast(stay_time_preds[:, i] - stay_time_preds[:, i - 1], dtype=tf.float32), 0.0),
                        axis=-1)
                    or_loss_list.append(or_loss_i)
            ce_loss = tf.add_n(ce_loss_list)
            mlt_loss = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mlt_logits,
                                                                                             labels=tf.cast(mlt_labels,
                                                                                                            dtype=tf.float32)),
                                                     axis=0))
            res_preds = tf.expand_dims(tf.reduce_sum(tf.concat(res_pred_list, axis=-1), axis=-1), axis=-1)

            def _masked_loss_v2(labels, preds):
                mask = tf.cast((self.label_gpage_stay_time_labels > 0), dtype=tf.float32)
                huber = tf.losses.huber_loss(labels, preds, delta=1.35, reduction="none")
                cnt = tf.reduce_sum(mask)
                return tf.cond(cnt > 0, lambda: tf.reduce_sum(huber * mask) / cnt, lambda: 0.0)

            stay_time_loss = _masked_loss_v2(self.label_gpage_stay_time_labels, res_preds)
            or_mask = tf.cast((self.label_gpage_stay_time_labels > 0), dtype=tf.float32)
            or_loss_sum = tf.expand_dims(tf.reduce_sum(tf.concat(or_loss_list, axis=-1), axis=-1), axis=-1)
            cnt = tf.reduce_sum(or_mask)
            or_loss = tf.cond(cnt > 0, lambda: tf.reduce_sum(or_loss_sum * or_mask) / cnt, lambda: 0.0)

            ce_logits = self[f'{mode}_ce_logits']
            new_ce_loss1 = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ce_logits[:, 0:9], labels=self.new_ce_labels1))
            new_ce_loss2 = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ce_logits[:, 9:15], labels=self.new_ce_labels2))

            # 最后的loss
            self.cvr_loss = loss
            loss += mlt_loss * self.mlt_loss_weight
            self.label_is_clk_buy_all = tf.reduce_sum(features['label_is_clk_buy_all'])
            self.mlt_loss, self.stay_time_loss, self.new_ce_loss = mlt_loss, stay_time_loss, new_ce_loss1 + new_ce_loss2
            loss += self.time_loss_weight * (1 * or_loss + 100 * ce_loss + 1 * stay_time_loss)
            loss += self.ce_loss_weight * self.new_ce_loss

        self.labels_pos_cvr_count = tf.reduce_sum(labels)

        if self.enable_delay_train_mode:
            labels_delay = tf.reshape(self[f'{mode}_labels_delay'], shape=[-1])
            pred_delay = tf.reshape(self[f'{mode}_pred_delay'], shape=[-1])

            self.loss = tf.reduce_mean(tf.losses.log_loss(predictions=pred_delay, labels=labels_delay))
            self.labels_pos_cvr_count_delay = tf.reduce_sum(labels_delay)
        else:
            self.loss = loss

        if self.enable_wide_cvr:
            labels_wide = tf.reshape(self[f'{mode}_labels_wide'], shape=[-1])
            pred_wide = tf.reshape(self[f'{mode}_pred_wide'], shape=[-1])
            self.labels_wide_pos_count = tf.reduce_sum(labels_wide)
            self.loss_wide = tf.reduce_mean(
                tf.losses.log_loss(predictions=pred_wide, labels=labels_wide)) * self.wide_loss_weight

            self.loss = self.loss + self.loss_wide

    def build_auc_copc_op(self, mode):
        """构建AUC和COPC评估指标操作
        Args:
            mode: 运行模式，支持 "train", "test", "predict"
        """
        # first
        self[f'{mode}_auc'] = flood_auc(self[f'{mode}_labels'], self[f'{mode}_pred'], name='auc/cvr',
                                        num_thresholds=2000)
        self[f'{mode}_copc'] = tf.reduce_sum(self[f'{mode}_pred']) / (tf.reduce_sum(self[f'{mode}_labels']) + 1e-8)
        self[f'{mode}_auc_values'] = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES, scope='auc')
        self[f'{mode}_reset_auc_op'] = tf.variables_initializer(var_list=self[f'{mode}_auc_values'])
        self[f'{mode}_pred_mean'] = tf.reduce_mean(self[f'{mode}_pred'])

        # last
        if self.enable_last_cvr:
            self[f'{mode}_auc_last'] = flood_auc(self[f'{mode}_labels_last'], self[f'{mode}_pred_last'],
                                                 name='auc/cvr_last', num_thresholds=2000)
            self[f'{mode}_copc_last'] = tf.reduce_sum(self[f'{mode}_pred_last']) / (
                    tf.reduce_sum(self[f'{mode}_labels_last']) + 1e-8)
            self[f'{mode}_pred_mean_last'] = tf.reduce_mean(self[f'{mode}_pred_last'])

        if self.enable_wide_cvr:
            self[f'{mode}_auc_wide'] = flood_auc(self[f'{mode}_labels_wide'], self[f'{mode}_pred_wide'],
                                                 name='auc/cvr_wide', num_thresholds=2000)
            self[f'{mode}_copc_wide'] = tf.reduce_sum(self[f'{mode}_pred_wide']) / (
                    tf.reduce_sum(self[f'{mode}_labels_wide']) + 1e-8)
            self[f'{mode}_pred_mean_wide'] = tf.reduce_mean(self[f'{mode}_pred_wide'])

        if self.enable_delay_train_mode:
            self[f'{mode}_auc_delay'] = flood_auc(self[f'{mode}_labels_delay'], self[f'{mode}_pred_delay'],
                                                  name='auc/cvr_delay', num_thresholds=2000)
            self[f'{mode}_copc_delay'] = tf.reduce_sum(self[f'{mode}_pred_delay']) / (
                    tf.reduce_sum(self[f'{mode}_labels_delay']) + 1e-8)
            self[f'{mode}_pred_mean_delay'] = tf.reduce_mean(self[f'{mode}_pred_delay'])

    def build_summary(self, mode):
        """构建summary
        Args:
            mode: 模式，可选 "train", "test", "predict"
        """
        auc_summary = tf.summary.scalar(f'{mode}/auc', self[f'{mode}_auc'])
        loss_summary = tf.summary.scalar(f'{mode}/loss', self.loss)
        copc_summary = tf.summary.scalar(f'{mode}/copc', self[f'{mode}_copc'])

        self.eval_summary = tf.summary.merge([loss_summary, auc_summary, copc_summary], name='eval_summary')

    def build_optimizer_op(self):
        """构建优化器操作，包括梯度计算和应用
        """
        # Learning rate decay
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
        if self.enable_delay_train_mode:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for (grad, var) in grads_and_vars:
                logging.info(f'[delay gradiant] {grad} {var}')
                if grad is not None:
                    tf.summary.histogram('train_delay/' + var.op.name + '/gradients', grad)
            self.train_op = [self.optimizer.apply_gradients(grads_and_vars, global_step=None)]
        else:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for (grad, var) in grads_and_vars:
                logging.info(f'[normal gradiant] {grad} {var}')
                if grad is not None:
                    tf.summary.histogram('train/' + var.op.name + '/gradients', grad)
            self.train_op = [self.optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())]

    def _schedule_lr(self, lr, schedule_config: dict):
        lr = tf.convert_to_tensor(lr)
        if 'type' in schedule_config:
            logging.info('use lr decay schedule')
            # create milestone step reset op
            learning_rate_utils.get_or_create_milestone_step_reset_op()
            schedule_type = schedule_config['type']
            lr = learning_rate_utils.learning_rate_schedule(
                lr,
                schedule_type,
                **schedule_config)
        return lr

    def _build_lr_schedule(self):
        learning_rate = self.learning_rate
        learning_rate = self._schedule_lr(learning_rate, self.schedule_config)
        self.learning_rate = learning_rate

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

    def dcnm_cross_layer(self, inputs, is_train, export):
        logging.info(f'============== DCNM (common + item + creative) ============== ')
        with tf.variable_scope("dcnm-cross", reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            logging.info('using dcnm cross layer')
            deep_inputs_cvr = inputs
            logging.info('inputs shape: {}'.format(inputs.get_shape()))
            inputs_shape = int(inputs.get_shape()[-1])

            for i in range(self.cross_num):
                logging.info('dcnm cross layer %s-%s', i, self.dcnm_layer)
                if self.use_cross_act:
                    logging.info('dcnm cross %d layer nonlinear activated with %s', i, self.act_type)
                last_layer_cvr = deep_inputs_cvr
                deep_inputs_cvr = tf.contrib.layers.fully_connected(inputs=deep_inputs_cvr, num_outputs=self.dcnm_layer,
                                                                    activation_fn=self.get_act_func(
                                                                        self.act_type) if self.use_cross_act else tf.identity,
                                                                    weights_initializer=tf.random_normal_initializer(
                                                                        stddev=1 / math.sqrt(
                                                                            deep_inputs_cvr.shape[1].value)),
                                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                        self.l2_deep),
                                                                    scope='dcnm_cross_layer0_%d' % i)

                deep_inputs_cvr = tf.contrib.layers.fully_connected(inputs=deep_inputs_cvr, num_outputs=inputs_shape,
                                                                    activation_fn=tf.identity,
                                                                    weights_initializer=tf.random_normal_initializer(
                                                                        stddev=1 / math.sqrt(
                                                                            deep_inputs_cvr.shape[1].value)),
                                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                        self.l2_deep),
                                                                    scope='dcnm_cross_layer1_%d' % i)

                deep_inputs_cvr = tf.multiply(inputs, deep_inputs_cvr) + last_layer_cvr

                if self.layer_norm_opt:
                    if not export:
                        deep_inputs_cvr = layer_norm_for_train(deep_inputs_cvr, scope=f'dcnm_ln_{i}',
                                                               begin_norm_axis=-1, begin_params_axis=-1)
                    else:
                        deep_inputs_cvr = tf.contrib.layers.layer_norm(deep_inputs_cvr, center=True, scale=True,
                                                                       activation_fn=None, scope=f'dcnm_ln_{i}',
                                                                       begin_norm_axis=-1, begin_params_axis=-1)
                else:
                    deep_inputs_cvr = tf.contrib.layers.layer_norm(deep_inputs_cvr, center=True, scale=True,
                                                                   activation_fn=None, scope=f'dcnm_ln_{i}')
        return deep_inputs_cvr

    def senet_layer(self, common_embedding, item_embedding, creative_embedding, is_train, export):
        logging.info(f'============== SENet (common + item + creative) ============== ')
        common_field_num = len(self.features.common_columns)
        item_field_num = len(self.features.item_columns)
        creative_field_num = len(self.features.creative_columns)

        with tf.variable_scope("senet", reuse=tf.AUTO_REUSE, partitioner=self.partitioner):
            weight_common_in = tf.get_variable(shape=[common_field_num, self.senet_hidden_size],
                                               initializer=tf.glorot_uniform_initializer(), name="common_weight_in")
            weight_common_out = tf.get_variable(shape=[self.senet_hidden_size, common_field_num],
                                                initializer=tf.glorot_uniform_initializer(), name="common_weight_out")
            weight_item_common_in = tf.get_variable(shape=[item_field_num + common_field_num, self.senet_hidden_size],
                                                    initializer=tf.glorot_uniform_initializer(),
                                                    name="common_item_weight_in")
            weight_item_out = tf.get_variable(shape=[self.senet_hidden_size, item_field_num],
                                              initializer=tf.glorot_uniform_initializer(), name="item_weight_out")
            weight_all_in = tf.get_variable(
                shape=[common_field_num + item_field_num + creative_field_num, self.senet_hidden_size],
                initializer=tf.glorot_uniform_initializer(), name="common_item_creative_weight_in")
            weight_creative_out = tf.get_variable(shape=[self.senet_hidden_size, creative_field_num],
                                                  initializer=tf.glorot_uniform_initializer(),
                                                  name="creative_weight_out")

            common_embedding_ = tf.reshape(common_embedding, [-1, common_field_num, self.embedding_size])
            common_mean = tf.reduce_mean(common_embedding_, axis=-1)
            A0_common = tf.matmul(common_mean, weight_common_in)
            if self.use_senet_bn:
                A0_common = ModelBase.batch_norm_layer_v2(x=A0_common, train_phase=is_train,
                                                          scope_bn='bn_input_common',
                                                          batch_norm_decay=self.batch_norm_decay,
                                                          use_riemann_bn=self.use_riemann_bn,
                                                          export=export)
            A1_common = tf.nn.tanh(A0_common)
            A2_common = tf.matmul(A1_common, weight_common_out)
            A3_common = 2 * tf.nn.sigmoid(A2_common)

            common_excitation_out = tf.expand_dims(A3_common, axis=2)
            common_reweight_out = tf.multiply(common_embedding_, common_excitation_out)
            common_reweight_out = tf.reshape(common_reweight_out, [-1, common_embedding.shape[-1].value])

            item_embedding_ = tf.reshape(item_embedding, [-1, item_field_num, self.embedding_size])
            item_mean = tf.reduce_mean(item_embedding_, axis=-1)
            item_common_mean = tf.concat([common_mean, item_mean], axis=-1)
            A0_item = tf.matmul(item_common_mean, weight_item_common_in)
            if self.use_senet_bn:
                A0_item = ModelBase.batch_norm_layer_v2(x=A0_item, train_phase=is_train,
                                                        scope_bn='bn_input_item',
                                                        batch_norm_decay=self.batch_norm_decay,
                                                        use_riemann_bn=self.use_riemann_bn,
                                                        export=export)
            A1_item = tf.nn.tanh(A0_item)
            A2_item = tf.matmul(A1_item, weight_item_out)
            A3_item = 2 * tf.nn.sigmoid(A2_item)

            item_excitation_out = tf.expand_dims(A3_item, axis=2)
            item_reweight_out = tf.multiply(item_embedding_, item_excitation_out)
            item_reweight_out = tf.reshape(item_reweight_out, [-1, item_embedding.shape[-1].value])

            creative_embedding_ = tf.reshape(creative_embedding, [-1, creative_field_num, self.embedding_size])
            creative_mean = tf.reduce_mean(creative_embedding_, axis=-1)
            all_mean = tf.concat([item_common_mean, creative_mean], axis=-1)
            A0_creative = tf.matmul(all_mean, weight_all_in)
            if self.use_senet_bn:
                A0_creative = ModelBase.batch_norm_layer_v2(x=A0_creative, train_phase=is_train,
                                                            scope_bn='bn_input_creative',
                                                            batch_norm_decay=self.batch_norm_decay,
                                                            use_riemann_bn=self.use_riemann_bn,
                                                            export=export)
            A1_creative = tf.nn.tanh(A0_creative)
            A2_creative = tf.matmul(A1_creative, weight_creative_out)
            A3_creative = 2 * tf.nn.sigmoid(A2_creative)

            creative_excitation_out = tf.expand_dims(A3_creative, axis=2)
            creative_reweight_out = tf.multiply(creative_embedding_, creative_excitation_out)
            creative_reweight_out = tf.reshape(creative_reweight_out, [-1, creative_embedding.shape[-1].value])
        return common_reweight_out, item_reweight_out, creative_reweight_out

    def _validate_architecture_config(self, kwargs):
        unsupported_buckets = {
            "coupon": getattr(self.fea_conf_obj, "coupon_fea_map", {}),
            "dense": getattr(self.fea_conf_obj, "dense_fea_map", {}),
            "sequence": getattr(self.fea_conf_obj, "seq_fea_map", {}),
            "gattr": getattr(self.fea_conf_obj, "gattr_fea_map", {}),
            "din": getattr(self.fea_conf_obj, "din_fea_map", {}),
        }
        nonempty_unsupported = {
            name: len(mapping)
            for name, mapping in unsupported_buckets.items()
            if mapping
        }
        if nonempty_unsupported:
            raise ValueError(
                "UniMixer v1 accepts only common/item/creative; "
                "non-empty extra buckets: {}".format(nonempty_unsupported)
            )

        # v1 固化 T=32、D=512，保证实验名和真实建图不会漂移。
        if self.um_token_num != self._TOKEN_NUM:
            raise ValueError(
                "UniMixer v1 requires um_token_num={}, got {}".format(
                    self._TOKEN_NUM,
                    self.um_token_num,
                )
            )
        if self.um_token_dim != self._TOKEN_DIM:
            raise ValueError(
                "UniMixer v1 requires um_token_dim={}, got {}".format(
                    self._TOKEN_DIM,
                    self.um_token_dim,
                )
            )
        if self.um_num_blocks <= 0:
            raise ValueError("um_num_blocks must be positive")
        if self.um_block_size <= 0:
            raise ValueError("um_block_size must be positive")
        flat_dim = self.um_token_num * self.um_token_dim
        if flat_dim % self.um_block_size != 0:
            raise ValueError(
                "T*D={} must be divisible by um_block_size={}".format(
                    flat_dim, self.um_block_size
                )
            )
        global_block_num = flat_dim // self.um_block_size
        if self.um_rank <= 0 or self.um_rank > global_block_num:
            raise ValueError(
                "um_rank must be in [1, {}], got {}".format(
                    global_block_num, self.um_rank
                )
            )
        if self.um_num_bases <= 0:
            raise ValueError("um_num_bases must be positive")
        if self.um_swiglu_expansion <= 0:
            raise ValueError("um_swiglu_expansion must be positive")
        if not 0.0 < self.um_tau_min <= self.um_tau_max:
            raise ValueError(
                "tau must satisfy 0 < min <= max, got min={}, max={}".format(
                    self.um_tau_min, self.um_tau_max
                )
            )
        if self.um_tau_decay_steps <= 0:
            raise ValueError("um_tau_decay_steps must be positive")
        if self.um_tau_schedule not in (
                "linear", "exponential", "cosine", "constant"):
            raise ValueError(
                "unsupported um_tau_schedule={}".format(
                    self.um_tau_schedule
                )
            )
        if self.um_sinkhorn_iters <= 0:
            raise ValueError("um_sinkhorn_iters must be positive")
        if self.um_global_step_base < 0:
            raise ValueError("um_global_step_base must be non-negative")
        if self.um_rms_epsilon <= 0.0:
            raise ValueError("um_rms_epsilon must be positive")
        if not self.um_use_token_bn:
            raise ValueError(
                "UniMixer v1 requires um_use_token_bn=true: each semantic "
                "token owns independent BN statistics before UniMixer"
            )

        configured_activation = kwargs.get("um_token_projection_activation", "linear")
        if configured_activation not in (None, "identity", "linear"):
            raise ValueError(
                "v1 uses the paper token-specific linear projection; "
                "um_token_projection_activation must be linear"
            )

    def _project_semantic_group(
            self,
            field_tensors,
            bucket_name,
            group_name,
            token_index,
            is_train,
            export):
        token_input = (
            field_tensors[0]
            if len(field_tensors) == 1
            else tf.concat(field_tensors, axis=-1)
        )
        input_dim = token_input.shape[-1].value
        if input_dim is None:
            raise ValueError(
                "semantic token input dimension must be statically known"
            )

        with tf.variable_scope(
                group_name,
                reuse=tf.AUTO_REUSE,
                partitioner=self.partitioner):
            projected = tf.contrib.layers.fully_connected(
                inputs=token_input,
                num_outputs=self.um_token_dim,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(
                    stddev=1.0 / math.sqrt(input_dim)
                ),
                weights_regularizer=tf.contrib.layers.l2_regularizer(
                    self.l2_deep
                ),
                biases_initializer=tf.zeros_initializer(),
                scope="projection",
            )
            # 每个语义组处于独立 variable_scope，因此这里会创建 32 套互不
            # 共享的 gamma/beta/moving_mean/moving_variance。输入是 [B, D]，
            # 统计量只沿 batch 维计算，不会混合 token 轴。
            projected = ModelBase.batch_norm_layer_v2(
                x=projected,
                train_phase=is_train,
                scope_bn="token_bn",
                batch_norm_decay=self.batch_norm_decay,
                use_riemann_bn=self.use_riemann_bn,
                export=export,
            )
            projected = tf.identity(projected, name="token_bn_output")

        logging.info(
            "UniMixer v1 token %02d %s/%s: fields=%d, "
            "input_dim=%d -> Linear/BN D=%d",
            token_index,
            bucket_name,
            group_name,
            len(field_tensors),
            input_dim,
            self.um_token_dim,
        )
        return projected

    def _semantic_tokenize(self, bucket_field_maps, is_train, export):
        tokens = []
        token_index = 0
        with tf.variable_scope(
                "um_semantic_tokenize",
                reuse=tf.AUTO_REUSE,
                partitioner=self.partitioner):
            for bucket_name in self._BUCKET_NAMES:
                groups = self.um_semantic_feature_groups[bucket_name]
                for group_name, feature_ids in groups:
                    group_tensors = [
                        bucket_field_maps[bucket_name][feature_id]
                        for feature_id in feature_ids
                    ]
                    tokens.append(
                        self._project_semantic_group(
                            group_tensors,
                            bucket_name,
                            group_name,
                            token_index,
                            is_train,
                            export,
                        )
                    )
                    token_index += 1

        stacked = tf.stack(tokens, axis=1, name="semantic_tokens")
        stacked.set_shape([None, self.um_token_num, self.um_token_dim])
        tf.add_to_collection("unimixer_v1_semantic_tokens", stacked)
        return stacked

    def _collect_bucket_embeddings(self, sparse_embeddings):
        bucket_embedding_maps = {
            name: {} for name in self._BUCKET_NAMES
        }
        columns = (
            self.features.lookup_nonseq_columns + self.features.seq_columns
        )
        for index, column in enumerate(columns):
            feature_id = get_sparse_fc_key(column)
            bucket_name = None
            if feature_id in self.fea_conf_obj.common_fea_map:
                bucket_name = "common"
            elif feature_id in self.fea_conf_obj.item_fea_map:
                bucket_name = "item"
            elif feature_id in self.fea_conf_obj.creative_fea_map:
                bucket_name = "creative"

            if bucket_name is None:
                continue
            if feature_id in bucket_embedding_maps[bucket_name]:
                raise ValueError(
                    "duplicated lookup embedding for feature {}".format(
                        feature_id
                    )
                )
            bucket_embedding_maps[bucket_name][feature_id] = sparse_embeddings[index]

        bucket_feature_ids = {
            "common": list(self.fea_conf_obj.common_fea_map.keys()),
            "item": list(self.fea_conf_obj.item_fea_map.keys()),
            "creative": list(self.fea_conf_obj.creative_fea_map.keys()),
        }
        bucket_embeddings = {}
        for bucket_name in self._BUCKET_NAMES:
            missing_ids = [
                feature_id
                for feature_id in bucket_feature_ids[bucket_name]
                if feature_id not in bucket_embedding_maps[bucket_name]
            ]
            if missing_ids:
                raise ValueError(
                    "lookup embeddings missing for {}: {}".format(
                        bucket_name, missing_ids
                    )
                )
            bucket_embeddings[bucket_name] = [
                bucket_embedding_maps[bucket_name][feature_id]
                for feature_id in bucket_feature_ids[bucket_name]
            ]

        return bucket_feature_ids, bucket_embeddings

    def model_fn(
            self,
            features,
            labels,
            timestamps=None,
            mode="train",
            export=False):
        del timestamps
        variable_partitions = self.num_ps
        if self.max_partitions is not None:
            variable_partitions = min(variable_partitions, self.max_partitions)
        self.partitioner = tf.min_max_variable_partitioner(
            max_partitions=variable_partitions,
            min_slice_size=1024000,
        )
        is_train = mode == "train"
        ps_mode = (
            "predict"
            if self.ps_stage == "join" and is_train
            else mode
        )

        sparse_embeddings = lookup_utils.flood_lookup_psv2(
            features=features,
            non_seq_columns=self.features.lookup_nonseq_columns,
            seq_columns=self.features.seq_columns,
            batch_size=self.batch_size,
            mode=ps_mode,
            clicks=tf.cast(labels, tf.float32),
            no_update_fea_names=list(self.fea_conf_obj.const_fea_map.keys()),
        )
        bucket_feature_ids, bucket_embeddings = self._collect_bucket_embeddings(
            sparse_embeddings
        )
        self.dnn_input_map = {
            bucket_name: tf.concat(
                bucket_embeddings[bucket_name],
                axis=-1,
            )
            for bucket_name in self._BUCKET_NAMES
        }

        with tf.variable_scope(
                "Cvr-task-part",
                reuse=tf.AUTO_REUSE,
                partitioner=self.partitioner):
            normalized_buckets = {}
            for bucket_name in self._BUCKET_NAMES:
                normalized_buckets[bucket_name] = ModelBase.batch_norm_layer_v2(
                    x=self.dnn_input_map[bucket_name],
                    train_phase=is_train,
                    scope_bn="bn_input_{}".format(bucket_name),
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
                bucket_tensors = dict(
                    zip(self._BUCKET_NAMES, gated_buckets)
                )
            else:
                bucket_tensors = normalized_buckets

            # BN/SENet 后按原始 field 维度无损切回，语义组投影不会绕过
            # base 的输入稳定化处理。
            bucket_field_maps = {}
            for bucket_name in self._BUCKET_NAMES:
                field_dims = [
                    tensor.shape[-1].value
                    for tensor in bucket_embeddings[bucket_name]
                ]
                if any(dim is None for dim in field_dims):
                    raise ValueError(
                        "all field embedding dimensions must be statically known"
                    )
                field_tensors = tf.split(
                    bucket_tensors[bucket_name],
                    field_dims,
                    axis=-1,
                )
                feature_ids = bucket_feature_ids[bucket_name]
                if len(field_tensors) != len(feature_ids):
                    raise ValueError(
                        "field tensor count mismatch for {}".format(
                            bucket_name
                        )
                    )
                bucket_field_maps[bucket_name] = dict(
                    zip(feature_ids, field_tensors)
                )

            input_tokens = self._semantic_tokenize(
                bucket_field_maps,
                is_train,
                export,
            )
            output_tokens = semantic_unimixer_stack(
                input_tokens=input_tokens,
                num_blocks=self.um_num_blocks,
                partitioner=self.partitioner,
                global_step=self.global_step,
                block_size=self.um_block_size,
                rank=self.um_rank,
                num_bases=self.um_num_bases,
                swiglu_expansion=self.um_swiglu_expansion,
                tau_max=self.um_tau_max,
                tau_min=self.um_tau_min,
                tau_decay_steps=self.um_tau_decay_steps,
                tau_decay_rate=self.um_tau_decay_rate,
                tau_schedule=self.um_tau_schedule,
                sinkhorn_iters=self.um_sinkhorn_iters,
                global_step_base=self.um_global_step_base,
                rms_epsilon=self.um_rms_epsilon,
            )

            flat_context = tf.reshape(
                output_tokens,
                [-1, self.um_token_num * self.um_token_dim],
                name="um_flatten_readout",
            )
            with tf.variable_scope(
                    "um_task_tower",
                    reuse=tf.AUTO_REUSE,
                    partitioner=self.partitioner):
                tower_output = self.mlp_tower(
                    flat_context,
                    self.cvr_layers,
                    is_train,
                    export,
                )
                with tf.device("/job:ps/task:0"):
                    output = tf.contrib.layers.fully_connected(
                        inputs=tower_output,
                        num_outputs=1,
                        activation_fn=tf.identity,
                        weights_initializer=tf.random_normal_initializer(
                            stddev=1.0 / math.sqrt(
                                tower_output.shape[-1].value
                            )
                        ),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(
                            self.l2_deep
                        ),
                        scope="deep_out",
                    )

            logits = tf.reshape(output, shape=[-1], name=mode)
            logits = tf.clip_by_value(
                logits,
                -self.clip_val,
                self.clip_val,
            )
            predictions = tf.sigmoid(logits, name=mode)

        logging.info(
            "UniMixer v1 output: input_tokens=%s, output_tokens=%s, "
            "flat_context=%s, tower_output=%s",
            input_tokens.get_shape(),
            output_tokens.get_shape(),
            flat_context.get_shape(),
            tower_output.get_shape(),
        )
        return {"logits": logits, "pred": predictions}

    def output_layer(self, inputs, scope_name, partitioner, units=1):
        with tf.variable_scope(scope_name, partitioner=partitioner, reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(inputs, units, kernel_initializer=tf.glorot_normal_initializer(),
                                     bias_initializer=tf.constant_initializer(0.00001))
            logits = tf.clip_by_value(logits, -self.clip_val, self.clip_val)
        return logits, tf.nn.sigmoid(logits)

    def mlp_tower(self, input_tensor, layers, is_train, export):
        logging.info(f"mlp input shape is: {input_tensor.get_shape()}")
        for i, layer_size in enumerate(layers):
            logging.info(f'Cvr-task-part: layer %s-%s', i, layer_size)
            input_tensor = tf.contrib.layers.fully_connected(inputs=input_tensor, num_outputs=layer_size,
                                                             activation_fn=None,
                                                             weights_initializer=tf.random_normal_initializer(
                                                                 stddev=1 / math.sqrt(
                                                                     input_tensor.shape[1].value)),
                                                             weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                 self.l2_deep),
                                                             scope=f'mlp{i}')

            if self.batch_norm:
                logging.info('used mlp bn')
                input_tensor = ModelBase.batch_norm_layer_v2(x=input_tensor, train_phase=is_train,
                                                             scope_bn=f'bn_{i}',
                                                             batch_norm_decay=self.batch_norm_decay,
                                                             use_riemann_bn=self.use_riemann_bn,
                                                             export=export)
            if self.mlp_act_type == 'prelu':
                input_tensor = self.prelu(input_tensor, name=f'mlp{i}')
            elif self.mlp_act_type == 'swish':
                beta = tf.get_variable(name='swish_beta_%d' % (i), dtype=tf.float32, shape=input_tensor.shape[1],
                                       initializer=tf.constant_initializer(1.702))
                logging.info('swish_beta_{}: {}'.format(i, beta))
                input_tensor = input_tensor * tf.nn.sigmoid(beta * input_tensor)
            else:
                input_tensor = self.get_act_func(self.mlp_act_type)(input_tensor)

            if self.use_mlp_gate:
                # gate 结构
                input_tensor_gate = tf.contrib.layers.fully_connected(inputs=input_tensor,
                                                                      num_outputs=layer_size,
                                                                      activation_fn=tf.identity,
                                                                      weights_initializer=tf.random_normal_initializer(
                                                                          stddev=1 / math.sqrt(
                                                                              input_tensor.shape[
                                                                                  1].value)),
                                                                      weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                          self.l2_deep),
                                                                      scope=f'mlp{i}_gate')
                input_tensor_gate = ModelBase.batch_norm_layer_v2(x=input_tensor_gate, train_phase=is_train,
                                                                  scope_bn=f'mlp{i}_gate_bn_input',
                                                                  batch_norm_decay=self.batch_norm_decay,
                                                                  use_riemann_bn=self.use_riemann_bn,
                                                                  export=export)
                input_tensor_gate = tf.sigmoid(input_tensor_gate) * 2
                input_tensor = input_tensor_gate * input_tensor

        return input_tensor

    def train(self, session, worker_id=0, **kwargs):
        """执行训练步骤（
        Args:
            session: TensorFlow会话
            worker_id: 工作节点ID
            **kwargs: 其他参数

        Returns:
            包含全局步数和重置次数的字典
        """

        self.train_count += 1
        if self.enable_delay_train_mode:
            fetch = {
                'train_op': self.train_op,
                'loss': self.loss,
                'labels_pos_cvr_count': self.labels_pos_cvr_count_delay,
                'global_step': self.global_step,
                'pred_mean': self['train_pred_mean_delay'],
                'auc': self['train_auc_delay'],
                'copc': self['train_copc_delay'],
                'learning_rate': self.learning_rate
            }
        else:
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
            if self.enable_last_cvr:
                fetch.update({
                    # last
                    'pred_mean_last': self['train_pred_mean_last'],
                    'auc_last': self['train_auc_last'],
                    'copc_last': self['train_copc_last'],
                    'loss_last': self.loss_last,
                    'labels_last_pos_count': self.labels_last_pos_count,
                })
        if self.enable_wide_cvr:
            fetch.update({
                'pred_mean_wide': self['train_pred_mean_wide'],
                'auc_wide': self['train_auc_wide'],
                'copc_wide': self['train_copc_wide'],
                'loss_wide': self.loss_wide,
                'labels_wide_pos_count': self.labels_wide_pos_count,
            })

        if self.enable_mlt_loss:
            fetch.update({
                'cvr_loss': self.cvr_loss,
                'mlt_loss': self.mlt_loss,
                'label_is_clk_buy_all': self.label_is_clk_buy_all,
            })

        res = session.run(fetch, options=self.timeout_options)

        if self.train_count % kwargs.get('train_log_step', 10) == 0:
            logging.info(f"----------------- train [{self.train_count}] ------------------------")
            logging.info(
                f"lstep: {self['train_count']}, "
                f"gstep: {res['global_step']}, "
                f"loss: {res['loss']:.6f}, "
                f"auc: {res['auc']:.6f}, "
                f"copc: {res['copc']:.6f}, "
                f"pred_mean: {res['pred_mean']:.6f},"
                f"labels_pos_cvr_count: {res['labels_pos_cvr_count']},"
                f"learning_rate:  {res['learning_rate']},"
            )

            if self.enable_last_cvr:
                logging.info(
                    f"loss_last: {res['loss_last']:.6f}, "
                    f"auc_last: {res['auc_last']:.6f}, "
                    f"copc_last: {res['copc_last']:.6f}, "
                    f"pred_mean_last: {res['pred_mean_last']:.6f},"
                    f"labels_last_pos_count: {res['labels_last_pos_count']},"
                )

            if self.enable_mlt_loss:
                logging.info(
                    f"label_is_clk_buy_all: {res['label_is_clk_buy_all']}, "
                    f"cvr_loss: {res['cvr_loss']}, "
                    f"mlt_loss: {res['mlt_loss']}"
                )

            if self.enable_wide_cvr:
                logging.info(
                    f"loss_wide: {res['loss_wide']:.6f}, "
                    f"auc_wide: {res['auc_wide']:.6f}, "
                    f"copc_wide: {res['copc_wide']:.6f}, "
                    f"pred_mean_wide: {res['pred_mean_wide']:.6f},"
                    f"labels_wide_pos_count: {res['labels_wide_pos_count']},"
                )
            logging.info("-------------------------------------------------------------")

        # In Warm start mode, initial 'global_step' could be a very large num,
        # so use `train_count` instead to track `reset_auc_op`.
        if self.task_index == 1 and self.train_reset_interval > 0 \
                and self.train_count * self.num_worker > self.train_reset_interval * self.train_reset_count:
            self.train_reset_count += 1
            logging.info(" >>>> reset auc <<<< ")
            session.run([self['train_reset_auc_op']])
        return {'global_step': res['global_step'], 'train_reset_count': self.train_reset_count}

    def test(self, session, worker_id=0, prefix='test', **kwargs):
        """测试操作
        Args:
            session: TensorFlow会话
            worker_id: 工作节点ID
            prefix: 测试前缀
            **kwargs: 其他参数

        Returns:
            包含测试结果的字典
        """
        # 1. 初始化环境
        self.train_init(session)
        FORMAT = '%(asctime)-15s [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s'
        file_handler = FileHandler('flood_worker_0.log')
        file_handler.setFormatter(Formatter(FORMAT))

        logger = getLogger(name='search_jarvis_logging')
        logger.addHandler(file_handler)

        # 2. 初始化测试状态
        test_cnt = 0
        session.run([self['test_init_op']])

        # 3. 指标收集器初始化
        auc_accum = RocAucAccum(num_thresholds=2000)
        pr_auc_accum = PrAucAccum(num_thresholds=2000)
        copc_accum = COPCAccum()
        bucket_error = BucketErrorAccum(1000)
        sample_cnt_accum = SampleCntAccum()
        auc_accum_wide = RocAucAccum(num_thresholds=2000)
        copc_accum_wide = COPCAccum()

        auc_accum_last = RocAucAccum(num_thresholds=2000)
        pr_auc_accum_last = RocAucAccum(num_thresholds=2000)
        copc_accum_last = COPCAccum()

        # 4. 准备需要获取的张量
        if self.enable_delay_train_mode:
            fetchs = {
                'sampleid': self['test_sampleid'],
                'test_search_id': self['test_search_id'],
                'test_example_id': self['test_example_id'],
                'labels': self['test_labels_delay'],
                'pred': self['test_pred_delay'],
                'auc': self['test_auc_delay'],
                'copc': self['test_copc_delay'],
            }
        else:
            fetchs = {
                'sampleid': self['test_sampleid'],
                'test_search_id': self['test_search_id'],
                'test_example_id': self['test_example_id'],
                'labels': self['test_labels'],
                'pred': self['test_pred'],
                'auc': self['test_auc'],
                'copc': self['test_copc'],
            }
            if self.enable_last_cvr:
                fetchs.update({
                    # last
                    'labels_last': self['test_labels_last'],
                    'pred_last': self['test_pred_last'],
                    'auc_last': self['test_auc_last'],
                    'copc_last': self['test_copc_last']
                })

        if self.enable_wide_cvr:
            fetchs.update({
                'labels_wide': self['test_labels_wide'],
                'pred_wide': self['test_pred_wide'],
                'auc_wide': self['test_auc_wide'],
                'copc_wide': self['test_copc_wide']
            })

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

        # 6. 执行测试循环
        while True:
            try:
                res = session.run(fetchs, options=self.timeout_options)
                label_cvr, pred = res['labels'], res['pred']
                test_cnt += 1

                # 更新指标
                auc_accum.update(label_cvr, pred)
                pr_auc_accum.update(label_cvr, pred)
                copc_accum.update(label_cvr, pred)
                bucket_error.update(label_cvr, pred)
                sample_cnt_accum.update(label_cvr, pred)
                # last
                if self.enable_last_cvr:
                    label_cvr_last, pred_last = res['labels_last'], res['pred_last']
                    auc_accum_last.update(label_cvr_last, pred_last)
                    pr_auc_accum_last.update(label_cvr_last, pred_last)
                    copc_accum_last.update(label_cvr_last, pred_last)
                if self.enable_wide_cvr:
                    label_cvr_wide, pred_wide = res['labels_wide'], res['pred_wide']
                    auc_accum_wide.update(label_cvr_wide, pred_wide)
                    copc_accum_wide.update(label_cvr_wide, pred_wide)

                if 0 < self.test_batch_num < test_cnt:
                    logging.info(f"finish test by test_batch_num={self.test_batch_num}")
                    break

                # 记录日志
                if test_cnt % kwargs.get('test_log_step', 10) == 0:
                    logging.info("----------------- test_cnt [%s] ------------------------" % test_cnt)
                    logging.info(f"FIRST AUC: {res['auc']:.6f}  FIRST COPC: {res['copc']:.6f}")
                    if self.enable_last_cvr:
                        logging.info(f"LAST AUC: {res['auc_last']:.6f}  LAST COPC: {res['copc_last']:.6f}")

                    if self.enable_wide_cvr:
                        logging.info(f"Wide AUC: {res['auc_wide']:.6f}  Wide COPC: {res['copc_wide']:.6f}")

                # 保存预测结果
                if self.save_predict_result:
                    if self.enable_wide_cvr and self.enable_last_cvr:
                        with tf.gfile.Open(local_path, 'a') as f:
                            for search_id, example_id, label_cvr, pred, label_wide, pred_wide, label_last, pred_last in zip(
                                    res['test_search_id'], res['test_example_id'], res['labels'], res['pred'],
                                    res['labels_wide'], res['pred_wide'], res['labels_last'], res['pred_last']):
                                line = '\t'.join(
                                    [search_id.decode(), example_id.decode(), str(label_cvr[0]), str(pred),
                                     str(label_wide[0]), str(pred_wide), str(label_last[0]), str(pred_last)]) + '\n'
                                f.write(line)
                                cnt += 1
                    else:
                        with tf.gfile.Open(local_path, 'a') as f:
                            for search_id, example_id, label_cvr, pred in zip(res['test_search_id'],
                                                                              res['test_example_id'], res['labels'],
                                                                              res['pred']):
                                line = '\t'.join(
                                    [search_id.decode(), example_id.decode(), str(label_cvr[0]), str(pred)]) + '\n'
                                f.write(line)
                                cnt += 1


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

        if self.enable_last_cvr:
            accum_metrics['cvr-tower'].update({
                # last
                'roc_auc_last': auc_accum_last.dump(),
                'pr_auc_last': pr_auc_accum_last.dump(),
                'copc_last': copc_accum_last.dump()
            })

        if self.enable_wide_cvr:
            accum_metrics['cvr-tower'].update({
                'roc_auc_wide': auc_accum_wide.dump(),
                'copc_wide': copc_accum_wide.dump()
            })

        res = {'accum_metrics': accum_metrics,
               'title': f'random-feature-{self.random_feature}' if self.random_feature else 'base'}

        # upload predict res to hdfs path
        if self.save_predict_result:
            upload_hdfs(local_path, hdfs_path, True)
            logging.info("upload predict result into hdfs: %s", hdfs_path)

        # only chief worker uploads log
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

        if self.enable_last_cvr:
            self.export_spec['output'].update({'last_cvr': pred_result['pred_last']})

        if self.enable_wide_cvr:
            self.export_spec['output'].update({'sim_c3_wcvr': pred_result['pred_wide']})

        if self.enable_delay_train_mode:
            self.export_spec['output'].update({'cvr_delay': pred_result['pred_delay'] / (pred_result['pred_delay'] + (1 - pred_result['pred_delay']) / self.delay_train_negative_sample_rate)})

    def export(self):
        return self.export_spec

    def train_init(self, session):
        logging.info("reinitialize train_init_op.")
        session.run(self['train_init_op'])

        if self.is_chief:
            session.run(learning_rate_utils.get_or_create_milestone_step_reset_op())
            logging.info('milestone step: {}'.format(session.run(learning_rate_utils.get_or_create_milestone_step())))

    def evaluate(self, session, **kwargs):
        self.eval_count += 1
        fetches = {
            # flood 逻辑，需要保留
            'summary': self['eval_summary'],
            # 'auc_cvr': self['test_auc'],
            # 'copc_cvr': self['test_copc'],
            # 'loss': self.loss,
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
            # if self.warmup_type == 'two_model_multi_tmp':
            #     pass
            #     hooks.append(TwoModelMultiWarmupHook(self.model_dir, model=self))
            # elif self.warmup_type == 'warm_2_epoch':
            #     hooks.append((WarmSecondEpochHook(self.model_dir, model=self)))
            # else:
            #     hooks.append(Senet2NewWarmupHook(self.model_dir, model=self))
        return hooks

