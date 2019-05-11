#encoding:utf-8

from .rpn_proposal import compute_rpn_proposals
from torch.autograd import Variable
import functools
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from srn.extensions import nms
import numpy as np

logger = logging.getLogger('global')

class SRN(nn.Module):
    def __init__(self):
        super(SRN, self).__init__()

    def feature_extractor(self, x):
        raise NotImplementedError

    def rpn(self, x):
        raise NotImplementedError

    def _multi_level_cfg(self, cfg, fpn_level):
        cfgs = []
        for i in range(fpn_level):
            tmp_cfg = cfg.copy()
            tmp_cfg['anchor_stride'] *= (2**i)
            cfgs.append(tmp_cfg)
        return cfgs

    def _pin_args_to_fn(self, cfg, image_info, fpn_level):
        partial_fn = {}
        partial_fn['rpn_proposal_fn'] = []
        multi_rpn_proposal_cfgs = self._multi_level_cfg(cfg['test_rpn_proposal_cfg'], fpn_level)
        for level in range(fpn_level):
            partial_fn['rpn_proposal_fn'].append(functools.partial(
                compute_rpn_proposals,
                cfg=multi_rpn_proposal_cfgs[level],
                image_info=image_info))
        return partial_fn

    def _multi_proposals(self, rpn_pred_fs,  rpn_pred_ss, multi_rpn_proposal_fn, fpn_level, topN,
                         num_classes, batch_size, iou, multi_cls_levels, multi_reg_levels):
        proposals_all = []
        for level in range(fpn_level):
            compute_rpn_proposals_fn = multi_rpn_proposal_fn[level]
            rpn_pred_cls_fs = rpn_pred_fs[level][0]
            rpn_pred_loc_fs = rpn_pred_fs[level][1]
            rpn_pred_cls_ss = rpn_pred_ss[level][0]
            rpn_pred_loc_ss = rpn_pred_ss[level][1]
            rpn_pred_cls_fs = F.sigmoid(rpn_pred_cls_fs)
            rpn_pred_cls_ss = F.sigmoid(rpn_pred_cls_ss)
            cls_flag = True
            reg_flag = True
            if level not in multi_cls_levels:
                cls_flag = False
            if level not in multi_reg_levels:
                reg_flag = False
            proposals = compute_rpn_proposals_fn(rpn_pred_cls_fs.data, rpn_pred_loc_fs.data,
                                                 rpn_pred_cls_ss.data, rpn_pred_loc_ss.data,
                                                 cls_flag, reg_flag)
            if proposals.size > 0:
                proposals_all.append(proposals)
        if len(proposals_all) > 0:
            proposals = np.vstack(proposals_all)
            all_bboxes = []
            for b_ix in range(batch_size):
                bboxes = []
                bps = proposals[proposals[:, 0] == b_ix]
                for cls in range(1, num_classes + 1):
                    bpcs = bps[bps[:, -1] == cls]
                    if bpcs.size <= 0:
                        continue
                    order = bpcs[:, -2].argsort()[::-1]
                    pre_bboxes = bpcs[order, :]
                    keep_index = nms(torch.from_numpy(pre_bboxes[:, 1:-1]).float().cuda(), iou).numpy()
                    post_bboxes = pre_bboxes[keep_index[:750]]
                    bboxes.append(post_bboxes)
                bboxes = np.vstack(bboxes)
                order = bboxes[:, -2].argsort()[::-1][:topN]
                bboxes = bboxes[order, :]
                all_bboxes.append(Variable(torch.from_numpy(bboxes)).float())
            proposals = torch.cat(all_bboxes, 0)
        else:
            proposals = Variable(torch.from_numpy(np.zeros((1, 7)))).float().cuda()
        return proposals

    def forward(self, input):
        cfg = input['cfg']
        x = input['image']
        ground_truth_bboxes = input['ground_truth_bboxes']
        image_info = input['image_info']
        outputs = {'losses': [], 'predict': [], 'accuracy': [], 'accuracy_neg':[]}
        rpn_pred_fs = []
        rpn_pred_ss = []
        pyramid_features_fs, pyramid_features_ss = self.feature_extractor(x)
        fpn_level = len(pyramid_features_fs)
        for feature in pyramid_features_fs:
            rpn_pred_fs.append(self.rpn(feature))
        for feature in pyramid_features_ss:
            rpn_pred_ss.append(self.rpn(feature))
        partial_fn = self._pin_args_to_fn(
                cfg,
                image_info,
                fpn_level)

        proposals = self._multi_proposals(rpn_pred_fs,
                                          rpn_pred_ss,
                                          partial_fn['rpn_proposal_fn'],
                                          fpn_level,
                                          cfg['test_rpn_proposal_cfg']['top_n'],
                                          cfg['shared']['num_classes'],
                                          ground_truth_bboxes.shape[0] if ground_truth_bboxes is not None else 1,
                                          cfg['test_rpn_proposal_cfg']['nms_iou_thresh'],
                                          cfg['shared']['multi_cls_levels'],
                                          cfg['shared']['multi_reg_levels'])
        outputs['predict'].append(proposals.cuda())
        return outputs

