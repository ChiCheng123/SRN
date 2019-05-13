#encoding: utf-8
from srn.utils import bbox_helper
from srn.utils import anchor_helper
import torch
from torch.autograd import Variable
import numpy as np
import logging

logger = logging.getLogger('global')

def to_np_array(x):
    if isinstance(x, Variable): x = x.data
    return x.cpu().numpy() if torch.is_tensor(x) else x

def compute_rpn_proposals(conv_cls_fs, conv_loc_fs, conv_cls_ss, conv_loc_ss,
                          multi_cls, multi_reg, cfg, image_info):
    '''
    :argument
        cfg: configs
        conv_cls: FloatTensor, [batch, num_anchors * num_classes, h, w], conv output of classification
        conv_loc: FloatTensor, [batch, num_anchors * 4, h, w], conv output of localization
        image_info: FloatTensor, [batch, 3], image size
    :returns
        proposals: Variable, [N, 7], 2-dim: batch_ix, x1, y1, x2, y2, score, label
    '''
    batch_size, num_anchors_num_classes, featmap_h, featmap_w = conv_cls_fs.shape
    # [K*A, 4]
    anchors_overplane = anchor_helper.get_anchors_over_plane(
        featmap_h, featmap_w, cfg['anchor_ratios'], cfg['anchor_scales'], cfg['anchor_stride'])
    B = batch_size
    A = num_anchors_num_classes // cfg['num_classes']
    assert(A * cfg['num_classes'] == num_anchors_num_classes)
    K = featmap_h * featmap_w
    cls_view_fs = conv_cls_fs.permute(0, 2, 3, 1).contiguous().view(B, K*A, cfg['num_classes']).cpu().numpy()
    loc_view_fs = conv_loc_fs.permute(0, 2, 3, 1).contiguous().view(B, K*A, 4).cpu().numpy()
    cls_view_ss = conv_cls_ss.permute(0, 2, 3, 1).contiguous().view(B, K * A, cfg['num_classes']).cpu().numpy()
    loc_view_ss = conv_loc_ss.permute(0, 2, 3, 1).contiguous().view(B, K * A, 4).cpu().numpy()

    if cfg['cls_loss_type'] == 'softmax_focal_loss':
        cls_view_fs = cls_view_fs[:, :, 1:]
        cls_view_ss = cls_view_ss[:, :, 1:]
    nmsed_bboxes = []
    pre_nms_top_n = cfg['top_n_per_level']
    thresh = cfg['score_thresh'] if K >= 120 else 0.0
    for b_ix in range(B):
        loc_delta_fs = loc_view_fs[b_ix, :, :]
        if multi_reg:
            anchors_overplane = bbox_helper.compute_loc_bboxes(anchors_overplane, loc_delta_fs)

        ka_ix_fs, cls_ix_fs = np.where(cls_view_fs[b_ix] > 0.01)
        ka_ix_ss, cls_ix_ss = np.where(cls_view_ss[b_ix] > thresh)
        if multi_cls:
            ka_ix = np.intersect1d(ka_ix_fs, ka_ix_ss)
        else:
            ka_ix = ka_ix_ss
        cls_ix = np.zeros_like(ka_ix)

        if ka_ix.size == 0:
            continue
        
        scores = cls_view_ss[b_ix, ka_ix, cls_ix]
        loc_delta_ss = loc_view_ss[b_ix, ka_ix, :]
        loc_anchors = anchors_overplane[ka_ix, :]

        if True or pre_nms_top_n <= 0 or pre_nms_top_n > scores.shape[0]:
            order = scores.argsort()[::-1][:pre_nms_top_n]
        else:
            inds = np.argpartition(-scores, pre_nms_top_n)[:pre_nms_top_n]
            order = np.argsort(-scores[inds])
            order = inds[order]

        scores = scores[order]
        cls_ix = cls_ix[order]
        cls_ix = cls_ix + 1
        loc_delta = loc_delta_ss[order]
        loc_anchors = loc_anchors[order]

        boxes = bbox_helper.compute_loc_bboxes(loc_anchors, loc_delta)

        batch_ix = np.full(boxes.shape[0], b_ix)
        post_bboxes = np.hstack([batch_ix[:, np.newaxis], boxes, scores[:, np.newaxis], cls_ix[:, np.newaxis]])
        nmsed_bboxes.append(post_bboxes)

    if len(nmsed_bboxes) > 0:    
        nmsed_bboxes = np.vstack(nmsed_bboxes)
    else:
        nmsed_bboxes = np.array([])
    return nmsed_bboxes
