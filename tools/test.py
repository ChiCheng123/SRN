from __future__ import division
# workaround of the bug where 'import torchvision' sets the start method to be 'fork'
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')

from srn.model.resnet_srn import resnet50
from srn.utils import bbox_helper
from srn.utils.log_helper import init_log
from srn.utils.load_helper import restore_from

import argparse
import logging
import os
import cv2
import math

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import json

parser = argparse.ArgumentParser(description='SRN Testing')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--results_dir', dest='results_dir', default='results_dir',
                    help='results dir of output for each class')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter in json format')
parser.add_argument('--img_list', dest='img_list', type=str,
                    help='meta file list for cross dataset validating')
parser.add_argument('--max_size', dest='max_size', type=int, required=True,
                    help='max test size')

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    for key in cfg.keys():
        if key != 'shared':
            cfg[key].update(cfg['shared'])
    return cfg


def main():
    init_log('global', logging.INFO)
    logger = logging.getLogger('global')
    global args
    args = parser.parse_args()
    cfg = load_config(args.config)

    model = resnet50(pretrained=False, cfg = cfg['shared'])
    print(model)

    # resume from the released model
    assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
    model = restore_from(model, args.resume)
    model = model.cuda()

    logger.info('build dataloader done')
    validate(model, cfg, args)


def validate(model, cfg, args):
    logger = logging.getLogger('global')

    # switch to evaluate mode
    model.eval()

    logger.info('start validate')
    if not os.path.exists(args.results_dir):
        try:
            os.makedirs(args.results_dir)
        except Exception as e:
            print(e)

    # define the largest input size
    largest_input = args.max_size * args.max_size

    f_list = open(args.img_list, 'r')
    path_set = f_list.readlines()
    for i in range(len(path_set)):
        proposals_b = np.zeros((1,7))
        img = cv2.imread(os.path.join('../', path_set[i][:-1]))
        img_var = preprocess(img)
        x = {
            'cfg': cfg,
            'image': img_var,
            'image_info': None,
            'ground_truth_bboxes': None,
            'ignore_regions': None
        }
        img_h = img.shape[0]
        img_w = img.shape[1]

        # multi-scale test
        if img_h * img_w < largest_input:
            proposals_o = model(x)['predict'][0].data.cpu().numpy()

            x['image'] = preprocess(cv2.flip(img, 1))
            proposals_f = bbox_flip(model(x)['predict'][0].data.cpu().numpy(), x['image'].data.cpu().numpy().shape[3])

            x['image'] = preprocess(cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
            proposals_s = bbox_resize(model(x)['predict'][0].data.cpu().numpy(), 0.5)
            index = np.where(np.maximum(proposals_s[:, 3] - proposals_s[:, 1] + 1, proposals_s[:, 4] - proposals_s[:, 2] + 1) > 30)[0]
            proposals_s = proposals_s[index, :]

            enlarge_time = int(math.floor(math.log(largest_input / img_w / img_h, 2.25)))

            for t in range(enlarge_time):
                resize_scale = math.pow(1.5, t+1)
                x['image'] = preprocess(cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale))
                try:
                    proposals_b = np.vstack((proposals_b, bbox_resize(model(x)['predict'][0].data.cpu().numpy(), resize_scale)))
                except:
                    proposals_b = bbox_resize(model(x)['predict'][0].data.cpu().numpy(), resize_scale)

            final_ratio = math.sqrt(largest_input / img_h / img_w)
            x['image'] = preprocess(cv2.resize(img, (0, 0), fx=final_ratio, fy=final_ratio))
            try:
                proposals_b = np.vstack((proposals_b, bbox_resize(model(x)['predict'][0].data.cpu().numpy(), final_ratio)))
            except:
                proposals_b = bbox_resize(model(x)['predict'][0].data.cpu().numpy(), final_ratio)
            index = np.where(np.minimum(proposals_b[:, 3] - proposals_b[:, 1] + 1, proposals_b[:, 4] - proposals_b[:, 2] + 1) < 100)[0]
            proposals_b = proposals_b[index, :]
        else:
            largest_ratio = math.sqrt(largest_input / img_w / img_h)
            largest_img = cv2.resize(img, (0, 0), fx=largest_ratio, fy=largest_ratio)
            x['image'] = preprocess(largest_img)
            proposals_o = bbox_resize(model(x)['predict'][0].data.cpu().numpy(), largest_ratio)

            x['image'] = preprocess(cv2.flip(largest_img, 1))
            proposals_f = bbox_resize(bbox_flip(model(x)['predict'][0].data.cpu().numpy(), largest_img.shape[1]), largest_ratio)

            x['image'] = preprocess(cv2.resize(largest_img, (0, 0), fx=0.75, fy=0.75))
            proposals_s = bbox_resize(model(x)['predict'][0].data.cpu().numpy(), largest_ratio * 0.75)

            x['image'] = preprocess(cv2.resize(largest_img, (0, 0), fx=0.5, fy=0.5))
            try:
                proposals_s = np.vstack((proposals_s, bbox_resize(model(x)['predict'][0].data.cpu().numpy(), largest_ratio * 0.5)))
            except:
                proposals_s = bbox_resize(model(x)['predict'][0].data.cpu().numpy(), largest_ratio * 0.5)
            index = np.where(np.maximum(proposals_s[:, 3] - proposals_s[:, 1] + 1,
                                        proposals_s[:, 4] - proposals_s[:, 2] + 1) > 30)[0]
            proposals_s = proposals_s[index, :]
        proposals = np.vstack((proposals_o, proposals_f, proposals_s, proposals_b))

        proposals_vote = bbox_vote(proposals)
        proposals_vote = np.hstack((np.zeros((proposals_vote.shape[0], 1)), proposals_vote, np.ones((proposals_vote.shape[0], 1))))

        # bbox clip
        dts_per_image_vote = bbox_helper.clip_bbox(proposals_vote[:, 1:-1], [img_h, img_w])
        dts_per_image_vote = dts_per_image_vote[dts_per_image_vote[:, 4] != 0]

        write_wider_result(path_set[i][:-1], dts_per_image_vote, args.results_dir)

        logger.info('Test progress: [{0} / {1}]'.format(i, len(path_set)))


def write_wider_result(img_dir, dts, output_path):
    img_cls_label = img_dir.split('/')[-2]
    output_dir = os.path.join(output_path, img_cls_label)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, img_dir.split('/')[-1].replace('jpg', 'txt')), 'w') as f:
        f.write(img_dir.split('/')[-1].strip('.jpg') + '\n')
        f.write(str(dts.shape[0]) + '\n')
        for i in range(dts.shape[0]):
            f.write('{} {} {} {} {}\n'.format(round(float(dts[i][0])),
                                         round(float(dts[i][1])),
                                         round(float(dts[i][2] - dts[i][0] + 1)),
                                         round(float(dts[i][3] - dts[i][1] + 1)),
                                         round(float(dts[i][4]), 3)))


def bbox_flip(proposal, img_width):
    proposal[:, 1], proposal[:, 3] = img_width - proposal[:, 3] - 1, img_width - proposal[:, 1] - 1
    return proposal


def bbox_resize(proposal, resize_scale):
    proposal[:, 1:5] = (proposal[:, 1:5] - (resize_scale - 1) / 2.0) / resize_scale
    return proposal


def preprocess(img):
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = totensor(img)
    img_tensor = normalize(img_tensor)
    return torch.autograd.Variable(img_tensor.unsqueeze(0).cuda(async=True))


def bbox_vote(det):
    order = det[:, 5].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 3] - det[:, 1] + 1) * (det[:, 4] - det[:, 2] + 1)
        xx1 = np.maximum(det[0, 1], det[:, 1])
        yy1 = np.maximum(det[0, 2], det[:, 2])
        xx2 = np.minimum(det[0, 3], det[:, 3])
        yy2 = np.minimum(det[0, 4], det[:, 4])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.5)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 1:5] = det_accu[:, 1:5] * np.tile(det_accu[:, -2:-1], (1, 4))
        max_score = np.max(det_accu[:, 5])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 1:5], axis=0) / np.sum(det_accu[:, -2:-1])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets


if __name__ == '__main__':
    main()


