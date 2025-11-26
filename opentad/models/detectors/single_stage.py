import torch
from ..builder import DETECTORS, build_backbone, build_projection, build_head, build_neck
from .base import BaseDetector
from ..utils.post_processing import batched_nms, convert_to_seconds
import numpy as np
from ..dense_heads.tridet_head import TriDetHead

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """
    Base class for single-stage detectors which should not have roi_extractors.
    """

    def __init__(self, backbone=None, projection=None, neck=None, rpn_head=None):
        super(SingleStageDetector, self).__init__()
        self.tri = False

        if backbone is not None:
            self.backbone = build_backbone(backbone)

        if projection is not None:
            self.projection = build_projection(projection)
            self.max_seq_len = self.projection.max_seq_len

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)

        # n_mha_win_size = self.projection.n_mha_win_size
        # if isinstance(n_mha_win_size, int):
        #     self.mha_win_size = [n_mha_win_size] * (1 + projection.arch[-1])
        # else:
        #     assert len(n_mha_win_size) == (1 + projection.arch[-1])
        #     self.mha_win_size = n_mha_win_size
        # max_div_factor = 1
        # for s, w in zip(rpn_head.prior_generator.strides, self.mha_win_size):
        #     stride = s * (w // 2) * 2 if w > 1 else s
        #     assert (
        #         self.max_seq_len % stride == 0
        #     ), f"max_seq_len {self.max_seq_len} must be divisible by fpn stride and window size {stride}"
        #     if max_div_factor < stride:
        #         max_div_factor = stride
        # self.max_div_factor = max_div_factor

    @property
    def with_backbone(self):
        """bool: whether the detector has backbone"""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_projection(self):
        """bool: whether the detector has projection"""
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_neck(self):
        """bool: whether the detector has neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_rpn_head(self):
        """bool: whether the detector has localization head"""
        if isinstance(self.rpn_head, TriDetHead):
            self.tri = True
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    def pad_data(self, inputs, masks):
        feat_len = inputs.shape[-1]
        if feat_len == self.max_seq_len:
            return inputs, masks
        elif feat_len < self.max_seq_len:
            max_len = self.max_seq_len
        else:  # feat_len > self.max_seq_len
            max_len = feat_len
            # pad the input to the next divisible size
            stride = self.max_div_factor
            max_len = (max_len + (stride - 1)) // stride * stride

        padding_size = [0, max_len - feat_len]
        inputs = torch.nn.functional.pad(inputs, padding_size, value=0)
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        pad_masks[:, :feat_len] = masks
        return inputs, pad_masks

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        # if not self.training:
        #     x, masks = self.pad_data(x, masks)

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            rpn_losses = self.rpn_head.forward_train(
                x,
                masks,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **kwargs,
            )
            losses.update(rpn_losses)

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs
        # x, masks = self.pad_data(x, masks)

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            if self.tri:
                points, rpn_reg, rpn_scores = self.rpn_head.forward_test(x, masks, **kwargs)
                predictions = points, rpn_reg, rpn_scores
            else:
                rpn_proposals, rpn_scores = self.rpn_head.forward_test(x, masks)
                predictions = rpn_proposals, rpn_scores
        else:
            rpn_proposals = rpn_scores = None
            predictions = rpn_proposals, rpn_scores

        return predictions

    # @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        if self.tri:
            points, rpn_reg, rpn_scores = predictions  # [N, 4], [B, num_classes, N, 2], [B, N, num_classes]

            pre_nms_thresh = 0.001
            pre_nms_topk = 2000
            num_classes = rpn_scores.shape[-1]
            points = points.cpu()

            results = {}
            for i in range(len(metas)):  # processing each video
                scores = rpn_scores[i].detach().cpu()  # [N]
                reg = rpn_reg[i].detach().cpu()  # [num_classes, N, 2]

                # scores_sum = scores.sum(0)
                # q1 = np.percentile(scores_sum, 25)
                # q3 = np.percentile(scores_sum, 75)
                # iqr = q3 - q1
                # upper_bound = q3 + 1.5 * iqr
                # filter_cls = [i for i, x in enumerate(scores_sum) if x > upper_bound]
                # one_hot_cls = [0]*len(scores_sum)
                # for idx in filter_cls:
                #     one_hot_cls[idx] = 1
                # scores = scores * torch.tensor(one_hot_cls)

                if num_classes == 1:
                    segments = self.rpn_head.get_proposals(points, reg.squeeze(0)).detach().cpu()  # [N, 2]
                    scores = scores.squeeze(-1)
                    labels = torch.zeros(scores.shape[0]).contiguous()
                else:
                    pred_prob = scores.flatten()  # [N*class]

                    # Apply filtering to make NMS faster following detectron2
                    # 1. Keep seg with confidence score > a threshold
                    keep_idxs1 = pred_prob > pre_nms_thresh
                    pred_prob = pred_prob[keep_idxs1]
                    topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                    # 2. Keep top k top scoring boxes only
                    num_topk = min(pre_nms_topk, topk_idxs.size(0))
                    pred_prob, idxs = pred_prob.sort(descending=True)
                    pred_prob = pred_prob[:num_topk].clone()
                    topk_idxs = topk_idxs[idxs[:num_topk]].clone()

                    # 3. gather predicted proposals
                    pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                    cls_idxs = torch.fmod(topk_idxs, num_classes)

                    segments = self.rpn_head.get_proposals(points[pt_idxs], reg[cls_idxs, pt_idxs]).detach().cpu()  # [N, 2]
                    scores = pred_prob
                    labels = cls_idxs

                # if not sliding window, do nms
                if post_cfg.sliding_window == False and post_cfg.nms is not None:
                    segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

                video_id = metas[i]["video_name"]

                # convert segments to seconds
                segments = convert_to_seconds(segments, metas[i])

                # merge with external classifier
                if isinstance(ext_cls, list):  # own classification results
                    labels = [ext_cls[label.item()] for label in labels]
                else:
                    segments, labels, scores = ext_cls(video_id, segments, scores)

                results_per_video = []
                for segment, label, score in zip(segments, labels, scores):
                    # convert to python scalars
                    results_per_video.append(
                        dict(
                            segment=[round(seg.item(), 2) for seg in segment],
                            label=label,
                            score=round(score.item(), 4),
                        )
                    )

                if video_id in results.keys():
                    results[video_id].extend(results_per_video)
                else:
                    results[video_id] = results_per_video

            return results

        else:
            with torch.no_grad():
                rpn_proposals, rpn_scores = predictions
                # rpn_proposals,  # [B,K,2]
                # rpn_scores,  # [B,K,num_classes] after sigmoid

                pre_nms_thresh = getattr(post_cfg, "pre_nms_thresh", 0.001)
                pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 2000)
                num_classes = rpn_scores[0].shape[-1]

                results = {}
                for i in range(len(metas)):  # processing each video
                    segments = rpn_proposals[i].detach().cpu()  # [N,2]
                    scores = rpn_scores[i].detach().cpu()  # [N,class]

                    if num_classes == 1:
                        scores = scores.squeeze(-1)
                        labels = torch.zeros(scores.shape[0]).contiguous()
                    else:
                        pred_prob = scores.flatten()  # [N*class]

                        # Apply filtering to make NMS faster following detectron2
                        # 1. Keep seg with confidence score > a threshold
                        keep_idxs1 = pred_prob > pre_nms_thresh
                        pred_prob = pred_prob[keep_idxs1]
                        topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                        # 2. Keep top k top scoring boxes only
                        num_topk = min(pre_nms_topk, topk_idxs.size(0))
                        pred_prob, idxs = pred_prob.sort(descending=True)
                        pred_prob = pred_prob[:num_topk].clone()
                        topk_idxs = topk_idxs[idxs[:num_topk]].clone()

                        # 3. gather predicted proposals
                        pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                        cls_idxs = torch.fmod(topk_idxs, num_classes)

                        segments = segments[pt_idxs]
                        scores = pred_prob
                        labels = cls_idxs

                    # if not sliding window, do nms
                    if post_cfg.sliding_window == False and post_cfg.nms is not None:
                        segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

                    video_id = metas[i]["video_name"]

                    # convert segments to seconds
                    segments = convert_to_seconds(segments, metas[i])

                    # merge with external classifier
                    if isinstance(ext_cls, list):  # own classification results
                        labels = [ext_cls[label.item()] for label in labels]
                    else:
                        segments, labels, scores = ext_cls(video_id, segments, scores)

                    results_per_video = []
                    for segment, label, score in zip(segments, labels, scores):
                        # convert to python scalars
                        results_per_video.append(
                            dict(
                                segment=[round(seg.item(), 2) for seg in segment],
                                label=label,
                                score=round(score.item(), 4),
                            )
                        )

                    if video_id in results.keys():
                        results[video_id].extend(results_per_video)
                    else:
                        results[video_id] = results_per_video

                return results

    # def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
    #     points, rpn_reg, rpn_scores = predictions  # [N, 4], [B, num_classes, N, 2], [B, N, num_classes]

    #     pre_nms_thresh = 0.001
    #     pre_nms_topk = 2000
    #     num_classes = rpn_scores.shape[-1]
    #     points = points.cpu()

    #     results = {}
    #     for i in range(len(metas)):  # processing each video
    #         scores = rpn_scores[i].detach().cpu()  # [N]
    #         reg = rpn_reg[i].detach().cpu()  # [num_classes, N, 2]

    #         # scores_sum = scores.sum(0)
    #         # q1 = np.percentile(scores_sum, 25)
    #         # q3 = np.percentile(scores_sum, 75)
    #         # iqr = q3 - q1
    #         # upper_bound = q3 + 1.5 * iqr
    #         # filter_cls = [i for i, x in enumerate(scores_sum) if x > upper_bound]
    #         # one_hot_cls = [0]*len(scores_sum)
    #         # for idx in filter_cls:
    #         #     one_hot_cls[idx] = 1
    #         # scores = scores * torch.tensor(one_hot_cls)

    #         if num_classes == 1:
    #             segments = self.rpn_head.get_proposals(points, reg.squeeze(0)).detach().cpu()  # [N, 2]
    #             scores = scores.squeeze(-1)
    #             labels = torch.zeros(scores.shape[0]).contiguous()
    #         else:
    #             pred_prob = scores.flatten()  # [N*class]

    #             # Apply filtering to make NMS faster following detectron2
    #             # 1. Keep seg with confidence score > a threshold
    #             keep_idxs1 = pred_prob > pre_nms_thresh
    #             pred_prob = pred_prob[keep_idxs1]
    #             topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

    #             # 2. Keep top k top scoring boxes only
    #             num_topk = min(pre_nms_topk, topk_idxs.size(0))
    #             pred_prob, idxs = pred_prob.sort(descending=True)
    #             pred_prob = pred_prob[:num_topk].clone()
    #             topk_idxs = topk_idxs[idxs[:num_topk]].clone()

    #             # 3. gather predicted proposals
    #             pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
    #             cls_idxs = torch.fmod(topk_idxs, num_classes)

    #             segments = self.rpn_head.get_proposals(points[pt_idxs], reg[cls_idxs, pt_idxs]).detach().cpu()  # [N, 2]
    #             scores = pred_prob
    #             labels = cls_idxs

    #         # if not sliding window, do nms
    #         if post_cfg.sliding_window == False and post_cfg.nms is not None:
    #             segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

    #         video_id = metas[i]["video_name"]

    #         # convert segments to seconds
    #         segments = convert_to_seconds(segments, metas[i])

    #         # merge with external classifier
    #         if isinstance(ext_cls, list):  # own classification results
    #             labels = [ext_cls[label.item()] for label in labels]
    #         else:
    #             segments, labels, scores = ext_cls(video_id, segments, scores)

    #         results_per_video = []
    #         for segment, label, score in zip(segments, labels, scores):
    #             # convert to python scalars
    #             results_per_video.append(
    #                 dict(
    #                     segment=[round(seg.item(), 2) for seg in segment],
    #                     label=label,
    #                     score=round(score.item(), 4),
    #                 )
    #             )

    #         if video_id in results.keys():
    #             results[video_id].extend(results_per_video)
    #         else:
    #             results[video_id] = results_per_video

    #     return results


