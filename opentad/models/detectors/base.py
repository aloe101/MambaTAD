import torch
from ..utils.post_processing import load_predictions, save_predictions


class BaseDetector(torch.nn.Module):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()

    def forward(
        self,
        inputs,     #end2end: 2 1 3 768 160 160; normal:2 3200 2304
        masks=None,      #2 768; 2 2304
        metas=None,
        gt_segments=None,       #2 x 2(s,e)
        gt_labels=None,         #2 x
        return_loss=True,
        infer_cfg=None,
        post_cfg=None,
        **kwargs
    ):
        if return_loss:
            if ('video_validation_0000185' or 'video_validation_0000208') == (metas[0]['video_name'] or metas[1]['video_name']):
                return self.forward_train(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs)
            return self.forward_train(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs)
            
        else:
            return self.forward_detection(inputs, masks, metas, infer_cfg, post_cfg, **kwargs)

    def forward_detection(self, inputs, masks, metas, infer_cfg, post_cfg, **kwargs):
        # step1: inference the model
        if infer_cfg.load_from_raw_predictions:  # easier and faster to tune the hyper parameter in postprocessing
            predictions = load_predictions(metas, infer_cfg)
        else:
            predictions = self.forward_test(inputs, masks, metas, infer_cfg)

            if infer_cfg.save_raw_prediction:  # save the predictions to disk
                save_predictions(predictions, metas, infer_cfg.folder)

        # step2: detection post processing
        results = self.post_processing(predictions, metas, post_cfg, **kwargs)
        return results
