import torch

from ..builder import BBOX_CODERS
from ..transforms import bbox_cxcywh_to_xyxy
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class PointDeltaXY_WHCoder(BaseBBoxCoder):
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (delta_x, delta_y, width,
    height) and decode it back to the original.
    """
    def __init__(self):
        super(PointDeltaXY_WHCoder, self).__init__()

    def encode(self, points, gt_bboxes):
        """Encode bounding box to delta_x, delta_y, width, height.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor): Shape (N, 4), The format is "xyxy"

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 4).
        """
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4
        cx = (gt_bboxes[..., 2] + gt_bboxes[..., 0]) * 0.5
        cy = (gt_bboxes[..., 3] + gt_bboxes[..., 1]) * 0.5
        w  = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h  = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        delta_x = cx - points[..., 0]
        delta_y = cy - points[..., 1]
        encoded_bbox = torch.stack([delta_x, delta_y, w, h], dim=-1)
        return encoded_bbox

    def decode(self, points, bboxes_pred):
        """Decode (center distance and width, height) prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to bbox
                center and width height (delta_x, delta_y, width, height). Shape (B, N, 4)
                or (N, 4)
        Returns:
            Tensor: Boxes with shape (N, 4) or (B, N, 4)
        """
        assert points.size(0) == bboxes_pred.size(0)
        assert points.size(-1) == 2
        assert bboxes_pred.size(-1) == 4
        cx = points[..., 0] + bboxes_pred[..., 0]
        cy = points[..., 1] + bboxes_pred[..., 1]
        w = bboxes_pred[..., 2]
        h = bboxes_pred[..., 3]
        bbox_cxcywh = torch.stack([cx, cy, w, h], dim=-1)
        return bbox_cxcywh_to_xyxy(bbox_cxcywh)
