"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                            accuracy, get_world_size, interpolate,
                            is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .depthaware_transformer import build_depthaware_transformer
from .depth_predictor import DepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from lib.losses.focal_loss import sigmoid_focal_loss
from .dn_components import prepare_for_dn, dn_post_process, compute_dn_loss
from .geometry_e1 import (
    GeometryE1,
    horizon_focal_loss,
    make_horizon_heatmap_target,
)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MonoDETR(nn.Module):
    """ This is the MonoDETR module that performs monocualr 3D object detection """
    def __init__(self, backbone, depthaware_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, init_box=False, use_dab=False,
                 group_num=11, two_stage_dino=False, geometry_cfg=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage MonoDETR
        """
        super().__init__()
 
        self.num_queries = num_queries
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.group_num = group_num
        self.two_stage_dino = two_stage_dino
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation
        self.use_dab = use_dab

        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if not two_stage:
            if two_stage_dino:
                self.query_embed = None
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries * group_num, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries * group_num, 6)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.num_classes = num_classes
        self.geometry_cfg = geometry_cfg or {}
        self.geometry_enabled = bool(self.geometry_cfg.get('enabled', False))
        self.fusion_start_epoch = int(
            self.geometry_cfg.get('training', {}).get('fusion_start_epoch', 8))
        self.current_epoch = 0
        if self.geometry_enabled:
            self.geometry_e1 = GeometryE1(
                self.geometry_cfg,
                hidden_dim,
                num_queries,
                self.depth_predictor.depth_bin_values,
            )

        if self.two_stage_dino:        
            _class_embed = nn.Linear(hidden_dim, num_classes)
            _bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
            # init the two embed layers
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(num_classes) * bias_value
            nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)   
            self.depthaware_transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.depthaware_transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (depthaware_transformer.decoder.num_layers + 1) if two_stage else depthaware_transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d  
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.depthaware_transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)


    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _query_embeddings(self):
        if self.two_stage or self.two_stage_dino:
            return None
        if self.use_dab:
            if self.training:
                tgt_embed = self.tgt_embed.weight
                refanchor = self.refpoint_embed.weight
            else:
                tgt_embed = self.tgt_embed.weight[:self.num_queries]
                refanchor = self.refpoint_embed.weight[:self.num_queries]
            return torch.cat((tgt_embed, refanchor), dim=1)
        if self.training:
            return self.query_embed.weight
        return self.query_embed.weight[:self.num_queries]

    def _input_frame_calibration(self, images, calibs, img_sizes, geometry_calibs):
        if geometry_calibs is not None:
            return geometry_calibs
        # Resize-only fallback. E1 configs disable random crop/flip; callers
        # with geometric augmentation must supply the exact transformed P2.
        p2 = calibs.clone()
        scale_x = float(images.shape[-1]) / img_sizes[:, 0].to(p2.dtype)
        scale_y = float(images.shape[-2]) / img_sizes[:, 1].to(p2.dtype)
        p2[:, 0] = p2[:, 0] * scale_x[:, None]
        p2[:, 1] = p2[:, 1] * scale_y[:, None]
        return p2

    def _decode_detection_heads(self, transformer_outputs, weighted_depth,
                                calibs, img_sizes, depth_map_logits):
        (hs, init_reference, inter_references, inter_references_dim,
         enc_outputs_class, enc_outputs_coord_unact) = transformer_outputs
        outputs_coords, outputs_classes = [], []
        outputs_3d_dims, outputs_depths, outputs_angles = [], [], []

        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_classes.append(self.class_embed[lvl](hs[lvl]))

            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(
                box2d_height_norm * img_sizes[:, 1:2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)
            depth_reg = self.depth_embed[lvl](hs[lvl])
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1), outputs_center3d,
                mode='bilinear', align_corners=True).squeeze(1)
            depth_ave = torch.cat([
                ((1. / (depth_reg[:, :, 0:1].sigmoid() + 1e-6) - 1.)
                 + depth_geo.unsqueeze(-1) + depth_map) / 3,
                depth_reg[:, :, 1:2]], -1)
            outputs_depths.append(depth_ave)
            outputs_angles.append(self.angle_embed[lvl](hs[lvl]))

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'pred_3d_dim': outputs_3d_dim[-1],
            'pred_depth': outputs_depth[-1],
            'pred_angle': outputs_angle[-1],
            'pred_depth_map_logits': depth_map_logits,
            '_query_features': hs[-1],
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)
        if self.two_stage:
            out['enc_outputs'] = {
                'pred_logits': enc_outputs_class,
                'pred_boxes': enc_outputs_coord_unact.sigmoid(),
            }
        return out

    def forward(self, images, calibs, targets, img_sizes, dn_args=None,
                geometry_calibs=None, force_phase=None):
        """Run baseline MonoDETR or the E1 one/two-pass geometry pipeline."""
        features, pos = self.backbone(images)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self._query_embeddings()
        depth_source = self.depth_predictor.extract_depth_feature(srcs)
        original_depth_logits = self.depth_predictor.depth_classifier(depth_source)
        original_depth_base = self.depth_predictor.encode_depth_feature(
            depth_source, masks[1], pos[1])
        original_weighted_depth, original_depth_position = self.depth_predictor.depth_from_logits(
            original_depth_logits)
        original_depth_memory = original_depth_base + original_depth_position

        phase2 = False
        if self.geometry_enabled:
            if force_phase is not None:
                phase2 = int(force_phase) == 2
            else:
                phase2 = (not self.training) or self.current_epoch >= self.fusion_start_epoch

        if phase2:
            pass1_transformer, visual_cache = self.depthaware_transformer(
                srcs, masks, pos, query_embeds, original_depth_memory,
                original_depth_position, return_visual_cache=True)
        else:
            pass1_transformer = self.depthaware_transformer(
                srcs, masks, pos, query_embeds, original_depth_memory,
                original_depth_position)
            visual_cache = None

        pass1 = self._decode_detection_heads(
            pass1_transformer, original_weighted_depth, calibs, img_sizes,
            original_depth_logits)
        if not self.geometry_enabled:
            pass1.pop('_query_features')
            return pass1

        image_hw = (images.shape[-2], images.shape[-1])
        horizon_logits, horizon_line, horizon_rows, horizon_confidence = (
            self.geometry_e1.predict_horizon(srcs, image_hw))
        p2_input = self._input_frame_calibration(
            images, calibs, img_sizes, geometry_calibs)
        geometry = self.geometry_e1.predict_geometry(
            pass1['_query_features'], pass1['pred_boxes'], pass1['pred_logits'],
            horizon_line, horizon_confidence, p2_input, image_hw,
            original_depth_logits.shape[-2:])
        pass1.update({
            'pred_horizon_logits': horizon_logits,
            'pred_horizon_line': horizon_line,
            'pred_horizon_rows': horizon_rows,
            'pred_horizon_confidence': horizon_confidence,
            'pred_touchpoints': geometry['touchpoints_abs'],
            'pred_touchpoints_rel': geometry['touchpoints_rel'],
            'geometry': geometry,
            'e1_phase': 1,
        })
        pass1.pop('_query_features')
        if not phase2:
            return pass1

        if self.geometry_e1.mode == 'feature_level':
            fused_source = self.geometry_e1.fuse_feature(depth_source, geometry)
            fused_depth_logits = self.depth_predictor.depth_classifier(fused_source)
            fused_depth_base = self.depth_predictor.encode_depth_feature(
                fused_source, masks[1], pos[1])
            fused_weighted_depth, fused_depth_position = self.depth_predictor.depth_from_logits(
                fused_depth_logits)
            fused_depth_memory = fused_depth_base + fused_depth_position
        else:
            fused_depth_logits = self.geometry_e1.fuse_distribution(
                original_depth_logits, geometry)
            fused_weighted_depth, fused_depth_position = self.depth_predictor.depth_from_logits(
                fused_depth_logits)
            fused_depth_memory = original_depth_base + fused_depth_position

        pass2_transformer = self.depthaware_transformer.decode_from_memory(
            visual_cache, query_embeds, fused_depth_memory, fused_depth_position)
        out = self._decode_detection_heads(
            pass2_transformer, fused_weighted_depth, calibs, img_sizes,
            original_depth_logits)
        out.update({
            'pred_fused_depth_map_logits': fused_depth_logits,
            'original_depth_map_logits': original_depth_logits,
            'pred_horizon_logits': horizon_logits,
            'pred_horizon_line': horizon_line,
            'pred_horizon_rows': horizon_rows,
            'pred_horizon_confidence': horizon_confidence,
            'pred_touchpoints': geometry['touchpoints_abs'],
            'pred_touchpoints_rel': geometry['touchpoints_rel'],
            'geometry': geometry,
            'pass1_outputs': pass1,
            'fusion_alpha': self.geometry_e1.alpha,
            'e1_phase': 2,
        })
        out.pop('_query_features')
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses,
                 group_num=11, geometry_cfg=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map
        self.group_num = group_num
        self.geometry_cfg = geometry_cfg or {}
        self.geometry_enabled = bool(self.geometry_cfg.get('enabled', False))
        horizon_cfg = self.geometry_cfg.get('horizon', {})
        touch_cfg = self.geometry_cfg.get('touchpoint', {})
        self.horizon_sigma = float(horizon_cfg.get('gaussian_sigma_heatmap_px', 1.5))
        self.horizon_stride = int(horizon_cfg.get('heatmap_stride', 4))
        self.touchpoint_class_index = int(touch_cfg.get('class_index', 1))
        self.touchpoint_loss_coordinate_mode = str(
            touch_cfg.get('loss_coordinate_mode', 'image_normalized')).lower()
        if self.touchpoint_loss_coordinate_mode not in (
                'image_normalized', 'box_relative'):
            raise ValueError(
                'touchpoint.loss_coordinate_mode must be image_normalized or box_relative')
        self.touchpoint_smooth_l1_beta = float(
            touch_cfg.get('smooth_l1_beta', 1.0))
        if self.touchpoint_smooth_l1_beta <= 0:
            raise ValueError('touchpoint.smooth_l1_beta must be positive')
        self.pass1_aux_detection_coef = float(
            self.geometry_cfg.get('losses', {}).get(
                'pass1_aux_detection_loss_coef_phase2', 0.0))

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @staticmethod
    def _has_matches(indices):
        return any(source.numel() > 0 for source, _ in indices)

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes):
        if not self._has_matches(indices):
            return {'loss_center': outputs['pred_boxes'].sum() * 0.0}
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        if not self._has_matches(indices):
            zero = outputs['pred_boxes'].sum() * 0.0
            return {'loss_bbox': zero, 'loss_giou': zero}
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes):  
        if not self._has_matches(indices):
            return {'loss_depth': outputs['pred_depth'].sum() * 0.0}
        idx = self._get_src_permutation_idx(indices)
   
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance  
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes 
        return losses  
    
    def loss_dims(self, outputs, targets, indices, num_boxes):  
        if not self._has_matches(indices):
            return {'loss_dim': outputs['pred_3d_dim'].sum() * 0.0}
        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):  
        if not self._has_matches(indices):
            return {'loss_angle': outputs['pred_angle'].sum() * 0.0}
        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(
            heading_target_cls.shape[0], 12,
            device=heading_input.device, dtype=heading_input.dtype
        ).scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes):
        depth_map_logits = outputs['pred_depth_map_logits']
        return self._depth_map_loss(depth_map_logits, targets, 'loss_depth_map')

    def _depth_map_loss(self, depth_map_logits, targets, loss_name):
        num_gt_per_img = [len(t['boxes']) for t in targets]
        height, width = depth_map_logits.shape[-2:]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor(
            [width, height, width, height],
            device=depth_map_logits.device,
            dtype=depth_map_logits.dtype)
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)
        return {loss_name: self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        }

    def loss_fused_depth_map(self, outputs, targets, indices, num_boxes):
        logits = outputs.get('pred_fused_depth_map_logits')
        if logits is None:
            return {'loss_fused_depth_map': outputs['pred_depth_map_logits'].sum() * 0.0}
        return self._depth_map_loss(logits, targets, 'loss_fused_depth_map')

    def loss_horizon(self, outputs, targets, indices, num_boxes):
        logits = outputs['pred_horizon_logits']
        predicted_line = outputs['pred_horizon_line']
        target_line = torch.stack([target['horizon_line'] for target in targets]).to(logits)
        valid = torch.stack([target['horizon_valid'] for target in targets]).bool().to(logits.device)
        input_hw = (
            int(logits.shape[-2] * self.horizon_stride),
            int(logits.shape[-1] * self.horizon_stride),
        )
        target_heatmap, target_mask = make_horizon_heatmap_target(
            target_line, valid, logits.shape[-2:], input_hw, self.horizon_sigma)
        heatmap_loss = horizon_focal_loss(logits, target_heatmap, target_mask)
        if bool(valid.any()):
            input_h, input_w = input_hw
            center_u = (input_w - 1.0) / 2.0
            pred_stable = torch.stack((
                predicted_line[:, 0] * input_w / input_h,
                (predicted_line[:, 0] * center_u + predicted_line[:, 1]) / input_h,
            ), dim=-1)
            target_stable = torch.stack((
                target_line[:, 0] * input_w / input_h,
                (target_line[:, 0] * center_u + target_line[:, 1]) / input_h,
            ), dim=-1)
            line_loss = F.smooth_l1_loss(
                pred_stable[valid], target_stable[valid], reduction='mean')
        else:
            line_loss = predicted_line.sum() * 0.0
        if bool(valid.any()):
            slope_error = (predicted_line[valid, 0] - target_line[valid, 0]).abs().mean()
            center_error = (
                predicted_line[valid, 0] * center_u + predicted_line[valid, 1]
                - target_line[valid, 0] * center_u - target_line[valid, 1]
            ).abs().mean()
        else:
            slope_error = predicted_line.sum() * 0.0
            center_error = predicted_line.sum() * 0.0
        return {
            'loss_horizon_heatmap': heatmap_loss,
            'loss_horizon_line': line_loss,
            'horizon_slope_mae': slope_error.detach(),
            'horizon_center_mae_px': center_error.detach(),
        }

    def loss_touchpoints(self, outputs, targets, indices, num_boxes):
        source_index = self._get_src_permutation_idx(indices)
        predicted_absolute = outputs['pred_touchpoints'][source_index]
        target_points = torch.cat([
            target['touch_points'][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ], dim=0)
        valid = torch.cat([
            target['touchpoint_valid'][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ], dim=0).bool()
        labels = torch.cat([
            target['labels'][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ], dim=0).long()
        valid &= labels[:, None] == self.touchpoint_class_index
        valid_count = valid.sum()
        if int(valid_count.item()) == 0:
            zero = predicted_absolute.sum() * 0.0
            return {
                'loss_touchpoint': zero,
                'touchpoint_mae_px': zero.detach(),
                'touchpoint_valid_count': valid_count.detach().float(),
            }

        if self.touchpoint_loss_coordinate_mode == 'box_relative':
            predicted_for_loss = outputs['pred_touchpoints_rel'][source_index]
            target_boxes = torch.cat([
                target['boxes_3d'][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ], dim=0)
            target_xyxy = box_ops.box_cxcylrtb_to_xyxy(target_boxes)
            target_top_left = target_xyxy[:, None, :2]
            target_size = (
                target_xyxy[:, None, 2:] - target_xyxy[:, None, :2]
            ).clamp_min(1e-6)
            target_for_loss = (target_points - target_top_left) / target_size
        else:
            predicted_for_loss = predicted_absolute
            target_for_loss = target_points

        difference = predicted_absolute - target_points
        loss = F.smooth_l1_loss(
            predicted_for_loss[valid], target_for_loss[valid],
            reduction='sum', beta=self.touchpoint_smooth_l1_beta) / valid_count
        input_h = outputs['pred_horizon_logits'].shape[-2] * self.horizon_stride
        input_w = outputs['pred_horizon_logits'].shape[-1] * self.horizon_stride
        pixel_error = (difference * difference.new_tensor([input_w, input_h])).abs()
        pixel_mae = pixel_error[valid].mean().detach()
        return {
            'loss_touchpoint': loss,
            'touchpoint_mae_px': pixel_mae,
            'touchpoint_valid_count': valid_count.detach().float(),
        }

    def loss_geometry_metrics(self, outputs, targets, indices, num_boxes):
        geometry = outputs.get('geometry')
        if geometry is None:
            zero = outputs['pred_depth'].sum() * 0.0
            return {'geometry_depth_mae_m': zero.detach(),
                    'geometry_valid_matched_ratio': zero.detach()}
        source_batch, source_query = self._get_src_permutation_idx(indices)
        target_depths = torch.cat([
            target['depth'][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ], dim=0).reshape(-1)
        canonical = geometry['object_depth'].shape[1]
        in_canonical = source_query < canonical
        source_batch = source_batch[in_canonical]
        source_query = source_query[in_canonical]
        target_depths = target_depths[in_canonical]
        if source_query.numel() == 0:
            zero = geometry['object_depth'].sum() * 0.0
            return {'geometry_depth_mae_m': zero.detach(),
                    'geometry_valid_matched_ratio': zero.detach()}
        valid = geometry['object_valid'][source_batch, source_query]
        valid_ratio = valid.float().mean()
        if bool(valid.any()):
            predicted = geometry['object_depth'][source_batch, source_query]
            mae = (predicted[valid] - target_depths[valid]).abs().mean()
        else:
            mae = geometry['object_depth'].sum() * 0.0
        return {'geometry_depth_mae_m': mae.detach(),
                'geometry_valid_matched_ratio': valid_ratio.detach()}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
            'fused_depth_map': self.loss_fused_depth_map,
            'horizon': self.loss_horizon,
            'touchpoints': self.loss_touchpoints,
            'geometry_metrics': self.loss_geometry_metrics,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_num=group_num)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses

        losses = {}
        pass1_indices = None
        if 'pass1_outputs' in outputs:
            pass1_indices = self.matcher(
                outputs['pass1_outputs'], targets, group_num=group_num)
        for loss in self.losses:
            #ipdb.set_trace()
            selected_indices = (
                pass1_indices if loss in ('touchpoints', 'geometry_metrics')
                and pass1_indices is not None else indices)
            losses.update(self.get_loss(
                loss, outputs, targets, selected_indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_num=group_num)
                for loss in self.losses:
                    if loss in ('depth_map', 'fused_depth_map', 'horizon', 'touchpoints',
                                'geometry_metrics'):
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Phase 2 may retain a low-weight stabilizing loss on the final layer
        # of pass 1. The original depth-map loss is intentionally excluded: it
        # is already computed exactly once from original logits above.
        if self.pass1_aux_detection_coef > 0 and 'pass1_outputs' in outputs:
            pass1 = outputs['pass1_outputs']
            if pass1_indices is None:
                pass1_indices = self.matcher(pass1, targets, group_num=group_num)
            for loss in ('labels', 'boxes', 'depths', 'dims', 'angles', 'center'):
                kwargs = {'log': False} if loss == 'labels' else {}
                values = self.get_loss(
                    loss, pass1, targets, pass1_indices, num_boxes, **kwargs)
                for key, value in values.items():
                    if key == 'class_error':
                        continue
                    losses['loss_pass1_' + key.replace('loss_', '', 1)] = value
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    depthaware_transformer = build_depthaware_transformer(cfg)

    # depth prediction module
    depth_predictor = DepthPredictor(cfg)

    model = MonoDETR(
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        two_stage=cfg['two_stage'],
        init_box=cfg['init_box'],
        use_dab = cfg['use_dab'],
        group_num=cfg.get('group_num', 11),
        two_stage_dino=cfg['two_stage_dino'],
        geometry_cfg=cfg.get('geometry_e1'))

    # matcher
    matcher = build_matcher(cfg)

    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']
    geometry_cfg = cfg.get('geometry_e1') or {}
    if geometry_cfg.get('enabled', False):
        geometry_losses = geometry_cfg.get('losses', {})
        weight_dict['loss_touchpoint'] = geometry_losses.get('touchpoint_loss_coef', 1.0)
        weight_dict['loss_horizon_heatmap'] = geometry_losses.get('horizon_heatmap_loss_coef', 1.0)
        weight_dict['loss_horizon_line'] = geometry_losses.get('horizon_line_loss_coef', 0.5)
        weight_dict['loss_fused_depth_map'] = geometry_losses.get('fused_depth_map_loss_coef', 1.0)
        pass1_coef = geometry_losses.get('pass1_aux_detection_loss_coef_phase2', 0.0)
        if pass1_coef > 0:
            for key in ('loss_ce', 'loss_bbox', 'loss_giou', 'loss_dim',
                        'loss_angle', 'loss_depth', 'loss_center'):
                weight_dict['loss_pass1_' + key.replace('loss_', '', 1)] = (
                    weight_dict[key] * pass1_coef)
    
    # dn loss
    if cfg['use_dn']:
        weight_dict['tgt_loss_ce']= cfg['cls_loss_coef']
        weight_dict['tgt_loss_bbox'] = cfg['bbox_loss_coef']
        weight_dict['tgt_loss_giou'] = cfg['giou_loss_coef']
        weight_dict['tgt_loss_angle'] = cfg['angle_loss_coef']
        weight_dict['tgt_loss_center'] = cfg['3dcenter_loss_coef']

    # TODO this is a hack
    if cfg['aux_loss']:
        aux_weight_dict = {}
        auxiliary_base_weights = {
            key: value for key, value in weight_dict.items()
            if key not in ('loss_depth_map', 'loss_fused_depth_map',
                           'loss_touchpoint', 'loss_horizon_heatmap',
                           'loss_horizon_line') and not key.startswith('loss_pass1_')
        }
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in auxiliary_base_weights.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in auxiliary_base_weights.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map']
    if geometry_cfg.get('enabled', False):
        losses.extend(['fused_depth_map', 'horizon', 'touchpoints', 'geometry_metrics'])
    
    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses,
        group_num=cfg.get('group_num', 11),
        geometry_cfg=geometry_cfg)

    device = torch.device(cfg['device'])
    criterion.to(device)
    
    return model, criterion
