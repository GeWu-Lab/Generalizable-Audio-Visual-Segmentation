import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 计算类别概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # 计算 Focal Loss

        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)  # 求均值
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)  # 求和

        return focal_loss
    
focal = FocalLoss()

def F10_IoU_BCELoss(pred_mask, ten_gt_masks, gt_temporal_mask_flag):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    ten_gt_masks: ground truth mask of the total five frames, shape: [bs*10, 224, 224]
    """
    # print('loss:', pred_mask.shape)
    assert len(pred_mask.shape) == 4
    if ten_gt_masks.shape[1] == 1:
        ten_gt_masks = ten_gt_masks.squeeze(1) # [bs*10, 224, 224]
    # loss = nn.CrossEntropyLoss()(pred_mask, ten_gt_masks)
    #! notice:
    loss = nn.CrossEntropyLoss(reduction='none')(pred_mask, ten_gt_masks) # [bs*10, 224, 224]
    # focal_loss_ = focal(pred_mask, ten_gt_masks)
    loss = loss.mean(-1).mean(-1) # + focal_loss_  # [bs*10]
    
    loss = loss  # * gt_temporal_mask_flag # [bs*10]
    # print('loss:', loss.shape)
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)

    return loss

def A_MaskedV_SimmLoss(pred_masks, a_fea_list, v_map_list, \
                        count_stages=[], \
                        mask_pooling_type='avg', norm_fea=True, threshold=False,\
                        euclidean_flag=False, kl_flag=False):
    """
    [audio] - [masked visual feature map] matching loss, Loss_AVM_AV reported in the paper

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    a_fea_list: audio feature list, lenth = nl_stages, each of shape: [bs, T, C], C is equal to [256]
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*5, C, H, W], C is equal to [256]
    count_stages: loss is computed in these stages
    """
    assert len(pred_masks.shape) == 4
    total_loss = 0
    for stage in count_stages:
        a_fea, v_map = a_fea_list[stage], v_map_list[stage] # v_map: [BT, C, H, W]
        a_fea = a_fea.view(-1, a_fea.shape[-1]) # [B*5, C]

        C, H, W = v_map.shape[1], v_map.shape[-2], v_map.shape[-1]
        assert C == a_fea.shape[-1], 'Error: dimensions of audio and visual features are not equal'

        if mask_pooling_type == "avg":
            downsample_pred_masks = nn.AdaptiveAvgPool2d((H, W))(pred_masks) # [bs*5, 1, H, W]
        elif mask_pooling_type == 'max':
            downsample_pred_masks = nn.AdaptiveMaxPool2d((H, W))(pred_masks) # [bs*5, 1, H, W]
        downsample_pred_masks = torch.sigmoid(downsample_pred_masks) # [B*5, 1, H, W]

        if threshold:
            downsample_pred_masks = (downsample_pred_masks > 0.5).float() # [bs*5, 1, H, W]
            obj_pixel_num = downsample_pred_masks.sum(-1).sum(-1) # [bs*5, 1]
            masked_v_map = torch.mul(v_map, downsample_pred_masks)  # [bs*5, C, H, W]
            masked_v_fea = masked_v_map.sum(-1).sum(-1) / (obj_pixel_num + 1e-6)# [bs*5, C]
        else:
            masked_v_map = torch.mul(v_map, downsample_pred_masks)
            masked_v_fea = masked_v_map.mean(-1).mean(-1) # [bs*5, C]

        if norm_fea:
            a_fea = F.normalize(a_fea, dim=-1)
            masked_v_fea = F.normalize(masked_v_fea, dim=-1)

        if euclidean_flag:
            euclidean_distance = F.pairwise_distance(a_fea, masked_v_fea, p=2)
            loss = euclidean_distance.mean()
        elif kl_flag:
            loss = F.kl_div(masked_v_fea.softmax(dim=-1).log(), a_fea.softmax(dim=-1), reduction='sum')
        total_loss += loss

    total_loss /= len(count_stages)

    return total_loss


def closer_loss(pred_masks, a_fea_list, v_map_list, \
                        count_stages=[], \
                        mask_pooling_type='avg', norm_fea=True, \
                        euclidean_flag=False, kl_flag=False):
    """
    [audio] - [masked visual feature map] matching loss, Loss_AVM_VV reported in the paper

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    a_fea_list: audio feature list, lenth = nl_stages, each of shape: [bs, T, C], C is equal to [256]
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*5, C, H, W], C is equal to [256]
    count_stages: loss is computed in these stages
    """
    assert len(pred_masks.shape) == 4
    total_loss = 0
    for stage in count_stages:
        a_fea, v_map = a_fea_list[stage], v_map_list[stage] # v_map: [BT, C, H, W]
        a_fea = a_fea.view(-1, a_fea.shape[-1]) # [B*5, C]

        C, H, W = v_map.shape[1], v_map.shape[-2], v_map.shape[-1]
        assert C == a_fea.shape[-1], 'Error: dimensions of audio and visual features are not equal'

        if mask_pooling_type == "avg":
            downsample_pred_masks = nn.AdaptiveAvgPool2d((H, W))(pred_masks) # [bs*5, 1, H, W]
        elif mask_pooling_type == 'max':
            downsample_pred_masks = nn.AdaptiveMaxPool2d((H, W))(pred_masks) # [bs*5, 1, H, W]
        downsample_pred_masks = torch.sigmoid(downsample_pred_masks) # [B*5, 1, H, W]

        ###############################################################################
        # pick the closest pair
        if norm_fea:
            a_fea = F.normalize(a_fea, dim=-1)

        a_fea_simi = torch.cdist(a_fea,a_fea,p=2) # [BT, BT]
        a_fea_simi = a_fea_simi + 10*torch.eye(a_fea_simi.shape[0]).cuda() #
        idxs = a_fea_simi.argmin(dim=0) # [BT]

        masked_v_map = torch.mul(v_map, downsample_pred_masks)
        masked_v_fea = masked_v_map.mean(-1).mean(-1) # [bs*5, C]
        if norm_fea:
            masked_v_fea = F.normalize(masked_v_fea, dim=-1)

        target_fea = masked_v_fea[idxs]
        ###############################################################################
        if euclidean_flag:
            euclidean_distance = F.pairwise_distance(target_fea, masked_v_fea, p=2)
            loss = euclidean_distance.mean()
        elif kl_flag:
            loss = F.kl_div(masked_v_fea.softmax(dim=-1).log(), target_fea.softmax(dim=-1), reduction='sum')
        total_loss += loss

    total_loss /= len(count_stages)

    return total_loss

def FocalLoss(pred_masks, gt_mask):
    n, c, h, w = pred_masks.size()
    pred_masks = pred_masks.permute(0, 2, 3, 1).contiguous().view(-1, c)  # 转换为(N * H * W, C)
    gt_mask = gt_mask.view(-1)  # 转换为(N * H * W,)
    
    logp = F.log_softmax(pred_masks, dim=1)
    target_onehot = F.one_hot(gt_mask, num_classes=c).float()  # 将target转换为one-hot编码
    pt = (target_onehot * logp).sum(dim=1)  # 提取对应类别的logit
    pt = pt.view(-1)
    
    alpha = 0.5
    gamma = 2
    at = alpha.to(pred_masks.device)[gt_mask]
    focal_loss = -at * (1 - pt) ** gamma * logp.sum(dim=1)

    focal_loss = focal_loss.mean()

    return focal_loss
    

def IouSemanticAwareLoss(pred_masks, gt_mask, gt_temporal_mask_flag):
    """
    loss for multiple sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    gt_mask: ground truth mask of the first frame (one-shot) or five frames, shape: [bs, 1, 1, 224, 224]
    a_fea_list: feature list of audio features
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*5, C, H, W]
    count_stages: additional constraint loss on which stages' visual-audio features
    """
    total_loss = 0
    iou_loss = F10_IoU_BCELoss(pred_masks, gt_mask, gt_temporal_mask_flag)
    total_loss += iou_loss

    return total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        """
        Focal Loss的初始化函数
        :param alpha: 平衡因子，用于调整正负样本的权重
        :param gamma: 聚焦因子，用于调整易分样本的权重
        :param reduction: 损失函数的降维模式，默认为'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Focal Loss的前向传播函数
        :param inputs: 模型的预测输出，大小为(N, C, H, W)，N为批量大小，C为类别数，H和W为图像尺寸
        :param targets: 真实标签，大小为(N, H, W)，取值为0到C-1之间的整数
        :return: Focal Loss值
        """
        # 将预测输出和真实标签转换为一维向量
        inputs = inputs.view(-1, inputs.size(1))
        targets = targets.view(-1)

        # 计算交叉熵损失
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')  # 每个像素点的交叉熵损失

        # 计算类别权重
        class_weights = torch.ones_like(inputs)
        class_weights[torch.arange(inputs.size(0)), targets] = 1 - self.alpha

        # 计算Focal Loss
        loss = (class_weights * (1 - torch.exp(-CE_loss))) ** self.gamma * CE_loss

        # 根据降维模式计算最终损失值
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss







if __name__ == "__main__":

    pdb.set_trace()
