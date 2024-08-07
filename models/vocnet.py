import torch
import torch.nn as nn
import torchvision.models as models


class VOCnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 使用预训练的 VGG16 作为特征提取器
        self.features = models.vgg16(pretrained=True).features
        # 以下是 Faster R-CNN 的头部结构示例，实际上可以根据需要调整
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.cls_score = nn.Linear(4096, num_classes)  # 分类得分
        self.bbox_pred = nn.Linear(4096, num_classes * 4)  # 边界框预测

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.classifier(x)
        cls_score = self.cls_score(x)  # 分类得分
        bbox_pred = self.bbox_pred(x)  # 边界框预测
        return cls_score, bbox_pred


def vocnet():
    return VOCnet(num_classes=20)  # VOC 数据集有 20 个类别

