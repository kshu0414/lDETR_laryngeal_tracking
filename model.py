import torch
import torch.nn as nn
from torchvision.models import resnet50

class lDETR(nn.Module):
    """
    l-DETR implementation adapted from a minimal DETR model available from 
        https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb
    """
    def __init__(self, num_classes=4, hidden_dim=256, nheads=4,
                 num_encoder_layers=5, num_decoder_layers=5):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50(pretrained=True)
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_pred = nn.Linear(hidden_dim, num_classes)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(1, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        
        #print(pos.shape,h.shape,self.query_pos.unsqueeze(1).repeat(1, BATCH_SIZE, 1).shape)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1).repeat(1, inputs.shape[0], 1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        h = h.squeeze(1)
        return self.linear_pred(h)

