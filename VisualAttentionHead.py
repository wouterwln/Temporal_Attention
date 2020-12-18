import torch
from torch import nn
import torch.nn.functional as F


class VisualAttentionHead(nn.Module):

    def __init__(self, input_channels,  affinity="softmax"):
        super().__init__()

        self.query_layer = nn.Conv2d(input_channels, input_channels//8, 1)
        self.key_layer = nn.Conv2d(input_channels, input_channels//8, 1)
        self.value_layer = nn.Conv2d(input_channels, input_channels//2, 1)

        self.combiner = nn.Conv2d(input_channels//2, input_channels//2, 1)
        if affinity == "softmax":
            self.affinity = nn.Softmax(dim=3)
        else:
            self.affinity = lambda x: torch.mul(x, x.shape[1] * x.shape[2])

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = F.relu(self.value_layer(x))

        x = self.elementwise_dot_product(q, k)
        x = self.affinity(x.view(x.shape[0], x.shape[1], x.shape[2], -1)).view(x.shape[0], x.shape[1], x.shape[2], x.shape[1], x.shape[2])
        x = self.combiner(torch.mul(x.unsqueeze(-1), v.permute(0, 2, 3, 1).unsqueeze(1).unsqueeze(1)).sum((3, 4)).permute(0, 3, 1, 2))

        return torch.cat((x, v), dim=1)


    @staticmethod
    def elementwise_dot_product(v_1, v_2):
        v_1 = v_1.permute(0, 2, 3, 1)
        r = torch.matmul(v_1.view(v_1.shape[0],-1, v_1.shape[3]), v_2.view(v_2.shape[0], v_2.shape[1], -1))
        return r.view(v_2.shape[0], v_2.shape[2], v_2.shape[3], v_2.shape[2], -1)
