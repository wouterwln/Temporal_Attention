import torch
from torch import nn
import torch.nn.functional as F


class TemporalAttentionModule(nn.Module):
    """
    This class is responsible for creating a Temporal Attention Module with 5 repetitions of the stabbing mechanism
    The amount of repetitions and depth of every stabbing step is hardcoded. A further refinement would be to make
    these hyperparameters.
    """

    def __init__(self, input_channels):
        super().__init__()

        self.visual_attention = VisualAttentionHead(input_channels)

        self.temporal_attention_head_1 = TemporalAttentionHead(input_channels)
        self.temporal_attention_head_2 = TemporalAttentionHead(input_channels)
        self.temporal_attention_head_3 = TemporalAttentionHead(input_channels)
        self.temporal_attention_head_4 = TemporalAttentionHead(input_channels)
        self.temporal_attention_head_5 = TemporalAttentionHead(input_channels)

    def forward(self, x):
        """
        Default forward pass of the Temporal Attention Module. Visual Attention here is the most time consuming step
        Similarly to my report, we concatenate the result of every step at the end in order to enrich the feature space
        """
        x = self.visual_attention(x)
        x_1 = self.temporal_attention_head_1(x)
        x_2 = self.temporal_attention_head_2(x_1)
        x_3 = self.temporal_attention_head_3(x_2)
        x_4 = self.temporal_attention_head_4(x_3)
        x_5 = self.temporal_attention_head_5(x_4)

        return torch.cat((x, x_1, x_2, x_3, x_4, x_5), dim=1)


class VisualAttentionHead(nn.Module):
    """
    This class is responsible for calculating traditional visual attention
    """

    def __init__(self, input_channels, affinity="softmax"):
        super().__init__()

        self.query_layer = nn.Conv2d(input_channels, input_channels // 8, 1)
        self.key_layer = nn.Conv2d(input_channels, input_channels // 8, 1)
        self.value_layer = nn.Conv2d(input_channels, input_channels // 2, 1)

        self.combiner = nn.Conv2d(input_channels // 2, input_channels // 2, 1)
        self.affinity = nn.Softmax(dim=3)

    def forward(self, x):
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = F.relu(self.value_layer(x))

        x = self.elementwise_dot_product(q, k)
        x = self.affinity(x.view(x.shape[0], x.shape[1], x.shape[2], -1))
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[1], x.shape[2])
        x = torch.mul(x.unsqueeze(-1), v.permute(0, 2, 3, 1).unsqueeze(1).unsqueeze(1)).sum((3, 4)).permute(0, 3, 1, 2)
        x = self.combiner(x)

        return torch.cat((x, v), dim=1)

    @staticmethod
    def elementwise_dot_product(v_1, v_2):
        """
        In visual attention, we have to compute the dot product between all pairs of pixels (for the keys and values).
        This function calculates this elementwise dot product between two tensors.
        :param v_1: The first tensor of values
        :param v_2: The second tensor of values
        :return: The elementwise dot product
        """
        v_1 = v_1.permute(0, 2, 3, 1)
        r = torch.matmul(v_1.view(v_1.shape[0], -1, v_1.shape[3]), v_2.view(v_2.shape[0], v_2.shape[1], -1))
        return r.view(v_2.shape[0], v_2.shape[2], v_2.shape[3], v_2.shape[2], -1)


class TemporalAttentionHead(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.query_layer = nn.Conv2d(input_channels, input_channels // 8, 1)
        self.key_layer = nn.Conv2d(input_channels, input_channels // 8, 1)
        self.value_layer = nn.Conv2d(input_channels, input_channels // 2, 1)
        self.affinity = nn.Softmax(dim=1)

        self.combiner = nn.Conv2d(input_channels // 2, input_channels // 2, 1)

    def forward(self, x):
        """
        In this forward pass, which is slightly more involved, we first transform the key tensor to an nx3xcxwxh tensor
        where every frame also has data on the two preceding frames. We can then use matrix multiplication to map the
        queries to the keys. By applying a similar trick to the values to transform them to nx3xcxwxh shape we can
        use elementwise multiplication and aggregation to obtain the reweighted attention feature map.
        :param x:
        :return:
        """
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = F.relu(self.value_layer(x))

        k = self._precede_with_zeroes(k)
        k = k.permute(0, 3, 4, 1, 2)
        q = q.permute(0, 2, 3, 1).unsqueeze(-1)
        k = k.reshape(k.shape[0] * k.shape[1] * k.shape[2], k.shape[3], k.shape[4])
        q = q.reshape(q.shape[0] * q.shape[1] * q.shape[2], q.shape[3], q.shape[4])
        x = torch.bmm(k, q).view(q.shape[0], k.shape[1])
        x = self.affinity(x)
        x = x.reshape(v.shape[0], v.shape[2], v.shape[3], k.shape[1], 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.mul(x, self._precede_with_zeroes(v)).sum(1)
        x = self.combiner(x)
        return torch.cat((x, v), dim=1)

    def _precede_with_zeroes(self, x):
        """
        This function is concerned with performing the trick that speeds up the rest of the computations: By grouping
        all relevant data in single elements of the tensor we can speed the rest up using default batched matrix multiplications
        This function transforms an nxcxwxh tensor in a nx3xcxwxh tensor and appends zeros at the beginning to ensure that
        we have compatible shapes.
        """
        k = x.unsqueeze(1)
        zeros = torch.zeros((1, 1) + x[0].shape).to("cuda")
        k_2 = torch.cat((torch.cat((zeros, k[:-1]), dim=0), k), dim=1)
        zeros = torch.zeros((2, 1) + x[0].shape).to("cuda")
        x = torch.cat((torch.cat((zeros, k[:-2]), dim=0), k_2), dim=1)
        return x
