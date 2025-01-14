# core/model/meta/insta_protonet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from meta_model import MetaModel
from core.utils import accuracy
from ..backbone.resnet_12 import resnet12

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 1, 5, 0, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 4, 0, 5, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, sigma, k, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.sigma = sigma
        self.k = k
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 5) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 5) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 5x5 frequency space
        # eg, (2,2) in 10x10 is identical to (1,1) in5x5

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel*self.sigma), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel*self.sigma), channel*self.k**2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, self.k, self.k)
        # return x * y.expand_as(x)
        return y

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter
class INSTA(nn.Module):
    def __init__(self, c, spatial_size, sigma, k, args, **kwargs):
        super().__init__()
        # TODO
        self.channel = c
        self.h1 = sigma
        self.h2 = k ** 2
        self.k = k

        self.conv = nn.Conv2d(c, self.h2, 1)
        self.fn_partial = nn.BatchNorm2d(spatial_size ** 2)
        self.fn_channel = nn.BatchNorm2d(c)
        self.Unfold = nn.Unfold(kernel_size=k, padding=int((self.k + 1) / 2 - 1))
        self.spatial_size = spatial_size

        c2wh = {512: 11, 640: self.spatial_size}

        self.channel_att = MultiSpectralAttentionLayer(
            c, c2wh[c], c2wh[c], sigma=self.h1, k=self.k, freq_sel_method='low16'
        )

        self.args = args

        self.CLM_upper = nn.Sequential(
            nn.Conv2d(c, c * 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),
            nn.Conv2d(c * 2, c * 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU()
        )

        self.CLM_lower = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),
            nn.Conv2d(c * 2, c, 1),
            nn.BatchNorm2d(c),
            nn.Sigmoid()  # Sigmoid activation to normalize the feature values between 0 and 1.
        )

    def CLM(self, featuremap):
        adap = self.CLM_upper(featuremap)
        intermediate = adap.sum(dim=0)  # Summing features across the batch dimension.
        adap_1 = self.CLM_lower(intermediate.unsqueeze(0))  # Applying the lower CLM.
        return adap_1

    def spatial_kernel_network(self, feature_map, conv):
        spatial_kernel = conv(feature_map)
        spatial_kernel = spatial_kernel.flatten(-2).transpose(-1, -2)
        size = spatial_kernel.size()
        spatial_kernel = spatial_kernel.view(size[0], -1, self.k, self.k)
        spatial_kernel = self.fn_partial(spatial_kernel)

        spatial_kernel = spatial_kernel.flatten(-2)
        return spatial_kernel

    def channel_kernel_network(self, feature_map):
        channel_kernel = self.channel_att(feature_map)
        channel_kernel = self.fn_channel(channel_kernel)
        channel_kernel = channel_kernel.flatten(-2)
        channel_kernel = channel_kernel.squeeze().view(channel_kernel.shape[0], self.channel, -1)
        return channel_kernel

    def unfold(self, x, padding, k):
        # Custom unfold operation
        x_padded = torch.cuda.FloatTensor(
            x.shape[0], x.shape[1],
            x.shape[2] + 2 * padding,
            x.shape[3] + 2 * padding
        ).fill_(0)
        x_padded[:, :, padding:-padding, padding:-padding] = x
        x_unfolded = torch.cuda.FloatTensor(*x.shape, k, k).fill_(0)
        for i in range(int((self.k + 1) / 2 - 1), x.shape[2] + int((self.k + 1) / 2 - 1)):
            for j in range(int((self.k + 1) / 2 - 1), x.shape[3] + int((self.k + 1) / 2 - 1)):
                x_unfolded[:, :, i - int(((self.k + 1) / 2 - 1)),
                           j - int(((self.k + 1) / 2 - 1)), :, :] = x_padded[:, :, 
                           i - int(((self.k + 1) / 2 - 1)):i + int((self.k + 1) / 2),
                           j - int(((self.k + 1) / 2 - 1)):j + int((self.k + 1) / 2)]
        return x_unfolded

    def forward(self, x):
        # Forward pass for INSTA
        spatial_kernel = self.spatial_kernel_network(x, self.conv).unsqueeze(-3)
        channel_kernel = self.channel_kernel_network(x).unsqueeze(-2)
        kernel = spatial_kernel * channel_kernel  # Combine spatial and channel kernels

        # Resize kernel and apply to the unfolded feature map
        kernel_shape = kernel.size()
        feature_shape = x.size()
        instance_kernel = kernel.view(
            kernel_shape[0], kernel_shape[1],
            feature_shape[-2], feature_shape[-1],
            self.k, self.k
        )

        # Get task-specific representation
        task_s = self.CLM(x)
        spatial_kernel_task = self.spatial_kernel_network(task_s, self.conv).unsqueeze(-3)
        channel_kernel_task = self.channel_kernel_network(task_s).unsqueeze(-2)
        task_kernel = spatial_kernel_task * channel_kernel_task

        task_kernel_shape = task_kernel.size()
        task_kernel = task_kernel.view(
            task_kernel_shape[0], task_kernel_shape[1],
            feature_shape[-2], feature_shape[-1],
            self.k, self.k
        )

        kernel = task_kernel * instance_kernel
        unfold_feature = self.unfold(x, int((self.k + 1) / 2 - 1), self.k)  # Perform a custom unfold operation
        adapted_feature = (unfold_feature * kernel).mean(dim=(-1, -2)).squeeze(-1).squeeze(-1)
        return adapted_feature + x, task_kernel  # Return the normal training output and task-specific kernel


class INSTA_ProtoNet(MetaModel):
    def __init__(self, args, init_type="normal", **kwargs):
        super().__init__(init_type, **kwargs)
        self.args = args
        self.feature = resnet12()
        self.loss_func = nn.CrossEntropyLoss()
        self.INSTA = INSTA(640, 5, 0.2, 3, args=args)
        self.classifier = nn.Sequential(
            nn.Linear(640, self.args.way)
        )

    def inner_loop(self, proto, support):
        """
        Performs an inner optimization loop to fine-tune prototypes on support sets during meta-training.

        Parameters:
        - proto: Initial prototypes, typically the mean of the support embeddings.
        - support: Support set embeddings used for fine-tuning the prototypes.

        Returns:
        - SFC: Updated (fine-tuned) prototypes.
        """
        # Clone and detach prototypes to prevent gradients from accumulating across episodes.
        SFC = proto.clone().detach()
        SFC = nn.Parameter(SFC, requires_grad=True)

        # Initialize an SGD optimizer specifically for this inner loop.
        optimizer = torch.optim.SGD([SFC], lr=0.6, momentum=0.9, dampening=0.9, weight_decay=0)

        # Create labels for the support set, used in cross-entropy loss during fine-tuning.
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label_shot = label_shot.type(torch.cuda.LongTensor)

        # Perform gradient steps to update the prototypes.
        with torch.enable_grad():
            for k in range(50):  # Number of gradient steps.
                rand_id = torch.randperm(self.args.way * self.args.shot).cuda()
                for j in range(0, self.args.way * self.args.shot, 4):
                    selected_id = rand_id[j: min(j + 4, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.classifier(batch_shot.detach(), SFC)
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    def classifier(self, query, proto):
        """
        Simple classifier that computes the negative squared Euclidean distance between query and prototype vectors,
        scaled by a temperature parameter for controlling the sharpness of the distribution.

        Parameters:
        - query: Query set embeddings.
        - proto: Prototype vectors.

        Returns:
        - logits: Logits representing similarity scores between each query and each prototype.
        """
        logits = -torch.sum((proto.unsqueeze(0) - query.unsqueeze(1)) ** 2, 2) / self.args.temperature
        return logits.squeeze()

    def _forward(self, instance_embs, support_idx, query_idx):
        """
        前向传播方法，处理支持集和查询集的特征，计算分类结果。

        功能分区：
        1. 支持集和查询集特征提取与适应
        2. 生成适应后的原型
        3. 查询集特征的适应与分类
        4. 返回分类输出和任务特定内核
        """
        # 1. 支持集和查询集特征提取与适应
        emb_dim = instance_embs.size()[-3:]
        channel_dim = emb_dim[0]

        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + emb_dim))
        query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + emb_dim))
        num_samples = support.shape[1]
        num_proto = support.shape[2]
        support = support.squeeze()

        adapted_s, task_kernel = self.INSTA(support.view(-1, *emb_dim))
        query = query.view(-1, *emb_dim)

        # 2. 生成适应后的原型
        adapted_proto = adapted_s.view(num_samples, -1, *adapted_s.shape[1:]).mean(0)
        adapted_proto = nn.AdaptiveAvgPool2d(1)(adapted_proto).squeeze(-1).squeeze(-1)

        # 3. 查询集特征的适应与分类
        query_ = nn.AdaptiveAvgPool2d(1)(
            (self.INSTA.unfold(query, int((task_kernel.shape[-1] + 1) / 2 - 1), task_kernel.shape[-1]) * task_kernel)
        ).squeeze()
        query = query + query_
        adapted_q = nn.AdaptiveAvgPool2d(1)(query).squeeze(-1).squeeze(-1)

        # 4. 微调原型（仅在测试阶段）
        if self.args.testing:
            adapted_proto = self.inner_loop(
                adapted_proto,
                nn.AdaptiveAvgPool2d(1)(support).squeeze().view(num_proto * num_samples, channel_dim)
            )

        # 5. 分类
        logits = self.classifier(adapted_q, adapted_proto)

        if self.training:
            reg_logits = None
            return logits, reg_logits
        else:
            return logits

    def set_forward(self, batch):
        """
        推理阶段调用，返回分类输出以及准确率。

        参数：
        - batch: 输入数据批次，包含图像和标签。

        返回：
        - output: 分类输出。
        - acc: 准确率。
        """
        image, global_target = batch
        image = image.to(self.device)
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        output = self._forward(feat, support_feat, query_feat)
        acc = accuracy(output.squeeze(), query_target.reshape(-1))
        return output, acc

    def set_forward_loss(self, batch):
        """
        训练阶段调用，返回分类输出、准确率以及前向损失。

        参数：
        - batch: 输入数据批次，包含图像和标签。

        返回：
        - output: 分类输出。
        - acc: 准确率。
        - loss: 计算得到的损失。
        """
        start_tm = time.time()
        image, global_target = batch
        image = image.to(self.device)
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)

        # 调用 _forward 获取 logits 和可能的 reg_logits
        logits, reg_logits = self._forward(feat, support_feat, query_feat)

        # 计算损失
        if reg_logits is not None:
            loss = self.loss_func(logits, query_target)
            # 假设有辅助标签 `label_aux`，这里需要根据具体情况调整
            # 例如，假设 label_aux 是另一组标签，可以通过 `batch` 获取
            label_aux = global_target  # 示例，实际应根据数据结构获取
            total_loss = self.args.balance_1 * loss + self.args.balance_2 * self.loss_func(reg_logits, label_aux)
        else:
            loss = self.loss_func(logits, query_target)
            total_loss = loss

        # 计算准确率
        acc = accuracy(logits, query_target)

        return logits, acc, total_loss
