import torch.nn as nn
import torch.nn.functional as F
import torch

from meta_model import MetaModel
from core.utils import accuracy
from backbone.fcanet import MultiSpectralAttentionLayer
class INSTA(nn.Module):
    def __init__(self, c, spatial_size, sigma, k ,args, **kwargs):
        super().__init__()
        # TODO
        self.channel = c
        self.h1 = sigma
        self.h2 = k **2
        self.k = k
        
        self.conv = nn.Conv2d(c, self.h2, 1)
        self.fn_partial = nn.BatchNorm2d(spatial_size**2)
        self.fn_channel = nn.BatchNorm2d(c)
        self.Unfold = nn.Unfold(kernel_size=k, padding=int((self.k+1)/2-1))
        self.spatial_size = spatial_size
        
        c2wh = dict([(512, 11), (640, self.spatial_size)])
        
        self.channel_att = MultiSpectralAttentionLayer(c, c2wh[c], c2wh[c], sigma=self.h1, k=self.k, freq_sel_method='low16')
        
        self.args = args
        
        self.CLM_upper = nn.Sequential(
            nn.Conv2d(c, c*2, 1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(),
            nn.Conv2d(c*2, c*2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU()
        )
        
        self.CLM_lower = nn.Sequential(
            nn.Conv2d(c*2, c*2, 1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(),
            nn.Conv2d(c*2, c, 1),
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
        spatial_kernel = self.fn_spatial(spatial_kernel)

        spatial_kernel = spatial_kernel.flatten(-2)
        return spatial_kernel
    
    def channel_kernel_network(self, feature_map):
        channel_kernel = self.channel_att(feature_map)
        channel_kernel = self.fn_channel(channel_kernel)
        channel_kernel = channel_kernel.flatten(-2)
        channel_kernel = channel_kernel.squeeze().view(channel_kernel.shape[0], self.channel, -1)
        return channel_kernel
    def unfold(self, x, padding, k):

        x_padded = torch.cuda.FloatTensor(x.shape[0], x.shape[1], x.shape[2] + 2 * padding, x.shape[3] + 2 * padding).fill_(0)
        x_padded[:, :, padding:-padding, padding:-padding] = x
        x_unfolded = torch.cuda.FloatTensor(*x.shape, k, k).fill_(0)
        for i in range(int((self.k+1)/2-1), x.shape[2] + int((self.k+1)/2-1)): 
            for j in range(int((self.k+1)/2-1), x.shape[3] + int((self.k+1)/2-1)):
                x_unfolded[:, :, i - int(((self.k+1)/2-1)), j - int(((self.k+1)/2-1)), :, :] = x_padded[:, :, i-int(((self.k+1)/2-1)):i + int((self.k+1)/2), j - int(((self.k+1)/2-1)):j + int(((self.k+1)/2))]
        return x_unfolded
    def forward(self, x):
        spatial_kernel = self.spatial_kernel_network(x, self.conv).unsqueeze(-3)
        channel_kernenl = self.channel_kernel_network(x).unsqueeze(-2)
        kernel = spatial_kernel * channel_kernenl  # Combine spatial and channel kernels
        # Resize kernel and apply to the unfolded feature map
        kernel_shape = kernel.size()
        feature_shape = x.size()
        instance_kernel = kernel.view(kernel_shape[0], kernel_shape[1], feature_shape[-2], feature_shape[-1], self.k, self.k)
        task_s = self.CLM(x)  # Get task-specific representation
        spatial_kernel_task = self.spatial_kernel_network(task_s, self.conv).unsqueeze(-3)
        channel_kernenl_task = self.channel_kernel_network(task_s).unsqueeze(-2)
        task_kernel = spatial_kernel_task * channel_kernenl_task
        task_kernel_shape = task_kernel.size()
        task_kernel = task_kernel.view(task_kernel_shape[0], task_kernel_shape[1], feature_shape[-2], feature_shape[-1], self.k, self.k)
        kernel = task_kernel * instance_kernel
        unfold_feature = self.unfold(x, int((self.k+1)/2-1), self.k)  # Perform a custom unfold operation
        adapted_feauture = (unfold_feature * kernel).mean(dim=(-1, -2)).squeeze(-1).squeeze(-1)
        return adapted_feauture + x, task_kernel  # Return the normal training output and task-specific kernel

class INSTA_ProtoNet(MetaModel):
    def __init__(self,args, init_type="normal",  **kwargs):
        super().__init__(init_type, **kwargs)
        self.args = args
        
        self.INSTA = INSTA(640, 5, 0.2, 3, args=args)
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

        emb_dim = instance_embs.size()[-3:]
        channel_dim = emb_dim[0]

        # Organize support and query data based on indices, and reshape accordingly.
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + emb_dim))
        query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + emb_dim))
        num_samples = support.shape[1]
        num_proto = support.shape[2]
        support = support.squeeze()

        # Adapt support features using the INSTA model and average to form adapted prototypes.
        adapted_s, task_kernel = self.INSTA(support.view(-1, *emb_dim))
        query = query.view(-1, *emb_dim)
        adapted_proto = adapted_s.view(num_samples, -1, *adapted_s.shape[1:]).mean(0)
        adapted_proto = nn.AdaptiveAvgPool2d(1)(adapted_proto).squeeze(-1).squeeze(-1)

        # Adapt query features using the INSTA unfolding and kernel multiplication approach.
        query_ = nn.AdaptiveAvgPool2d(1)((self.INSTA.unfold(query, int((task_kernel.shape[-1]+1)/2-1), task_kernel.shape[-1]) * task_kernel)).squeeze()
        query = query + query_
        adapted_q = nn.AdaptiveAvgPool2d(1)(query).squeeze(-1).squeeze(-1)

        # Optionally perform an inner loop optimization during testing.
        if self.args.testing:
            adapted_proto = self.inner_loop(adapted_proto, nn.AdaptiveAvgPool2d(1)(support).squeeze().view(num_proto*num_samples, channel_dim))
        
        # Classify using the adapted prototypes and query embeddings.
        logits = self.classifier(adapted_q, adapted_proto)

        if self.training:
            reg_logits = None
            return logits, reg_logits
        else:
            return logits
        
    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=1)
        output, reg_output = self._forward(feat, support_feat, query_feat)
        acc = accuracy(output.squeeze(), query_target.reshape(-1))
        return output, acc