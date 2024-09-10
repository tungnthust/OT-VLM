import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def Sinkhorn(K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        max_iter = 100
        
        for i in range(max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, local_image_features, local_text_features):
        
        # print(f"Local image features: {local_image_features.shape}") #[4, 3, 49, 512]
        # print(f"Local text features: {local_text_features.shape}")   #[4, 3, 76, 512]
        ot_loss = 0
        device = image_features.device
        # Subject, Object, Action
        channels = 3
        logits_per_image_list = []
        logits_per_text_list = []

        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

        for i in range(channels):
            image_channel = image_features[:, i, :]
            text_channel = text_features[:, i, :]
            local_image_channel = local_image_features[:,i,:,:] # shape = [4,49,512]
            local_text_channel = local_text_features[:,i,:,:]  #shape = [4,76,512]
            local_image_channel =  F.normalize(local_image_channel, dim=2)
            local_text_channel = F.normalize(local_text_channel,dim=2)
            sim  = torch.einsum('mbd,mcd->mbc', local_image_channel, local_text_channel).contiguous() #shape= [4,49,76] --> chỗ này cần [4,4,49,76]
            wdist = 1.0 - sim
            xx=torch.zeros(local_image_channel.shape[0],local_image_channel.shape[1], dtype=sim.dtype, device=sim.device).fill_(1. / 49) #shape 128*100, 49 = 128000, 49
            yy=torch.zeros(local_text_channel.shape[0],local_text_channel.shape[1], dtype=sim.dtype, device=sim.device).fill_(1. / 76) #shape = 128000, 4
            eps = 0.1
            max_iter = 100
                    

            with torch.no_grad():
                KK = torch.exp(-wdist / eps)
                T = Sinkhorn(KK,xx,yy)  #shape 12800,49,4
            if torch.isnan(T).any():
                print("None")
            sim_op = torch.sum(T * sim, dim=(1, 2))
            print(f"Shape of sim_op is {sim_op}")
            ot_loss += torch.sum(sim_op)
            if self.world_size > 1: # = 1
                all_image_channel = all_image_features[:, i, :]
                all_text_channel = all_text_features[:, i, :]

                if self.local_loss: #False
                    logits_per_image_channel = logit_scale * image_channel @ all_text_channel.T
                    logits_per_text_channel = logit_scale * text_channel @ all_image_channel.T
                else:
                    logits_per_image_channel = logit_scale * all_image_channel @ all_text_channel.T
                    logits_per_text_channel = logits_per_image_channel.T
            else: #Run this
                logits_per_image_channel = logit_scale * image_channel @ text_channel.T
                logits_per_text_channel = logit_scale * text_channel @ image_channel.T

            logits_per_image_list.append(logits_per_image_channel)
            logits_per_text_list.append(logits_per_text_channel)

        # Taking the mean of the logits computed for each channel
        logits_per_image = sum(logits_per_image_list) / channels
        logits_per_text = sum(logits_per_text_list) / channels


        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        print(f"OT loss is: {ot_loss}")
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2 + ot_loss
        print(f"Shape of total_loss: {total_loss.shape}")
        print(f"total_loss: {total_loss}")
        return total_loss