import torch
import torch.nn as nn
import torch.nn.functional as F


class Con_Loss(torch.nn.Module):
  def __init__(self, kernel_size=7, sigma=0.25, threshold=0.5, crit='mse'):
    super(Con_Loss, self).__init__()
    ks = kernel_size
    indices_x, indices_y = torch.meshgrid(torch.range(0, ks-1), torch.range(0, ks-1))  
    indices = torch.cat([indices_x.unsqueeze(0), indices_y.unsqueeze(0)], dim=0)
    indices = indices - (ks - 1)/2
    indices = indices.reshape(2, -1).unsqueeze(-1).unsqueeze(-1)

    self.kernel = nn.Parameter(indices.unsqueeze(0), requires_grad=False)
    gather = torch.zeros(1, ks * ks, ks, ks)
    
    for i in range(ks):
        for j in range(ks):
            gather[0, i*ks+j, i, j] = 1
    
    self.gather = nn.Parameter(gather, requires_grad=False)
    self.padding = int((ks-1) /2)
    self.sigma = sigma
    self.thresh = threshold

    if crit == 'mse':
      self.crit = torch.nn.MSELoss(reduction='none')
      print('using {} for computing consistency loss'.format(crit))
    elif crit == 'kld':
      self.crit = torch.nn.KLDivLoss(reduction='sum')
      print('using {} for computing consistency loss'.format(crit))
    else:
      self.crit = torch.nn.MSELoss(reduction='sum')
      print('using {} for computing consistency loss'.format(crit))

  def forward(self, heatmap, offset, target):
    offset = -(offset.unsqueeze(2) - self.kernel).norm(dim=1) / self.sigma
    weights = torch.exp(offset) #  * 1 / (2 * np.pi * opt.consistency_sigma)
    # weights = weights / torch.max(weights)
    mask = target > self.thresh
    heatmap_moved = F.conv2d(heatmap.mul(weights), self.gather, padding=self.padding).clamp(min=0.0001, max=0.9999)
    return torch.mean(mask * self.crit(heatmap_moved, target))# / torch.sum(target > 0.95)



    
class trackLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mse = nn.MSELoss()
        self.consistency = Con_Loss(cfg.kernel_size, cfg.sigma, cfg.threshold, cfg.crit)
        self.pre = cfg.pre
        self.norm = cfg.norm
        self.consistency_weights = cfg.consistency_weights
    
    
    
    def forward(self, yhat, y):
        offset = yhat[:, 0:2, :, :]
        pre_hm = y[:, 0:1, :, :]
        if self.pre:
            mse_loss = self.mse(yhat, y)
            pre_hm = yhat[:, 0:1, :, :]
            offset = yhat[:, 1:3, :, :]
            if self.norm:
                cur_hm = torch.norm(yhat[:, 1:, :, :], dim=(1), keepdim=True)
            else:
                cur_hm = yhat[:, -1, :, :]
        elif self.norm:
            cur_hm = torch.norm(yhat, dim=(1), keepdim=True)
            mse_loss = self.mse(yhat, y[:, 1:, :, :])
        else:
            cur_hm = yhat[:, -1, :, :]
            mse_loss = self.mse(yhat, y[:, 1:, :, :])
        
        consistency_loss = self.consistency(cur_hm, offset, pre_hm)
            
        return [self.consistency_weights * consistency_loss, mse_loss]