import torch
import torch.nn as nn
import torch.nn.functional as F



class RMSE_Q_NormLoss(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.mse = nn.MSELoss()
        self.q = q

    def forward(self, yhat, y):
        yhat_norm = yhat.pow(2).sum(dim=1, keepdim=True).sqrt()
        y_norm = y.pow(2).sum(dim=1, keepdim=True).sqrt()
        dis = y_norm - yhat_norm
        dis_q = torch.max((self.q - 1) * dis, self.q * dis)
        dis_q_mse = torch.mean((dis_q) ** 2)

        return self.mse(yhat, y) + dis_q_mse
        # return torch.sqrt(self.mse(yhat, y)) + torch.sqrt(dis_q_mse)

class trackLoss(nn.Module):
    def __init__(self, kernel_size=7, sigma=0.25, threshold=0.5, crit='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.consistency = Con_Loss(kernel_size, sigma, threshold, crit)
        
        
    def forward(self, yhat, y):
        if y.shape[1] > 3: # output predetection
            pre_hm = yhat[:, 0, :, :]
            cur_hm = torch.norm(yhat[:, 1:, :, :], dim=(1))
            offset = 
            
        return self.mse(yhat, y) 
    

# class regtrackLoss(nn.Module):
#     def __init__(self, axis_size, th, z_value, device):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.cos = nn.CosineSimilarity()
#         self.th = th
#         ys, xs = torch.meshgrid(torch.arange(axis_size[0]), torch.arange(axis_size[1]))
#         self.cord = torch.cat([ys.unsqueeze(0), xs.unsqueeze(0)], 0).unsqueeze(0).to(device)
        
#     def forward(self, yhat, y):        
#         det1 = yhat[:, 0:1, :, :]
#         det2 = torch.norm(yhat[:, 1:, :, :], dim=1, keepdim=True)
#         pos_mask = det2 > self.th
#         pos_vec = yhat[:, 2:4, :, :] * pos_mask
        
#         ##### pos mask indicate that det2 + vec must have high mag in the det1
#         mag1 = det1[(self.cord + pos_vec)[pos_mask]]
#         mag2 = det2[(self.cord - pos_vec)[pos_mask]]
#         mag1loss = - torch.mean(torch.log(mag1) + )
#         reg = 
        
#         # cosloss = (1 - self.cos(vec_hat, vec).unsqueeze(1)) * mag[:, 1:2, :, :]
#         self.cord + yhat / det2
#         cord1_pred = self.cord * (f1>self.th).float()
#         cord2_pred = self.cord * (f2>self.th).float()
        
#         reg = torch.norm(cord1_pred - cord2_pred - vec, dim=1) 
#         return self.mse(yhat, y) + torch.mean(reg) #+ torch.mean(cosloss)

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
      self.crit = torch.nn.MSELoss(reduction='sum')
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
    mask = heatmap > self.thresh
    heatmap_moved = F.conv2d(mask * heatmap.mul(weights), self.gather, padding=self.padding).clamp(min=0.0001, max=0.9999)
    return self.crit(heatmap_moved, target) / torch.sum(target > 0.95)

