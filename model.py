import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleResidualBlock(nn.Module):
    def __init__(self, ch_in, mult=1):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, mult * ch_in, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mult * ch_in, mult * ch_in, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x_ = x.clone()
        x_ = torch.relu(self.conv1(x_))
        x_ = self.conv2(x_)
        x = x + x_
        return x

# adapted from Assignment 5
class SimpleSR(nn.Module):
  def __init__(self, ch_in, upsample_factor, n_blocks = 5, depth=1):
    super().__init__()
    n_feats = 64
    # self.conv0 = nn.Conv2d(ch_in,ch_in*depth,kernel_size=3,stride=1,padding=1)
    # self.residual_layers = nn.ModuleList([SimpleResidualBlock(ch_in*depth) for i in range(n_blocks)])
    # self.upsample_factor = upsample_factor
    # self.conv1 = nn.Conv2d(ch_in*depth,ch_in*depth,kernel_size=3,stride=1,padding=1)
    # self.conv2 = nn.Conv2d(ch_in*depth,ch_in*upsample_factor**2,kernel_size=3,stride=1,padding=1)
    # self.conv3 = nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=1,padding=1)
    # self.conv4 = nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=1,padding=1)
    
    # self.conv1 = nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=1,padding=1)
    self.residual_layers = nn.ModuleList([SimpleResidualBlock(n_feats) for i in range(n_blocks)])
    self.upsample_factor = upsample_factor
    self.conv2 = nn.Conv2d(ch_in,ch_in*upsample_factor**2,kernel_size=3,stride=1,padding=1)
    # self.conv3 = nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=1,padding=1)

    n_colors = 3
    
    scale = 2
    kernel_size = 3

    self.conv0 = nn.Conv2d(n_colors, n_feats, kernel_size,stride=1,padding=1)
    self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size,stride=1,padding=1)

    self.conv2 = nn.Conv2d(n_feats,n_feats*self.upsample_factor**2,kernel_size,stride=1,padding=1)
    self.conv3 = nn.Conv2d(n_feats, n_colors, kernel_size,stride=1,padding=1)



  # def forward(self, x):
  #   x = F.relu(self.conv0(x))
  #   _x = x.clone()
  #   for residual in self.residual_layers:
  #     _x = F.relu(residual(_x))
  #   _x = F.relu(self.conv1(_x))
  #   x = _x + x
  #   x = F.relu(self.conv2(x))
  #   x = F.relu(torch.nn.functional.pixel_shuffle(x,self.upsample_factor))
  #   x = F.relu(self.conv3(x))
  #   x = self.conv4(x)
  #   x = torch.clamp(x,0,1)
  #   return x
  # def forward(self, x):
  #   x = self.conv1(x)
  #   for residual in self.residual_layers:
  #     x = residual(x)
  #   x = self.conv2(x)
  #   x = torch.nn.functional.pixel_shuffle(x,self.upsample_factor)
  #   x = self.conv3(x)
  #   # x = torch.clamp(x,0.0,1.0)
  #   return x

  def forward(self, x):
      x = self.conv0(x)
      x = self.conv1(x)
      res = x.clone()
      for residual in self.residual_layers:
        res = residual(res)
      res += x
      x = self.conv2(res)
      x = torch.nn.functional.pixel_shuffle(x,self.upsample_factor)
      x = self.conv3(x)
      return x
  
  def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
