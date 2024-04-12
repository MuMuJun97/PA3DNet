import torch.nn.functional as F
import torch.nn as nn
import torch

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

mlp1 = MLP(input_dim=16,hidden_dim=32,output_dim=16,num_layers=2)

n = 1000
c = 16
feat_img = torch.rand((n,c))
feat_pt = torch.rand((n,c))

feat_img = mlp1(feat_img)
feat_pt = mlp1(feat_pt)

# (n,c1+c2)
cat_feat = torch.cat([feat_img,feat_pt],dim=1)

cat_mlp = nn.Sequential(
            nn.Linear(c+c, c+c),
            nn.Tanh(),
        )

cat_feat = cat_mlp(cat_feat)

fuse_mlp = MLP(input_dim=c+c,hidden_dim=c,output_dim=c,num_layers=3)
fuse_feat = fuse_mlp(cat_feat)

att = F.sigmoid(fuse_feat)


print(mlp1)
print(cat_mlp)