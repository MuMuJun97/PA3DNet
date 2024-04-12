from functools import partial
import torch.nn.functional as F
import torch.nn as nn
import torch
from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

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

class AttentionFusion_MLP(nn.Module):

    def __init__(self,dim=16):
        super(AttentionFusion_MLP, self).__init__()
        self.mlp1 = MLP(input_dim=dim,hidden_dim=dim*2,output_dim=dim,num_layers=2)
        self.mlp2 = MLP(input_dim=dim,hidden_dim=dim*2,output_dim=dim,num_layers=2)
        self.mlp3 = MLP(input_dim=dim, hidden_dim=dim * 2, output_dim=dim, num_layers=2)
        self.fc_fuse = nn.Sequential(
            nn.Linear(dim*2,dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(dim*2,dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(dim*2,1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            # nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            # nn.ReLU()
        )

        self.conv_fuse = torch.nn.Conv1d(dim + dim, dim, 1)
        self.conv_fuse_2 = nn.Sequential(
            nn.Linear(dim*2,dim*2),
            nn.ReLU(),
            nn.Linear(dim*2,dim),
            nn.ReLU(),
            nn.Linear(dim,dim)
        )
        # self.bn1 = torch.nn.BatchNorm1d(dim)
    def forward(self,f_img,f_pt):
        # feat_img (N,C), feat_pt (N,C)
        feat_img = self.mlp1(f_img)
        feat_pt = self.mlp2(f_pt)

        # (n,c1+c2)
        fuse_feat = torch.cat([feat_img,feat_pt],dim=1)

        fuse_feat = self.fc_fuse(fuse_feat)
        att = F.sigmoid(fuse_feat)
        att = att.transpose(0, 1).contiguous().view(1, 1, -1)

        # (N,C)-->(1,C,N)
        img_features = self.conv1(f_img.unsqueeze(0).transpose(1,2).contiguous())
        point_features = self.conv2(f_pt.unsqueeze(0).transpose(1,2).contiguous())
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = self.conv_fuse(fusion_features)

        fusion_features = fusion_features * att

        fusion_features = fusion_features.squeeze(0).transpose(0,1).contiguous()

        pt_feats = self.mlp3(f_pt)

        final_feat = torch.cat([pt_feats,fusion_features],dim=1)

        final_feat = self.conv_fuse_2(final_feat)

        return final_feat

class AttentionFusion(nn.Module):
    def __init__(self,dim=16):
        super(AttentionFusion, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, 1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

        self.conv_fuse = torch.nn.Conv1d(dim + dim, dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(dim)
    def forward(self,f_img,f_pt):
        # feat_img (N,C), feat_pt (N,C)
        feat_img = self.fc1(f_img)
        feat_pt = self.fc2(f_pt)

        feat_fuse = self.fc3(F.tanh(feat_img + feat_pt))
        att = F.sigmoid(feat_fuse)

        # (N,C)-->(1,C,N)
        feat_img_new = self.conv1(f_img.unsqueeze(0).transpose(1,2).contiguous())
        att = att.transpose(0,1).contiguous().view(1,1,-1)
        img_features = feat_img_new * att

        point_features = self.conv2(f_pt.unsqueeze(0).transpose(1,2).contiguous())

        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv_fuse(fusion_features)))

        return fusion_features.squeeze(0).transpose(0,1).contiguous()

class AttentionFusion_ADD(nn.Module):
    def __init__(self,dim=16):
        super(AttentionFusion_ADD, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Tanh(),
            nn.Linear(dim,dim),
            nn.BatchNorm1d(dim),
            nn.Tanh(),
            nn.Linear(dim,1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

        self.conv_fuse = torch.nn.Conv1d(dim + dim, dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(dim)
    def forward(self,f_img,f_pt):
        # feat_img (N,C), feat_pt (N,C)
        feat_img = self.fc1(f_img)
        feat_pt = self.fc2(f_pt)

        # att
        feat_fuse = self.fc3(feat_img + feat_pt)

        att = F.sigmoid(feat_fuse)

        # (N,C)-->(1,C,N)
        feat_img_new = self.conv1(f_img.unsqueeze(0).transpose(1,2).contiguous())
        att = att.transpose(0,1).contiguous().view(1,1,-1)
        img_features = feat_img_new * att

        point_features = self.conv2(f_pt.unsqueeze(0).transpose(1,2).contiguous())

        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv_fuse(fusion_features)))

        return fusion_features.squeeze(0).transpose(0,1).contiguous()


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        block = post_act_block


        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(4, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        if input_channels > 4:
            attn_func = model_cfg.get("attn_func",'AttentionFusion')
            self.fusion = True

            img_dim = input_channels - 4
            self.conv_input_i = spconv.SparseSequential(
                spconv.SubMConv3d(img_dim, 16, 3, padding=1, bias=False, indice_key='subm1_i'),
                norm_fn(16),
                nn.ReLU(),
            )
            self.conv1_i = spconv.SparseSequential(
                block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1_i'),
            )

            if attn_func == 'AttentionFusion':
                self.fuse_layer = AttentionFusion()
            elif attn_func == 'AttentionFusion_MLP':
                self.fuse_layer = AttentionFusion_MLP()
            elif attn_func == "AttentionFusion_ADD":
                self.fuse_layer = AttentionFusion_ADD()
            else:
                self.fuse_layer = AttentionFusion()

        else:
            self.fusion = False

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        if self.fusion:
            voxel_features = batch_dict['voxel_features'][:,0:4]
            img_features = batch_dict['voxel_features'][:,4:]
        else:
            voxel_features = batch_dict['voxel_features']

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)

        if self.fusion:
            img_sp_tensor = spconv.SparseConvTensor(
                features=img_features,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            x_i = self.conv_input_i(img_sp_tensor)
            x_conv1_i = self.conv1_i(x_i)

            feat_img = x_conv1_i.features
            feat_pt = x_conv1.features

            fuse_features = self.fuse_layer(feat_img,feat_pt)

            fuse_sp_tensor = spconv.SparseConvTensor(
                features=fuse_features,
                indices=x_conv1.indices,
                spatial_shape=x_conv1.spatial_shape,
                batch_size=batch_size
            )
            x_conv1 = fuse_sp_tensor

        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict
