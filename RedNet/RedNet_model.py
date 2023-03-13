import torch
from torch import nn
import torch.nn.functional as F 
import math
import torch.utils.model_zoo as model_zoo
import utils
from torch.utils.checkpoint import checkpoint
import os
import torchvision


class RedNet(nn.Module):
    def __init__(self, num_classes=40, pretrained=False):

        super().__init__()

        block = Bottleneck
        transblock = TransBasicBlock
        layers = [3, 4, 6, 3]
        # original resnet
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # resnet for depth channel
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)

        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64 * 4, 64)
        self.agant2 = self._make_agant_layer(128 * 4, 128)
        self.agant3 = self._make_agant_layer(256 * 4, 256)
        self.agant4 = self._make_agant_layer(512 * 4, 512)

        # final block
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.final_deconv_custom = nn.ConvTranspose2d(self.inplanes, num_classes, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        self.out5_conv_custom = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self.out4_conv_custom = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, bias=True)
        self.out3_conv_custom = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)
        self.out2_conv_custom = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)

        if pretrained:
            self._load_resnet_pretrained()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transpose(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):

        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _load_resnet_pretrained(self):
        pretrain_dict = torch.load(ckpt, map_location='cpu')['model_state']
        prefix = 'module.'
        pretrain_dict = {
            (k[len(prefix):] if k[:len(prefix)] == prefix else k): v for k, v in pretrain_dict.items()
        }
        pretrain_dict.pop('final_deconv_custom.weight')
        pretrain_dict.pop('final_deconv_custom.bias')

        pretrain_dict.pop('out5_conv_custom.weight')
        pretrain_dict.pop('out5_conv_custom.bias')

        pretrain_dict.pop('out4_conv_custom.weight')
        pretrain_dict.pop('out4_conv_custom.bias')

        pretrain_dict.pop('out3_conv_custom.weight')
        pretrain_dict.pop('out3_conv_custom.bias')

        pretrain_dict.pop('out2_conv_custom.weight')
        pretrain_dict.pop('out2_conv_custom.bias')

        model_dict = {}
        state_dict = self.state_dict()

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward_downsample(self, rgb, depth):
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu(depth)

        fuse0 = x + depth

        x = self.maxpool(fuse0)
        depth = self.maxpool(depth)

        # block 1
        x = self.layer1(x)
        # debug_tensor('post1', depth)
        # for i, mod in enumerate(self.layer1_d):
        #     depth = mod(depth)
        #     debug_tensor(f'post1d-{i}', depth)

        depth = self.layer1_d(depth)
        # debug_tensor('post1d', depth)

        fuse1 = x + depth
        # block 2
        x = self.layer2(fuse1)
        depth = self.layer2_d(depth)
        fuse2 = x + depth
        # block 3
        x = self.layer3(fuse2)
        depth = self.layer3_d(depth)
        fuse3 = x + depth
        # block 4
        x = self.layer4(fuse3)
        depth = self.layer4_d(depth)
        fuse4 = x + depth

        agant4 = self.agant4(fuse4)

        return fuse0, fuse1, fuse2, fuse3, agant4

    def forward_upsample(self, fuse0, fuse1, fuse2, fuse3, agant4, train=False):

        # upsample 1
        x = self.deconv1(agant4)
        if train:
            out5 = self.out5_conv_custom(x)
        x = x + self.agant3(fuse3)
        # upsample 2
        x = self.deconv2(x)
        if train:
            out4 = self.out4_conv_custom(x)
        x = x + self.agant2(fuse2)
        # upsample 3
        x = self.deconv3(x)
        if train:
            out3 = self.out3_conv_custom(x)
        x = x + self.agant1(fuse1)
        # upsample 4
        x = self.deconv4(x)
        if train:
            out2 = self.out2_conv_custom(x)
        x = x + self.agant0(fuse0)
        # final
        x = self.final_conv(x)

        last_layer = x

        out = self.final_deconv_custom(x)

        if train:
            return out, out2, out3, out4, out5

        return out

    def forward(self, rgb, depth, train=False):

        fuses = self.forward_downsample(rgb, depth)

        # We only need predictions.
        # features_encoder = fuses[-1]
        # scores, features_lastlayer = self.forward_upsample(*fuses)
        scores = self.forward_upsample(*fuses, train)
        # debug_tensor('scores', scores)
        # return features_encoder, features_lastlayer, scores
        # print("scores: ", scores.shape)
        return scores


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class BatchNormalize(nn.Module):
    r"""
        I can't believe this isn't supported
        https://github.com/pytorch/vision/issues/157
    """

    def __init__(self, mean, std, device):
        super().__init__()
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def forward(self, x):
        # mean = torch.tensor(self.mean, device=x.device)
        # std = torch.tensor(self.std, device=x.device)
        # mean = self.mean[None, :, None, None]
        # std = self.std[None, :, None, None]
        # print(x.shape)
        # print(self.mean.shape)
        return (x - self.mean) / self.std
        # x.sub_(self.mean).div_(self.std)
        # return x

class RedNetResizeWrapper(nn.Module):
    def __init__(self, device, resize=True, stabilize=False):
        super().__init__()
        self.rednet = RedNet()
        self.rednet.eval()
        self.semmap_rgb_norm = BatchNormalize(
            mean=[0.493, 0.468, 0.438],
            std=[0.544, 0.521, 0.499],
            device=device
        )
        self.semmap_depth_norm = BatchNormalize(
            mean=[0.213],
            std=[0.285],
            device=device
        )
        self.pretrained_size = (480, 640)
        self.resize = resize
        self.stabilize = stabilize

    def forward(self, rgb, depth, train=False):
        r"""
            Args:
                Raw sensor inputs.
                rgb: B x H=256 x W=256 x 3
                depth: B x H x W x 1
            Returns:
                semantic: drop-in replacement for default semantic sensor. B x H x W  (no channel, for some reason)
        """
        # Not quite sure what depth is produced here. Let's just check
        if self.resize:
            _, og_h, og_w, _ = rgb.size() # b h w c
        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)

        rgb = rgb.float() / 255
        if self.resize:
            rgb = F.interpolate(rgb, self.pretrained_size, mode='bilinear')
            depth = F.interpolate(depth, self.pretrained_size, mode='nearest')
        # print("rgb: ", rgb.shape)
        rgb = self.semmap_rgb_norm(rgb)

        # rgb = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(rgb)
        # depth = torchvision.transforms.Normalize(mean=[19050],
        #                                          std=[9650])(depth)

        depth_clip = (depth < 1.0).squeeze(1)
        # depth_clip = ((depth < 1.0) & (depth > 0.0)).squeeze(1)
        # print("depth: ", depth.shape)
        depth = self.semmap_depth_norm(depth)
        with torch.no_grad():
            scores = self.rednet(rgb, depth, train)
        # print("scores: ", scores[0][5][100])
        
        if train:
            return scores
        else:
            with torch.no_grad():
                scores = nn.Softmax(dim=1)(scores)
                # print("score: ", scores.shape) #torch.Size([1, 40, 480, 640])
                # print("pred: ", pred.shape) # torch.Size([1, 40, 480, 640])
            pred = (torch.max(scores, 1)[1] + 1).float() # B x 480 x 640

            # print("torch: ", pred[0][100])
            pred[torch.max(scores, 1)[0] < 0.8] = 1

            if self.stabilize: # downsample tiny
                # Mask out out of depth samples
                pred[~depth_clip] = -1 # 41 is UNK in MP3D, but hab-sim renders with -1
                # pred = F.interpolate(pred.unsqueeze(1), (15, 20), mode='nearest').squeeze(1)
            if self.resize:
                pred = F.interpolate(pred.unsqueeze(1), (og_h, og_w), mode='nearest')

            return pred.squeeze(1)

# def load_rednet(device, ckpt="", resize=True, stabilize=False):
#     if not os.path.isfile(ckpt):
#         raise Exception(f"invalid path {ckpt} provided for rednet weights")

#     model = RedNetResizeWrapper(device, resize=resize, stabilize=stabilize).to(device)

#     print("=> loading RedNet checkpoint '{}'".format(ckpt))
#     if device.type == 'cuda':
#         checkpoint = torch.load(ckpt, map_location='cpu')
#     else:
#         checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)

#     state_dict = checkpoint['model_state']
#     prefix = 'module.'
#     state_dict = {
#         (k[len(prefix):] if k[:len(prefix)] == prefix else k): v for k, v in state_dict.items()
#     }
#     state_dict.pop('final_deconv_custom.weight')
#     state_dict.pop('final_deconv_custom.bias')

#     state_dict.pop('out5_conv_custom.weight')
#     state_dict.pop('out5_conv_custom.bias')

#     state_dict.pop('out4_conv_custom.weight')
#     state_dict.pop('out4_conv_custom.bias')

#     state_dict.pop('out3_conv_custom.weight')
#     state_dict.pop('out3_conv_custom.bias')

#     state_dict.pop('out2_conv_custom.weight')
#     state_dict.pop('out2_conv_custom.bias')

#     model.rednet.state_dict().update(state_dict)
#     # model.rednet.load_state_dict(state_dict)
#     print("=> loaded checkpoint '{}' (epoch {})"
#             .format(ckpt, checkpoint['epoch']))
#     # print(model)
#     print("Model's state_dict:")
#     # for param_tensor in model.state_dict():
#     #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
      
#     return model


def load_rednet(device, ckpt="", resize=True, stabilize=False):
    if not os.path.isfile(ckpt):
        raise Exception(f"invalid path {ckpt} provided for rednet weights")

    model = RedNetResizeWrapper(device, resize=resize, stabilize=stabilize).to(device)

    print("=> loading RedNet checkpoint '{}'".format(ckpt))
    if device.type == 'cuda':
        checkpoint = torch.load(ckpt, map_location='cpu')
    else:
        checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)

    # state_dict = checkpoint['state_dict']
    state_dict = checkpoint['model_state']
    prefix = 'module.'
    state_dict = {
        (k[len(prefix):] if k[:len(prefix)] == prefix else k): v for k, v in state_dict.items()
    }
    # model.load_state_dict(state_dict)
    model.rednet.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(ckpt, checkpoint['epoch']))
    # print(model)
    print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
      
    return model

def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train):
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))