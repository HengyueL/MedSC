import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import sys




__all__ = ['ResNeXt', 'resnext50', 'resnext101', 'resnext152']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size=112,
                 sample_duration=16,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400,
                 truncate_level=4):
        self.inplanes = 64
        self.truncate_level = truncate_level
        self.num_classes = num_classes
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        #self.conv1 = nn.Conv3d(
        #    3,
        #    64,
        #    kernel_size=(3,7,7),
        #    stride=(1, 2, 2),
        #    padding=(1, 3, 3),
        #    bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        if self.truncate_level == 1:
            self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        elif self.truncate_level == 2:
            self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
            self.layer2 = self._make_layer(
                block, 256, layers[1], shortcut_type, cardinality, stride=2)
        elif self.truncate_level == 3:
            self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
            self.layer2 = self._make_layer(
                block, 256, layers[1], shortcut_type, cardinality, stride=2)
            self.layer3 = self._make_layer(
                block, 512, layers[2], shortcut_type, cardinality, stride=2)
        else:
            self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                        cardinality)
            self.layer2 = self._make_layer(
                block, 256, layers[1], shortcut_type, cardinality, stride=2)
            self.layer3 = self._make_layer(
                block, 512, layers[2], shortcut_type, cardinality, stride=2)
            self.layer4 = self._make_layer(
                block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        #last_duration = 1
        last_size = int(math.ceil(sample_size / 32))
        # self.avgpool = nn.AvgPool3d(
        #     (last_duration, last_size, last_size), stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        output_planes_lst = [256, 512, 1024, 2048]
        # self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        self.fc = nn.Linear(output_planes_lst[self.truncate_level-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.expand(-1, 3 // x.size(1), -1, -1, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.truncate_level == 1:
            x = self.layer1(x)
        elif self.truncate_level == 2:
            x = self.layer1(x)
            x = self.layer2(x)
        elif self.truncate_level == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `PENet(**model_args)`.
        """
        model_args = {
            'num_classes': self.num_classes
        }

        return model_args

    def load_pretrained(self, ckpt_path, gpu_ids):
        """Load parameters from a pre-trained PENetClassifier from checkpoint at ckpt_path.
        Args:
            ckpt_path: Path to checkpoint for PENetClassifier.
        Adapted from:
            https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        """
        device = 'cuda:{}'.format(gpu_ids[0]) if len(gpu_ids) > 0 else 'cpu'
        pretrained_dict = torch.load(ckpt_path, map_location=device)['model_state']
        model_dict = self.state_dict()

        # Filter out unnecessary keys
        pretrained_dict = {k[len('module.'):]: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict.pop("fc.weight")
        pretrained_dict.pop("fc.bias")

        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)

    def fine_tuning_parameters(self, fine_tuning_boundary, fine_tuning_lr=0.0):
        """Get parameters for fine-tuning the model.
        Args:
            fine_tuning_boundary: Name of first layer after the fine-tuning layers.
            fine_tuning_lr: Learning rate to apply to fine-tuning layers (all layers before `boundary_layer`).
        Returns:
            List of dicts that can be passed to an optimizer.
        """

        def gen_params(boundary_layer_name, fine_tuning):
            """Generate parameters, if fine_tuning generate the params before boundary_layer_name.
            If unfrozen, generate the params at boundary_layer_name and beyond."""
            saw_boundary_layer = False
            for name, param in self.named_parameters():
                if name.startswith(boundary_layer_name):
                    saw_boundary_layer = True

                if saw_boundary_layer and fine_tuning:
                    return
                elif not saw_boundary_layer and not fine_tuning:
                    continue
                else:
                    yield param

        # Fine-tune the network's layers from encoder.2 onwards
        optimizer_parameters = [{'params': gen_params(fine_tuning_boundary, fine_tuning=True), 'lr': fine_tuning_lr},
                                {'params': gen_params(fine_tuning_boundary, fine_tuning=False)}]

        return optimizer_parameters

class ResNeXt_Layer(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size=112,
                 sample_duration=16,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400,
                 truncate_level=4,
                 trunc_layer=21,
                 pretrained=True):
        super(ResNeXt_Layer, self).__init__()
        self.full_model = ResNeXt(
            block=block,
            layers=layers,
            sample_size=sample_size,
            sample_duration=sample_duration,
            shortcut_type=shortcut_type,
            cardinality=cardinality,
            num_classes=num_classes,
            truncate_level=truncate_level
        )
        self.num_classes = num_classes
        if pretrained:
            self.full_model.load_pretrained("../penet/kinetics_resnext_101_RGB_16_best.pth", [0,1])
        self.full_model, _ = truncate_resnext(self.full_model)
        self.features = nn.Sequential(*list(self.full_model.children())[:trunc_layer])
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(1024, num_classes)
        self.full_model=None
    
    def forward(self,x):
        x = x.expand(-1, 3 // x.size(1), -1, -1, -1)

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `PENet(**model_args)`.
        """
        model_args = {
            'num_classes': self.num_classes
        }

        return model_args
    
    def load_pretrained(self, ckpt_path, gpu_ids):
        # """Load parameters from a pre-trained PENetClassifier from checkpoint at ckpt_path.
        # Args:
        #     ckpt_path: Path to checkpoint for PENetClassifier.
        # Adapted from:
        #     https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        # """
        # device = 'cuda:{}'.format(gpu_ids[0]) if len(gpu_ids) > 0 else 'cpu'
        # pretrained_dict = torch.load(ckpt_path, map_location=device)['model_state']
        # model_dict = self.state_dict()

        # # Filter out unnecessary keys
        # pretrained_dict = {k[len('module.'):]: v for k, v in pretrained_dict.items()}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # pretrained_dict.pop("fc.weight")
        # pretrained_dict.pop("fc.bias")

        # # Overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # self.load_state_dict(model_dict, strict=False)
        pass

    def fine_tuning_parameters(self, fine_tuning_boundary, fine_tuning_lr=0.0):
        """Get parameters for fine-tuning the model.
        Args:
            fine_tuning_boundary: Name of first layer after the fine-tuning layers.
            fine_tuning_lr: Learning rate to apply to fine-tuning layers (all layers before `boundary_layer`).
        Returns:
            List of dicts that can be passed to an optimizer.
        """

        def gen_params(boundary_layer_name, fine_tuning):
            """Generate parameters, if fine_tuning generate the params before boundary_layer_name.
            If unfrozen, generate the params at boundary_layer_name and beyond."""
            saw_boundary_layer = False
            for name, param in self.named_parameters():
                if name.startswith(boundary_layer_name):
                    saw_boundary_layer = True

                if saw_boundary_layer and fine_tuning:
                    return
                elif not saw_boundary_layer and not fine_tuning:
                    continue
                else:
                    yield param

        # Fine-tune the network's layers from encoder.2 onwards
        optimizer_parameters = [{'params': gen_params(fine_tuning_boundary, fine_tuning=True), 'lr': fine_tuning_lr},
                                {'params': gen_params(fine_tuning_boundary, fine_tuning=False)}]


        return optimizer_parameters

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def resnext50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnext101_layer(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt_Layer(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model

def unsqueeze_net(net, return_net=False, sequential_break=True):
    ret_list = []
    for x in net:
        if ((isinstance(x, nn.Conv3d) or isinstance(x, nn.BatchNorm3d) \
            or isinstance(x, nn.MaxPool3d) or isinstance(x, nn.AdaptiveAvgPool3d) \
            or isinstance(x, nn.ReLU) or isinstance(x, nn.Linear)
            or isinstance(x, ResNeXtBottleneck))) \
            or ((not sequential_break) and isinstance(x, nn.Sequential)):
            ret_list.append(x)
        else:
            ret_list.extend(list(x.children()))
    if return_net:
        ret_list = nn.Sequential(*ret_list)
    return ret_list

def find_next_trunc(net, idx):
    l = list(net.children())
    for i, layer in enumerate(l[idx:]):
        if isinstance(layer, nn.Conv3d) or isinstance(layer, ResNeXtBottleneck):
            idx += i
            break
    return nn.Sequential(*list(net.children())[:idx]), idx

def truncate_resnext(net: nn.Module, imagenet=False):
    net = [net]
    net = unsqueeze_net(net, sequential_break=True)
    net = unsqueeze_net(net, sequential_break=True)
    pre_count = 0
    while len(net) != pre_count:
        pre_count = len(net)
        print(len(net))
        net = unsqueeze_net(net)
    net = nn.Sequential(*net)
    p = 1
    idx = []
    while p <= len(net):
        _, next_p = find_next_trunc(net, p)
        p = next_p + 1
        idx.append(next_p-1)
    return net, idx

if __name__ == '__main__':
    model = resnext101(num_classes=1, truncate_level=1)
    
    # model.load_pretrained("../kinetics_resnext_101_RGB_16_best.pth", [0])