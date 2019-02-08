from torchvision import models
from torch.autograd import Variable
from torch._thnn import type2backend
import torch
# from torch.nn import nn
from torch import nn

# Model classes
from models.model_classes import *

        
def load_model(arch):
    '''
    Args:
        arch: (string) valid torchvision model name,
            recommendations 'vgg16' | 'googlenet' | 'resnet50'
    '''
    """
    First, we analyze the name to get all the information needed on the model
    """
    model_name, batch_size, number_epochs = extract_info_from_name(arch)

    if model_name == 'googlenet':
        from googlenet import get_googlenet
        model = get_googlenet(pretrain=True)

    elif model_name == 'alexnet':
        model_path = "./models/model_saves/alexnet_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        model = AlexNet()

    elif model_name == 'vgg11':
        model_path = "./models/model_saves/vgg11_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        kwargs['init_weights'] = False
        model = VGG(make_layers(cfg['A']), **kwargs)

    elif model_name == 'vgg11bn':
        model_path = "./models/model_saves/vgg11bn_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        kwargs['init_weights'] = False
        model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    
    elif model_name == 'vgg13':
        model_path = "./models/model_saves/vgg13_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        kwargs['init_weights'] = False
        model = VGG(make_layers(cfg['B']), **kwargs)

    elif model_name == 'vgg13bn':
        model_path = "./models/model_saves/vgg13bn_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        kwargs['init_weights'] = False
        model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    
    elif model_name == 'vgg16':
        model_path = "./models/model_saves/vgg16_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        kwargs['init_weights'] = False
        model = VGG(make_layers(cfg['D']), **kwargs)
    
    elif model_name == 'vgg16bn':
        model_path = "./models/model_saves/vgg16bn_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        kwargs['init_weights'] = False
        model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    
    elif model_name == 'vgg19':
        model_path = "./models/model_saves/vgg19_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        kwargs['init_weights'] = False
        model = VGG(make_layers(cfg['E']), **kwargs)
    
    elif model_name == 'vgg19bn':
        model_path = "./models/model_saves/vgg19bn_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        kwargs['init_weights'] = False
        model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)

    elif model_name == 'resnet18':
        model_path = "./models/model_saves/resnet18_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        model = ResNet(BasicBlock, [2, 2, 2, 2])

    elif model_name == 'resnet34':
        model_path = "./models/model_saves/resnet34_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        model = ResNet(BasicBlock, [3, 4, 6, 3])
    
    elif model_name == 'resnet50':
        model_path = "./models/model_saves/resnet50_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        model = ResNet(BasicBlock, [3, 4, 6, 3])
    
    elif model_name == 'resnet101':
        model_path = "./models/model_saves/resnet101_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        model = ResNet(BasicBlock, [3, 4, 23, 3])

    elif model_name == 'resnet152':
        model_path = "./models/model_saves/resnet152_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        model = ResNet(BasicBlock, [3, 8, 36, 3])

    elif model_name == 'squeezenet10':
        model_path = "./models/model_saves/squeezenet10_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        model = SqueezeNet(version=1.0)

    elif model_name == 'squeezenet11':
        model_path = "./models/model_saves/squeezenet11_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        model = SqueezeNet(version=1.1)

    elif model_name == 'densenet121':
        model_path = "./models/model_saves/densenet121_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
        )
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/densenet121-a639ec97.pth')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    elif model_name == 'densenet169':
        model_path = "./models/model_saves/densenet169_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
        )
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/densenet169-b2777c0a.pth')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        
    elif model_name == 'densenet201':
        model_path = "./models/model_saves/densenet201_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
        )
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/densenet201-c1103571.pth')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    elif model_name == 'densenet161':
        model_path = "./models/model_saves/densenet161_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
        )
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/densenet161-8d451a50.pth')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    elif model_name == 'inceptionv3':
        model_path = "./models/model_saves/inceptionv3_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        model = Inception3()
        
    else:
        model = models.__dict__[arch](pretrained=True)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def cuda_var(tensor, requires_grad=False):
    return Variable(tensor.cuda(), requires_grad=requires_grad)


def upsample(inp, size):
    '''
    Args:
        inp: (Tensor) input
        size: (Tuple [int, int]) height x width
    '''
    backend = type2backend[inp.type()]
    f = getattr(backend, 'SpatialUpSamplingBilinear_updateOutput')
    upsample_inp = inp.new()
    # The last argument stands for "align_corners" (if True, the corner pixels of the input 
    # and output tensors are aligned, and thus preserving the values at those pixels). This 
    # only has effect when :attr:`mode` is `linear`, `bilinear` or `trilinear` (default: False)
    f(backend.library_state, inp, upsample_inp, size[0], size[1], False)
    return upsample_inp


def extract_info_from_name(arch_name):
    # Architecture is named using the following code: 
    # saving_model_name = model_name + "_" + str(batch_size) + "_" + str(num_epochs) + ".pth"
    list_names = arch_name.split("_")
    if(len(list_names) < 3):
        return arch_name
    else:
        model_name = list_names[0]
        batch_size = list_names[1]
        number_epochs = list_names[2]
        return model_name, batch_size, number_epochs