from torchvision import models
from torch.autograd import Variable
from torch._thnn import type2backend
import torch

# Model classes
from models.model_classes import SqueezeNet

        
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
        print("Nothing happened")
    elif model_name == 'vgg':
        print("Nothing happened")
    elif model_name == 'resnet':
        print("Nothing happened")
    elif model_name == 'squeezenet':
        model_path = "./models/model_saves/squeezenet_" + str(batch_size) + "_" + str(number_epochs) + ".pth"
        model = SqueezeNet(version=1.0)
        model.load_state_dict(torch.load(model_path))
    elif model_name == 'densenet':
        print("Nothing happened")
    elif model_name == 'inception':
        print("Nothing happened")
    else:
        model = models.__dict__[arch](pretrained=True)
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