from torchvision import models
from torch.autograd import Variable
from torch._thnn import type2backend

        
def load_model(arch):
    '''
    Args:
        arch: (string) valid torchvision model name,
            recommendations 'vgg16' | 'googlenet' | 'resnet50'
    '''
    if arch == 'googlenet':
        from googlenet import get_googlenet
        model = get_googlenet(pretrain=True)
    elif arch == 'alexnet':

    elif arch == 'vgg':
    
    elif arch == 'resnet':
    
    elif arch == 'squeezenet':

    elif arch == 'densenet':

    elif arch == 'inception':

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
    f(backend.library_state, inp, upsample_inp, size[0], size[1])
    return upsample_inp


def extract_info_from_name(arch_name):
    # Architecture is named using the following code: 
    # saving_model_name = model_name + "_" + str(batch_size) + "_" + str(num_epochs) + ".pth"
    list_names = arch_name.split("_")
    model_name = list_name[0]
    batch_size = list_name[1]
    num_epochs = list_name[2][:-4]
    return model_name, batch_size, num_epochs