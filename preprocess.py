import torchvision.transforms as transforms
import numpy as np
import torch
from utils import extract_info_from_name

"""
For the time being, we use "basic" definitions for the different versions
of the models
However, this should be redefined, so that it is correct for every model.
I guess it should be specific to each version.
"""

class PatternPreprocess(object):
    # only work for VGG16
    def __init__(self, scale_size):
        self.scale = transforms.Compose([
            transforms.Resize(scale_size),
        ])
        self.offset = np.array([103.939, 116.779, 123.68])[:, np.newaxis, np.newaxis]

    def __call__(self, raw_img):
        scaled_img = self.scale(raw_img)
        ret = np.array(scaled_img, dtype=np.float)
        # Channels first.
        ret = ret.transpose(2, 0, 1)
        # To BGR.
        ret = ret[::-1, :, :]
        # Remove pixel-wise mean.
        ret -= self.offset
        ret = np.ascontiguousarray(ret)
        ret = torch.from_numpy(ret).float()

        return ret


def get_preprocess(arch_name, method):
    arch, batch_size, number_epochs = extract_info_from_name(arch_name)

    if arch == 'googlenet':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123. / 255, 117. / 255, 104. / 255],
                                 std=[1. / 255, 1. / 255, 1. / 255])
        ])
    elif arch == 'vgg16':
        if method.find('pattern') != -1:  # pattern_net, pattern_lrp
            transf = transforms.Compose([
                PatternPreprocess((224, 224))
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    elif arch == 'alexnet':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'vgg':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'resnet18':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'resnet34':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'resnet50':
        if method == 'real_time_saliency':
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    elif arch == 'resnet101':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'resnet152':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'squeezenet10':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'squeezenet11':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'densenet121':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'densenet169':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'densenet201':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'densenet161':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    elif arch == 'inceptionv3':
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])

    else:
        print("ERROR: Architecture not found")
    return transf


