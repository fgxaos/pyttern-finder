from create_explainer import get_explainer
from preprocess import get_preprocess
import utils
import viz
import torch
import time
import os
import pylab
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
params = {
    'font.family': 'sans-serif',
    'axes.titlesize': 25,
    'axes.titlepad': 10,
}
pylab.rcParams.update(params)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## Add more model_methods if it works
# Already implemented: ['squeezenet10_8_15', 'resnet18_8_15']
model_name = 'resnet18_8_15'

model_methods = [
    [model_name, 'vanilla_grad', 'imshow'],
    [model_name, 'grad_x_input', 'imshow'], 
    [model_name, 'saliency', 'imshow'],
    [model_name, 'integrate_grad', 'imshow'],
    [model_name, 'deconv', 'imshow'],
    [model_name, 'guided_backprop', 'imshow'],
    #[model_name, 'gradcam', 'camshow'],
    #[model_name, 'excitation_backprop', 'camshow'],
    #[model_name, 'contrastive_excitation_backprop', 'camshow']
]
# Change 'displayed_class' to "dog" if you want to display for a dog
displayed_class = "dog"
# Change 'image_class' to 0 if you want to display for a dog
image_class = 1

# Take the sample image, and display it (original form)
if displayed_class == "cat1":
    image_path= 'models/cat1.jpg'
elif displayed_class == "cat2":
    image_path = 'models/cat2.jpg'    
elif displayed_class == "dog":
    image_path = 'models/Natural-Dog-Law-2-To-dogs,-energy-is-everything.jpg'
else:
    print("Bad value for 'displayed_class', please check it")

raw_img = viz.pil_loader(image_path)
plt.figure(figsize=(5,5))
plt.imshow(raw_img)
plt.axis('off')
plt.title(displayed_class)

# Now, we want to display the saliency maps of this image, for every model_method element
all_saliency_maps = []

for model_name, method_name, _ in model_methods:
    # Get a specific picture transformation (see torchvision.transforms documentation) 
    transf = get_preprocess(model_name, method_name)
    # Load the pretrained model
    model = utils.load_model(model_name)
    model.cuda()
    # Get the explainer
    explainer = get_explainer(model, method_name)

    # Transform the image
    inp = transf(raw_img)
    if method_name == 'googlenet':      # swap channel due to caffe weights
        inp_copy = inp.clone()
        inp[0] = inp_copy[2]
        inp[2] = inp_copy[0]
    inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)

    target = torch.LongTensor([image_class]).cuda()
    saliency = explainer.explain(inp, target)
    saliency = utils.upsample(saliency, (raw_img.height, raw_img.width))
    #all_saliency_maps.append(saliency.cpy().numpy())
    all_saliency_maps.append(saliency.cpu().numpy())

# Display all the results
plt.figure(figsize=(25, 15))
plt.subplot(3, 5, 1)
plt.imshow(raw_img)
plt.axis('off')
plt.title(displayed_class)
for i, (saliency, (model_name, method_name, show_style)) in enumerate(zip(all_saliency_maps, model_methods)):
    plt.subplot(3, 5, i + 2 + i // 4)
    if show_style == 'camshow':
        viz.plot_cam(np.abs(saliency).max(axis=1).squeeze(), raw_img, 'jet', alpha=0.5)
    else:
        if model_name == 'googlenet' or method_name == 'pattern_net':
            saliency = saliency.squeeze()[::-1].transpose(1, 2, 0)
        else:
            saliency = saliency.squeeze().transpose(1, 2, 0)
        saliency -= saliency.min()
        saliency /= (saliency.max() + 1e-20)
        plt.imshow(saliency, cmap='gray')

    plt.axis('off')
    if method_name == 'excitation_backprop':
        plt.title('Exc_bp')
    elif method_name == 'contrastive_excitation_backprop':
        plt.title('CExc_bp')
    else:
        plt.title('%s' % (method_name))

plt.tight_layout()
save_destination = 'images/' + model_name + '-' + displayed_class + '_saliency.png'
plt.savefig(save_destination)
plt.show()