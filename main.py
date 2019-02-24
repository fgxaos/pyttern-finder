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
# Already implemented: ['alexnet_8_15', 'resnet18_8_15', 'squeezenet10_8_15']
model_name = 'squeezenet10_8_15'

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
number_image = 5
# Change 'image_class' to 0 if you want to display for a dog
if(displayed_class == "dog"):
    image_class = 0
elif(displayed_class == "cat"):
    image_class = 1
else:
    print("ERROR: wrong displayed class")

# Take the sample image, and display it (original form)
image_path = "models/test_" + displayed_class + "_images/" + displayed_class + str(number_image) + ".jpg"

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

if not os.path.exists('images/' + model_name + '/'):
    os.makedirs('images/' + model_name + '/')
save_destination = 'images/' + model_name + '/' + displayed_class + str(number_image) + '_saliency.png'

plt.savefig(save_destination)
plt.show()