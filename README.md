# PyTorch Visual Attribution Methods on Cats vs. Dogs
*Inspired by https://github.com/yulongwang12/visual-attribution*

A collection of visual attribution methods for model interpretability, in the case of the Cats vs. Dogs classification problem.

## Motivation
Deep Learning can currently handle image classification pretty well, but the problem is trust. As it is explained in this [paper](https://arxiv.org/pdf/1602.04938.pdf) and implemented [here](https://github.com/marcotcr/lime).

However, the existing models are not satisfying enough, as they have still trouble identifying the correct patterns. As it is shown in the previous article, a model might classify pictures according to unusual patterns (for example, a species of dog could be identified by the color of its collar, which is a bad identifier).

With this project, I intend to implement a method which would give a result similar to lime's, but with a different process to be even more accurate.

## Workflow



## General To Do List: 
This project implemented GoogleNet using an old version of PyTorch; this doesn't work anymore
Thus, I need to use another neural network. I will use model finetuning in order to do that with the elephant.  

1 - Pre-train different models for the Cat vs Dog classification, using Kaggle data (pictures)
2 - Once the models are pre-trained, add them to the method_models (main.py)
3 - Test with different explainers


## Detailed To Do List
### User Interface
* For now, 'model_methods' is hard coded and every saliency map is displayed. However, this was practical for the former user (only a few models), but that doesn't apply anymore to my program. What should be done is the following: when _main.py_ is launched, a 'model_methods' is created. It is only an empty list. Then, the user will be able to choose the model architecture first, then the explainer and at the end, _camshow_ or _imshow_ (depending on the chosen model, so I think that this should actually be chosen automatically).
* This could be implemented in a CLI, or even a GUI if you have enough time (ticking boxes to choose).
* Anyway, I'll have to adapt the displayed pictures and create a display grid for the images, that depend on the chosen parameters

### Models
* For now, only the _squeezenet_ model has been implemented. Add _resnet_, _alexnet_, _vgg_ (maybe already implemented, check it), _densenet_, _inception_
* Once all these pre-trained models are implemented, create your own to find an even better solution
