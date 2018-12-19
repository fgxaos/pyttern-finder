# PyTorch Visual Attribution Methods on Cats vs. Dogs
*Inspired by https://github.com/yulongwang12/visual-attribution*

A collection of visual attribution methods for model interpretability, in the case of the Cats vs. Dogs classification problem.

## Motivation
Deep Learning can currently handle image classification pretty well, but the problem is trust. As it is explained in this [paper](https://arxiv.org/pdf/1602.04938.pdf) and implemented [here](https://github.com/marcotcr/lime).

However, the existing models are not satisfying enough, as they have still trouble identifying the correct patterns. As it is shown in the previous article, a model might classify pictures according to unusual patterns (for example, a species of dog could be identified by the color of its collar, which is a bad identifier).

With this project, I intend to implement a method which would give a result similar to lime's, but with a different process to be even more accurate.

## Workflow



### To Do: 
This project implemented GoogleNet using an old version of PyTorch; this doesn't work anymore
Thus, I need to use another neural network. I will use model finetuning in order to do that with the elephant.  

1 - Pre-train different models for the Cat vs Dog classification, using Kaggle data (pictures)
2 - Once the models are pre-trained, add them to the method_models (main.py)
3 - Test with different explainers