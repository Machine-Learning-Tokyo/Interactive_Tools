# Interactive Tools for machine learning, deep learning, and math

## Content

## Deep Learning

- [Transformer Explainer](#transformer-explainer)
- [exBERT](#exbert)
- [BertViz](#bertviz)
- [CNN Explainer](#cnn-explainer)
- [Play with GANs in the Browser](#play-with-gans-in-the-browser)
- [ConvNet Playground](#convnet-playground)
- [Distill: Exploring Neural Networks with Activation Atlases](#distill-exploring-neural-networks-with-activation-atlases)
- [A visual introduction to Machine Learning](#a-visual-introduction-to-machine-learning)
- [Interactive Deep Learning Playground](#interactive-deep-learning-playground)
- [Initializing neural networks](#initializing-neural-networks)
- [Embedding Projector](#embedding-projector)
- [OpenAI Microscope](#openai-microscope)

## Data

- [Atlas Data Exploration](#atlas-data-exploration)
  
## Interpretability

- [The Language Interpretability Tool](#the-language-interpretability-tool)
- [What if](#what-if)
- [Measuring diversity](#measuring-diversity)

## Math
- [Sage Interactions](#sage-interactions)
- [Probability Distributions](#probability-distributions)
- [Bayesian Inference](#bayesian-inference)
- [Seeing Theory: Probability and Stats](#seeing-theory-probability-and-stats)
- [Interactive Gaussian Process Visualization](#interactive-gaussian-process-visualization)

---

# Deep Learning

## Transformer Explainer

Transformer Explainer is an interactive visualization tool designed to help anyone learn how Transformer-based models like GPT work. It runs a live GPT-2 model right in your browser, allowing you to experiment with your own text and observe in real time how internal components and operations of the Transformer work together to predict the next tokens.

- [Source: GitHub](https://github.com/poloclub/transformer-explainer)
- [Source: Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

[<img width="1426" alt="Transformer Explainer" src="https://github.com/user-attachments/assets/9221d106-da89-4ef4-87fe-e142e3d9ebb6">](https://poloclub.github.io/transformer-explainer/)

## exBERT

"exBERT is a tool to help humans conduct flexible, interactive investigations and formulate hypotheses for the model-internal reasoning process, supporting analysis for a wide variety of Hugging Face Transformer models. exBERT provides insights into the meaning of the contextual representations and attention by matching a human-specified input to similar contexts in large annotated datasets."

- Source: [exBERT](https://huggingface.co/exbert/)

<img width="1438" alt="exbert" src="https://user-images.githubusercontent.com/27798583/157429282-352f48f2-0d7b-43ef-8aff-79feaa8ae47a.png">


## BertViz

"BertViz is a tool for visualizing attention in the Transformer model, supporting most models from the transformers library (BERT, GPT-2, XLNet, RoBERTa, XLM, CTRL, MarianMT, etc.). It extends the [Tensor2Tensor visualization tool](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/visualization) by Llion Jones and the [transformers library](https://github.com/huggingface/transformers) from [HuggingFace](https://github.com/huggingface)."

- Source: [BertViz](https://github.com/jessevig/bertviz)

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/bert_vis.gif" width="600"></p>](https://github.com/jessevig/bertviz)

## CNN Explainer

An interactive visualization system designed to help non-experts learn about Convolutional Neural Networks (CNNs). It runs a pre-tained CNN in the browser and lets you explore the layers and operations.

- [Live Demo](https://poloclub.github.io/cnn-explainer/) | [Video](https://youtube.com/watch?v=udVN7fPvGe0) | [Code](https://github.com/poloclub/cnn-explainer) | [Paper](https://arxiv.org/abs/2004.15004)

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/cnn_explainer.gif" width="400"></p>](https://poloclub.github.io/cnn-explainer/)

## Play with GANs in the Browser
Explore Generative Adversarial Networks directly in the browser with GAN Lab. There are many cool features that support interactive experimentation.

- Interactive hyperparameter adjustment
- User-defined data distribution
- Slow-motion mode
- Manual step-by-step execution

- [Source: GAN Lab](https://poloclub.github.io/ganlab/)

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/gans.png" width="1000"></p>](https://poloclub.github.io/ganlab/)

# ConvNet Playground
ConvNet Playground is an interactive visualization tool for exploring Convolutional Neural Networks applied to the task of semantic image search. 

- [Source: ConvNet Playground](https://convnetplayground.fastforwardlabs.com)

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/convnetplaygroud.png" width="1000"></p>](https://convnetplayground.fastforwardlabs.com/#/)


## Distill: Exploring Neural Networks with Activation Atlases

Feature inversion to visualize millions of activations from an image classification network leads to an explorable activation atlas of features the network has learned. This can reveal how the network typically represents some concepts.

- [Source: Distill](https://distill.pub/2019/activation-atlas/)

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/activation_atlas.png" width="1000"></p>](https://distill.pub/2019/activation-atlas/)


## A visual introduction to Machine Learning
Available in many different languages.

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/intro_ML.png" width="1000"></p>](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)


## Interactive Deep Learning Playground
New to Deep Learning? Tinker with a Neural Network in your browser.

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/dl_playground.png" width="1000"></p>](https://playground.tensorflow.org)

## Initializing neural networks

Initialization can have a significant impact on convergence in training deep neural networks. Simple initialization schemes can accelerate training, but they require care to avoid common pitfalls. In this post, deeplearning.ai folks explain how to initialize neural network parameters effectively.

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/weight_init.png" width="1000"></p>](https://www.deeplearning.ai/ai-notes/initialization/)


## Embedding Projector

It's increaingly important to understand how data is being interpreted by machine learning models. To translate the things we understand naturally (e.g. words, sounds, or videos) to a form that the algorithms can process, we often use embeddings, a mathematical vector representation that captures different facets (dimensions) of the data. In this interactive, you can explore multiple different algorithms (PCA, t-SNE, UMAP) for exploring these embeddings in your browser.

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/embedding-mnist.gif" width="1000"></p>](https://projector.tensorflow.org/)

## OpenAI Microscope

The OpenAI Microscope is a collection of visualizations of every significant layer and neuron of eight important vision models.

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/openai_microscope.png" width="1000"></p>](https://microscope.openai.com/)

# Data

## Atlas Data Exploration

Atlas allows you to explore real, up-to-date data from sources like social media, news, and academic journals curated by the Nomic team.

- [Source: Nomic](https://atlas.nomic.ai/discover)

[<img width="1436" alt="Atlas" src="https://github.com/user-attachments/assets/9409355e-7930-4fb1-9fe1-236be2549db6">](https://atlas.nomic.ai/discover)


# Interpretability, Fairness

## The Language Interpretability Tool

The Language Interpretability Tool (LIT) is an open-source platform for visualization and understanding of NLP models.

You can use LIT to ask and answer questions like:

- What kind of examples does my model perform poorly on?
- Why did my model make this prediction? Can it attribute it to adversarial behavior, or undesirable priors from the training set?
- Does my model behave consistently if I change things like textual style, verb tense, or pronoun gender?

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/Interactive_Tools/blob/master/images/lit.gif" width="1000"></p>](https://pair-code.github.io/lit/)

## What if

The What-If Tool lets you visually probe the behavior of trained machine learning models, with minimal coding.

- [Source: PAIR](https://pair-code.github.io/what-if-tool/)

![what-if](https://user-images.githubusercontent.com/27798583/118443855-b3cc9b80-b6ec-11eb-9c28-849d7e755cd4.gif)

## Measuring diversity

PAIR Explorables around measuring diversity.

"Search, ranking and recommendation systems can help find useful documents in large datasets. However, these datasets reflect the biases of the society in which they were created and the systems risk re-entrenching those biases. For example, if someone who is not a white man searches for “CEO pictures” and sees a page of white men, they may feel that only white men can be CEOs, further perpetuating lack of representation at companies’ executive levels."

- Mitchell et. al. (2020) [Diversity and Inclusion Metrics in Subset Selection](https://arxiv.org/abs/2002.03256)
- [Interactive explorables](https://pair.withgoogle.com/explorables/measuring-diversity/)

- [Source: PAIR](https://pair-code.github.io/lit/)


[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/mdiv.png" width="1000"></p>](https://pair.withgoogle.com/explorables/measuring-diversity/)

# Math

## Sage Interactions

This is a collection of pages demonstrating the use of the **interact** command in Sage. It should be easy to just scroll through and copy/paste examples into Sage notebooks. 

Examples include Algebra, Bioinformatics, Calculus, Cryptography, Differential Equations, Drawing Graphics, Dynamical Systems, Fractals, Games and Diversions, Geometry, Graph Theory, Linear Algebra, Loop Quantum Gravity, Number Theory, Statistics/Probability, Topology, Web Applications.

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/sage.png" width="1000"></p>](https://wiki.sagemath.org/interact/)


## Probability Distributions

by Simon Ward-Jones. A visual 👀 tour of probability distributions.

- Bernoulli Distribution
- Binomial Distribution
- Normal Distribution
- Beta Distribution
- LogNormal Distribution

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/prob.png" width="1000"></p>](https://www.simonwardjones.co.uk/posts/probability_distributions/)

## Bayesian Inference

by Simon Ward-Jones. Explaining the basics of bayesian inference with the example of flipping a coin.

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/bayes.png" width="1000"></p>](https://www.simonwardjones.co.uk/posts/probability_distributions/)


## Seeing Theory: Probability and Stats

A visual introduction to probability and statistics.

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/seeing_theory.png" width="1000"></p>](https://seeing-theory.brown.edu)

## Interactive Gaussian Process Visualization

"A Gaussian process can be thought of as an extension of the multivariate normal distribution to an infinite number of random variables covering each point on the input domain. The covariance between function values at any two points is given by the evaluation of the kernel of the Gaussian process. For an in-depth explanation, read this excellent [distill.pub article](https://distill.pub/2019/visual-exploration-gaussian-processes/) and then come back to this interactive visualisation!"

- Source: [Infinite curiosity](http://www.infinitecuriosity.org/vizgp/)

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/DL_study_group/blob/master/images/gaussian_vis.png" width="1000"></p>](http://www.infinitecuriosity.org/vizgp/)

