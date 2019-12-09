# machine_learning

I use this depository to collect all of my crazy ideas for machine learning. (neural networks)

### Dependencies
All programs will be written with the following dependencies:

- [Python 3](https://www.python.org/download/releases/3.0/)
- [{ickle](https://docs.python.org/3/library/pickle.html)
- [NumPy](https://numpy.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

(all avaialable with [pip](https://pip.pypa.io/en/stable/reference/pip_install/))



## Knowledge Distillation

One of the first things I am exploring is knowledge distillation (KD). The basic idea of knowledge distillation is to take a very accurate, but large and computationally expensive model and transfer the knowledge learned from the expensive model to a much smaller model which is easier to deploy on smaller machines such as cell phones (or even smaller like hearing aids). Many papers have been written on this subject, but the heart of the manner is explained and investigaed very well in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). 


### Extracting Knowledge

In most current approaches (like that in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)) the objective is to take the softmax (probability like answers) of the expensive model and use this to help train a smaller model. It is believed (and backed up experimentally [Self-supervised Knowledge Distillation Using Singular Value Decomposition](https://link.springer.com/chapter/10.1007/978-3-030-01231-1_21) and others) that training a model to learning the relationships between identifiers will produce some of the best models.
