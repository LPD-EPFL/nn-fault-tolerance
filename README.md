### Probabilistic Fault Tolerance of Neural Networks in the Continuous Limit
Code for the paper "Probabilistic Fault Tolerance of Neural Networks in the Continuous Limit"

### Installation
To run a Jupyter notebook:
1. Install miniconda https://conda.io/miniconda.html for Python 3.7.3
2. Install requirements from environment.yml
3. Activate Freeze nbextension to skip the computation cells automatically and just plot data from pickled results
4. You can turn on cells which re-run the experiments, but do not enable _configuration_ cells which will just screw the initial conditions

Most results are pickled so that the figures can be generated without running the computation again

Tested on a 12CPU machine with 2xGPU NVIDIA GTX 1080 running Ubuntu 16.04.5 LTS

### Code description
Code is written in Python 3 with Keras/TensorFlow and is documented.

#### Classes
1. `model.py` the definition of a fully-connected network model with crashes using Keras
2. `experiment.py` is the main class `Experiment` which computes error experimentally or theoretically
3. `bounds.py` implements bounds b1, b2, b3, b4 (see supplementary) and its methods are added to the `Experiment` class.
4. `experiment_random.py` provides random initialization for an `Experiment`, `experiment_train.py` instead trains a network with data. `experiment_datasets.py` runs a `TrainExperiment` for specific datasets.
5. `process_data.py` implements functions to plot the experimental results
6. `helpers.py` various small functions used in the project
7. `derivative_decay.py` implements routines for the experiments of derivative decay rate test
8. `continuity.py` implements the smooth() functions from Eq. (1) from the paper
9. `model_conv.py` contains various helpers for convolutional models (replacing activations with smooth ones, ...)
10. `experiment_model.py` wraps around a Sequential Keras model and adds faults to all layers

#### Notebooks for the main paper
1. `ComparisonIncreasingDropoutMNIST.ipynb` compares networks trained with different dropout on MNIST using bounds
2. `Regularization.ipynb` regularizes networks with b3 variance bound to achieve fault tolerance
3. `ConvNetTest-MNIST.ipynb` trains a small convolutional network and verifies bound on it
4. `ConvNetTest-VGG16.ipynb` loads the pre-trained VGG model and verifies the bound on it
5. `ConvNetTest-ft.ipynb` compares fault tolerance of pre-trained CNN models
6. `FaultTolerance-Continuity-FC-MNIST.ipynb` shows a decay of VarDelta when n increases when using our regularizer (Eq. 1 main paper)
7. `TheAlgorithm.ipynb` tests Algorithm 1 on a small convnet for MNIST


### Notebooks from the supplementary
1. `FilterPlayground.ipynb` allows to tune the smooth() coefficients online
2. `WeightDecay-FC-MNIST.ipynb` shows that weights do not decay as we expect without regularization
3. `DerivativeDecay-FC-MNIST.ipynb` shows that derivatives do not decay without regularization as we expect
4. `WeightDecay-Continuity-FC-MNIST.ipynb` shows that with regularization, continuity holds (derivatives decay, weights stabilize)
5. `ConvNetTest-VGG16-ManyImages.ipynb` investigates into filter size and how well b3 works in CNNs and tries to apply pooling on input for VGG (uncomment a line to download images first)
6. `ErrorAdditivityRandom.ipynb` is the test of error additivity on Boston dataset
7. `ErrorComparisonBoston.ipynb` compares Boston-trained networks
8. `ConvNetTest-MNIST.ipynb`, `ConvNetTest-VGG16.ipynb` test the b3 bound on larger networks
9. `ErrorOnTraining.ipynb` tests the prediction of AP9 in the supplementary

##### Additional
`Riemann.ipynb` generates the continuity figure

#### Unused
`bad_input_search.py` is a genetic algorithm for searching for worst fault tolerance input. `tests.py` provides internal tests for the functions. `tfshow.py` shows a TensorFlow graph in a Jupyter Notebook. `onepixel` and `AE_*.ipynb` investigates into adversarial robustness of networks regularized with b3.
