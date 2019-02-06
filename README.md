### Fatal Brain Damage
Code for the paper "Fatal Brain Damage" by El Mahdi El Mhamdi, Rachid Guerraoui, Sergei Volodin, <a href="https://arxiv.org/abs/1902.01686">arxiv:1902.01686</a>.

Swiss Federal Institute of Technology in Lausanne (EPFL), Distributed Computing Laboratory, 2018-2019.

Correspondence to sergei.volodin@epfl.ch

### Installation
To run a Jupyter notebook:
1. Install miniconda https://conda.io/miniconda.html for Python 3.6.6
2. Install requirements from requirements.txt
3. Activate Freeze nbextension to skip the computation cells automatically and just plot data from pickled results

All results are pickled so that the figures can be generated without running the computation again

Tested on a 12CPU machine with 2xGPU NVIDIA GTX 1080 running Ubuntu 16.04.5 LTS

### Code description
Code is written in Python 3 with Keras/TensorFlow and is documented.

#### Classes
1. `model.py` the definition of a fully-connected network model with crashes using Keras
2. `experiment.py` is the main class `Experiment` which computes error experimentally or theoretically
3. `bounds.py` implements bounds b1, b2, b3, b4 and its methods are added to the `Experiment` class.
4. `experiment_random.py` provides random initialization for an `Experiment`, `experiment_train.py` instead trains a network with data. `experiment_datasets.py` runs a `TrainExperiment` for specific datasets.
5. `process_data.py` implements functions to plot the experimental results
6. `helpers.py` various small functions used in the project

#### Notebooks
1. `BostonTh7Test.ipynb` tests Proposition 7 from the main paper
2. `ComparisonIncreasingDropoutMNIST.ipynb` compares networks trained with different dropout on MNIST using bounds
3. `ErrorComparisonBoston.ipynb` compares Boston-trained networks (see main paper)
4. `ErrorAdditivityRandom.ipynb` is the test of error additivity on Boston dataset
5. `ErrorOnTraining.ipynb` tests the prediction of Corollary 4 in the main paper
6. `Regularization.ipynb` regularizes networks with b3 variance bound to achieve fault tolerance

##### Additional or failed experiments
1. `ConvNetTest-MNIST.ipynb`, `ConvNetTest-VGG16.ipynb` test the b3 bound on larger networks
2. `BostonTh2QuadraticTest.ipynb` tries to test Corollary 4 from the main paper, see supplementary/Failed experiments
3. `ErrorComparisonRandom.ipynb` compares random networks (see supplementary)
4. `ComparisonIncreasingDropout.ipynb` compares networks trained with increasing dropout on Boston dataset (failed experiment)

#### Unused
`bad_input_search.py` is a genetic algorithm for searching for worst fault tolerance input. `tests.py` provides internal tests for the functions. `tfshow.py` shows a TensorFlow graph in a Jupyter Notebook.
