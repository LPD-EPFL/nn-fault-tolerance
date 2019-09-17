from model import *
import tensorflow as tf
from vis.utils.utils import apply_modifications
from keras.activations import softplus
from keras.utils import CustomObjectScope
from keras.layers import Activation, InputLayer, Flatten
from keras.layers.pooling import AveragePooling2D
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
from keras import Model, Input
from tqdm import tqdm

def softplus10(x, scaler = 1.0):
    """ Softplus with variable softness """
    return softplus(x * scaler) / scaler

# Identity activation function
def identity(x): return x

# custom activation functions
custom_fcn = {'identity': identity, 'softplus10': softplus10}

def remove_activation(model, layer):
  """ For a given model, remove an activation """
  print("Removing activation from layer %s" % str(layer))

  # obtaining the layer object
  layer = model.layers[layer]

  # replacing ReLU with SoftPlus10
  assert hasattr(layer, 'activation')
  layer.activation = identity

  # applying modifications
  print('Applying modifications...')
  with CustomObjectScope(custom_fcn):
    model = apply_modifications(model)

  # returning the resulting model
  return model

def replace_relu_with_softplus(model, scaler = 1.0):
  """ For a given keras model, replace all ReLU activation functions with softplus """
  print("Replacing ReLU to Softplus(%.2f)" % scaler)

  # replacing ReLU with SoftPlus10
  replaced = []
  activations = set()
  for i, layer in enumerate(model.layers):
    if hasattr(layer, 'activation'):
      if layer.activation.__name__ == 'relu':
        activations.add(str(layer.activation.__name__))
        replaced.append(i)
        layer.activation = softplus10

  print("Replaced activations %s on layers %s with softplus" % (str(activations), str(replaced)))
  # applying modifications
  print('Applying modifications...')
  with CustomObjectScope(custom_fcn):
    model = apply_modifications(model)

  # returning the resulting model
  return model

def cut_and_flatten(model, layer):
  """ Cut a submodel from the model up to layer 'layer', sum over inputs, full model is returned is layer = -1 """
  if layer == -1: return model
  return Model(inputs = model.inputs, outputs = Dense(1, kernel_initializer = 'ones', activation = 'linear')(Flatten()(model.layers[layer].output)))

def poolinput(model, pooling = 5):
  """ Make input grayscale for the net """
  # new (small) input
  inp_shape = model.input.shape[1:]
  d = inp_shape[0]
  input_tensor = Input(shape = inp_shape)

  x = input_tensor
  x = AveragePooling2D(padding = 'same', pool_size = pooling, input_shape = inp_shape)(x)
  x = Lambda(partial(tf.image.resize_images, size = (d, d)))(x)

  model_gs = Model(inputs = input_tensor, outputs = model(x))
  return model_gs

def grayscale(model):
  """ Make input grayscale for the net """
  # new (small) input
  d = int(model.inputs[0].shape[1])
  input_tensor = Input(shape = (d, d))

  # new model taking small images and upscaling them
  def grayscale_to_rgb(x):
      """ NWH -> NWHC with 3 channels """
      x = tf.expand_dims(x, 3)
      x = tf.tile(x, multiples = [1,1,1,3])
      return x
  out = model(Lambda(grayscale_to_rgb)(input_tensor))
  model_gs = Model(inputs = input_tensor, outputs = out)
  return model_gs

def upscale_from(model, d):
  """ Upscale image and then feed to the model as a new model """
  # new (small) input
  input_tensor = Input(shape = (d, d, 3))

  # new model taking small images and upscaling them
  n = int(model.inputs[0].shape[1])
  out = model(Lambda(lambda x : tf.image.resize_images(x, (n, n)))(input_tensor))
  model_upscale = Model(inputs = input_tensor, outputs = out)

def split_x_rc0(x, rc0 = 10):
    """ Cut first rc0 components from red channel column 0, return [cut components, rest] with rest having zeros at these places """
    # copying the cat
    x_without_first = np.copy(x)

    # taking the red channel in the first column
    x_taken = np.copy(x[:, :rc0, 0, 0])

    # removing red channel in the first column
    x_without_first[:, :rc0, 0, 0] = 0
    return x_taken, x_without_first

def merge_with_taken(model, x_without_first, rc0 = 10):
  """ Create a new model taking rc0-shaped inputs which are then merged with x_without_first from @see split_x_rc0 """
  # input with size just d
  input_tensor = InputLayer(input_shape = (rc0,))

  # side of the original image
  n = int(model.inputs[0].shape[1])

  # creating a new model
  model_upscale = Sequential()
  model_upscale.add(input_tensor)

  # adding all x but first red column
  model_upscale.add(Lambda(lambda y : tf.pad(tf.reshape(y, shape = (-1, rc0, 1, 1)),
                                           paddings = ([0, 0], [0, n - rc0], [0, n - 1], [0, 2])) + x_without_first))

  # adding the rest of VGG15
  model_upscale.add(model)

  # sanity check: when fed x_taken, will reconstruct the input correctly
  #assert np.all(K.function([model_upscale.input], [model_upscale.layers[0].output])([x_taken])[0] == x_orig), "Input is handled in a wrong way"

  return model_upscale

def pool_max2avg(model):
    """ Replacing maxpool with avgpool """

    x = model.layers[0].output

    replaced = []
    for i, layer in enumerate(model.layers[1:]):
        #print(dir(layer))
        if isinstance(layer, keras.layers.pooling.MaxPooling2D):
            config = layer.get_config()
            config['name'] += "newpool"
            L = keras.layers.pooling.AveragePooling2D(**config)
            replaced.append(i)
        else:
            L = layer
        #model1.add(L)
        x = L(x)
    model = Model(inputs = model.input, outputs = x)
    print("Replaced maxpool to avgpool at layers %s" % str(replaced))
    return model

def load_image(img_path, dimension = 224, axis = plt, color = False):
  """ Load an image """
  img = image.load_img(img_path, target_size=(dimension, dimension))
  x = image.img_to_array(img)
  x = 255.0 * x / np.max(x)
  if not color:
    x = x.mean(axis = 2)
  axis.imshow(x / 255., cmap = 'gray')
  x = np.expand_dims(x, axis=0)
  #x = preprocess_input(x)
  return x

def load_cat(dimension = 224):
  # getting picture of a cat
  return load_image('cat.jpg', dimension, color = True)

def SliceLayer(to_keep):
  """ Keep only these components in the layer """
  return Lambda(lambda x : tf.gather(x, to_keep, axis = 1))

def keep_oindices(model, out_to_keep):
  """ A model with only these output indices kept """
  return Model(inputs = model.input, outputs = SliceLayer(out_to_keep)(model.output))

def compute_error_stack(exp, x, K, k):
  """ Compute error in experiment exp on input x using K repetitions and k repetitions of repetitions """
  datas = [exp.compute_error(np.array(x), repetitions = K) for _ in tqdm(range(k))]
  return np.hstack(datas)

def experiment_mean_std(exp, x, repetitions):
  """ Compute experimental mean/std for input x using number of repetitions """
  result = {}
  r = np.array(exp.compute_error(x, repetitions), dtype = np.float64)
  result['mean'] = np.mean(r, axis = 1)
  result['std']  = np.std (r, axis = 1)
  return result

def predict_kept(model, x, to_keep = None):
    """ Predict knowing that the model output was modified with to_keep """
    preds = model.predict(x)
    if to_keep is not None:
        result = np.zeros(1000)
        for i, key in enumerate(to_keep):
            result[key] = preds[0][i]
        result = result.reshape(-1, 1000)
    else:
        result = preds
    return result
