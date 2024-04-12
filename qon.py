# This module contains all the functions to simulate and train 
# the quantum optical neuron, modelled by a spatial light modulator

import numpy as np
from typing import Callable

def sig(x):
  """ Compute the sigmoid activation function, with input x. """
  y = -11*x + 5.5
  return 1/(1 + np.exp(y))

def sigPrime(x):
  """ Compute the sigmoid derivative, with input x. """
  return sig(x)*(1-sig(x))*11

def neuron(weights, bias, Img):
  """ Compute the output of the quantum optical neuron, with parameters 
      weights and bias, and input Img. """
  norm = np.sqrt(np.sum(np.square(weights)))
  f = np.abs(np.sum(np.multiply(Img, weights/norm)))**2
  return sig(f + bias)

def loss(output, target):
  """ Compute the binary cross-entropy between output and target. """
  return - target*np.log(output) - (1-target)*np.log(1-output)

def accuracy(outputs, targets):
  """ Compute the total accuracy of the thresholded outputs 
      against targets. """
  threshold = 0.5
  predicted = np.reshape((outputs >= threshold).astype(int),(-1))
  true_positive = np.sum(targets == predicted)
  return true_positive / len(targets)
          
def spatial_loss_derivative(output, target, weights, bias, Img):
  """ Compute the derivative of the binary cross-entropy with respect to the
      neuron parameters, with spatial-encoded input. """
  # Check
  if output == 1:
    raise ValueError('Output is 1!')
  elif output <= 0:
     raise ValueError('Output is negative!')
  elif 1 - output <= 0:
     raise ValueError('Output is greater than 1!')

  # Declarations
  F = output
  y = target
  norm = np.sqrt(np.sum(np.square(weights)))

  # Compute the derivative with respect to the weights
  g = np.sum(np.multiply(Img, weights/norm)) # <I, U>
  gPrime = (Img - g*weights/norm)/norm # <I, dlambdaU>

  fPrime = 2*np.real(g*np.conjugate(gPrime)) # 2Re[<I, U><I, dU>*]

  crossPrime = (F - y)/(F*(1-F))

  gAbs = np.abs(g) # sqrt(f)
  weights_derivative = crossPrime*sigPrime(gAbs**2 + bias)*fPrime

  # Compute the derivative with respect to the bias
  bias_derivative = crossPrime*sigPrime(gAbs**2 + bias)

  return weights_derivative, bias_derivative

def Fourier_loss_derivative(output, target, weights, bias, Img):
  """ Compute the derivative of the binary cross-entropy with respect to the
      neuron parameters, with Fourier-encoded input. """
  # Check
  if output == 1:
    raise ValueError('Output is 1!')
  elif output <= 0:
     raise ValueError('Output is negative!')
  elif 1 - output <= 0:
     raise ValueError('Output is greater than 1!')

  # Declarations
  F = output
  y = target
  norm = np.sqrt(np.sum(np.square(weights)))

  # Compute the derivative with respect to the weights
  g = np.sum(np.multiply(Img, weights/norm)) # <I, U>
  gAbs = np.abs(g) # sqrt(f)

  gPrime = (Img - gAbs*weights/norm)/norm # Approximation
  fPrime = 2*np.real(gAbs*np.conjugate(gPrime)) # Approximation

  crossPrime = (F - y)/(F*(1-F))

  weights_derivative = crossPrime*sigPrime(gAbs**2 + bias)*fPrime

  # Compute the derivative with respect to the bias
  bias_derivative = crossPrime*sigPrime(gAbs**2 + bias)

  return weights_derivative, bias_derivative

def update_rule(weights, bias, lossWeightsDerivatives, lossBiasDerivatives,\
                lrWeights, lrBias):
  """ Parameters update rule of the gradient descent algorithm. """
  new_weights = weights - lrWeights*np.mean(lossWeightsDerivatives, axis = 0)
  new_bias = bias - lrBias*np.mean(lossBiasDerivatives, axis = 0)
  return new_weights, new_bias

def optimization(loss_derivative: Callable, weights, bias, targets,\
                 test_targets, trainImgs, testImgs, num_epochs,\
                 lrWeights, lrBias):
  """ Gradient descent optimization. """
  # Training set
  outputs = np.array([ neuron(weights, bias, trainImgs[idx,:,:]) \
                         for idx in range(trainImgs.shape[0])] )

  losses = np.array([ loss(outputs[idx], targets[idx]) \
                     for idx in range(outputs.shape[0])])

  # History initialization
  loss_history = [np.mean(losses)]
  accuracy_history = [accuracy(outputs, targets)]

  # Weights initialization
  lossWeightsDerivatives = np.zeros(trainImgs.shape)
  lossBiasDerivatives = np.zeros(trainImgs.shape[0])

  # Compute derivates of the loss function
  for idx in range(trainImgs.shape[0]):
    lossWeightsDerivatives[idx,:,:], lossBiasDerivatives[idx] = \
      loss_derivative(outputs[idx], targets[idx], \
        weights, bias, trainImgs[idx,:,:])

  # Validation set
  test_outputs = np.array([ neuron(weights, bias, testImgs[idx,:,:]) \
                         for idx in range(testImgs.shape[0])] )
  test_losses = np.array([ loss(test_outputs[idx], test_targets[idx]) \
                     for idx in range(test_outputs.shape[0])])

  test_loss_history = [np.mean(test_losses)]
  test_accuracy_history = [accuracy(test_outputs, test_targets)]

  # Verbose
  print('EPOCH', 0)

  print('Loss', loss_history[0], 'Val_Loss', test_loss_history[0] )

  print('Accuracy', accuracy_history[0], 'Val_Acc', test_accuracy_history[0])

  print('---')

  for epoch in range(num_epochs):
    # Update weights
    weights, bias = update_rule(weights, bias, \
                              lossWeightsDerivatives, lossBiasDerivatives, \
                                lrWeights, lrBias)

    ## Training set
    # Update outputs
    outputs = np.array([ neuron(weights, bias, trainImgs[idx,:,:]) \
                         for idx in range(trainImgs.shape[0])] )
    # Update loss
    losses = np.array([ loss(outputs[idx], targets[idx]) \
                        for idx in range(outputs.shape[0])])
    loss_history.append(np.mean(losses))

    # Update accuracy
    accuracy_history.append(accuracy(outputs, targets))

    ## Validation set
    # Update outputs
    test_outputs = np.array([ neuron(weights, bias, testImgs[idx,:,:]) \
                         for idx in range(testImgs.shape[0])] )
    # Update loss
    test_losses = np.array([ loss(test_outputs[idx], test_targets[idx]) \
                        for idx in range(test_outputs.shape[0])])
    test_loss_history.append(np.mean(test_losses))

    # Update accuracy
    test_accuracy_history.append(accuracy(test_outputs, test_targets))

    # Update loss derivate
    for idx in range(trainImgs.shape[0]):
      lossWeightsDerivatives[idx,:,:], lossBiasDerivatives[idx] = \
        loss_derivative(outputs[idx], targets[idx], \
          weights, bias, trainImgs[idx,:,:])

    # Verbose
    print('EPOCH', epoch + 1)

    print('Loss', loss_history[epoch + 1], \
          'Val_Loss', test_loss_history[epoch + 1] )

    print('Accuracy', accuracy_history[epoch + 1], \
          'Val_Acc', test_accuracy_history[epoch + 1])

    print('---')

  return weights, bias, loss_history, test_loss_history, accuracy_history, \
          test_accuracy_history