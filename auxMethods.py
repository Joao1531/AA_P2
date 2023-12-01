import numpy as np


def cMSE(y_hat, y, c):
  err = y-y_hat
  err = (1-c)*err**2 + c*np.maximum(0,err)**2
  return np.sum(err)/err.shape[0]
