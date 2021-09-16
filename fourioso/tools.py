import numpy as np
import scipy.interpolate

def piecewise_cossqr(x, xs, ys):
  xs, ys = np.array(xs), np.array(ys)
  y = np.zeros_like(x, (ys.flat[:1]+1.0).dtype)

  for xstart, xend, ystart, yend in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
    piece = (xstart<=x) & (x<=xend)
    xscaled = np.ones_like(x[piece]) if xstart==-np.inf else x[piece]/(xend-xstart) - xstart/(xend-xstart)
    y[piece] = (ystart-yend) * np.cos(xscaled*np.pi/2)**2 + yend

  y[x<xs[0]] = ys[0]
  y[x>xs[-1]] = ys[-1]

  return y

def get_fwhm(x, y):
  spline = scipy.interpolate.UnivariateSpline(x, y-np.max(y)/2, s=0)

  try:
    r1, r2 = spline.roots()
    return r2-r1
  except ValueError:
    return np.nan
