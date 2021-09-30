import functools

import numpy
import scipy.fft

def axis_shape(n, axis, ndim):
  axis %= ndim
  return tuple(n if i==axis else 1 for i in range(ndim))

def n_spacing(max_spacing, span):
  n = scipy.fft.next_fast_len(int(numpy.ceil(span/max_spacing)))
  spacing = span/n
  return n, spacing

def get_axis(n, spacing, axis_nr=0, ndim=1, axis_center=0, np=numpy):
  axis = np.arange(n, dtype=float)-n//2
  axis *= spacing
  axis += axis_center
  axis = axis.reshape(axis_shape(n,axis_nr,ndim))
  return axis

def linear_phase_oft(spacing, n, data=None, axis=-1, axis_center=0, additional_linear_phase=0, overwrite=False, multiply_phase=False, inplace_fft=functools.partial(scipy.fft.fft,overwrite_x=True), np=numpy):
  axis_center_out = additional_linear_phase/2/np.pi
  additional_linear_phase_out = -2*np.pi*axis_center

  if data is not None:
    if not overwrite or not np.iscomplexobj(data): data = np.array(data, dtype=(data.flat[0]+np.array(1,dtype=np.complex64)).dtype)

    shape = axis_shape(n, axis, data.ndim)

    b1 = np.exp(-1j*2*np.pi * ((n+1)//2)/n)
    b2 = np.exp(1j*2*np.pi * (n//2)/n)
    factor = np.exp(1j*2*np.pi * ((n//2)/2+1)/n * (n%2)) * spacing * np.exp(-1j*2*np.pi*axis_center_out*axis_center)
    if multiply_phase:
      factor *= np.exp(-1j*additional_linear_phase_out/spacing/n*(1+n//2))
      b2 *= np.exp(1j*additional_linear_phase_out/spacing/n)
    b1,b2,factor = b1.astype(data.dtype),b2.astype(data.dtype),factor.astype(data.dtype)

    data *= np.cumprod(np.broadcast_to(b1,shape),axis=axis)
    data = inplace_fft(data,None,axis)
    data *= np.cumprod(np.broadcast_to(b2,shape),axis=axis) * factor

  return 1/spacing/n, axis_center_out, data, additional_linear_phase_out

def linear_phase_ioft(spacing, n, data=None, axis=-1, axis_center=0, additional_linear_phase=0, overwrite=False, multiply_phase=False, inplace_ifft=functools.partial(scipy.fft.ifft,overwrite_x=True), np=numpy):

  axis_center_out = -additional_linear_phase/2/np.pi
  additional_linear_phase_out = 2*np.pi*axis_center

  if data is not None:
    if not overwrite or not np.iscomplexobj(data): data = np.array(data, dtype=(data.flat[0]+np.array(1,dtype=np.complex64)).dtype)

    shape = axis_shape(n, axis, data.ndim)

    b1 = np.exp(1j*2*np.pi * ((n+1)//2)/n)
    b2 = np.exp(-1j*2*np.pi * (n//2)/n)
    factor = np.exp(-1j*2*np.pi * ((n//2)/2+1)/n * (n%2)) * spacing*data.shape[axis] * np.exp(1j*2*np.pi*axis_center_out*axis_center)
    if multiply_phase:
      factor *= np.exp(-1j*additional_linear_phase_out/spacing/n*(1+n//2))
      b2 *= np.exp(1j*additional_linear_phase_out/spacing/n)
    b1,b2,factor = b1.astype(data.dtype),b2.astype(data.dtype),factor.astype(data.dtype)

    data *= np.cumprod(np.broadcast_to(b1,shape),axis=axis)
    data = inplace_ifft(data,None,axis)
    data *= np.cumprod(np.broadcast_to(b2,shape),axis=axis) * factor

  return 1/spacing/n, axis_center_out, data, additional_linear_phase_out

def linear_phase_odoubleft(spacing, data=None, axis=-1, axis_center=0, additional_linear_phase=0, overwrite=False, multiply_phase=False, np=numpy):
  if data.shape[axis]%2==1:
    if data is not None: data_out = np.flip(data,axis) if overwrite else np.array(np.flip(data,axis))
    else: data = None
  else:
    if data is not None: data_out = np.flip(np.roll(data,-1,axis), axis)
    else: data = None

  spacing_out, axis_center_out, additional_linear_phase_out = spacing, -axis_center, -additional_linear_phase

  if data is not None:
    if multiply_phase:
      n = data.shape[axis]
      b = np.exp(1j*additional_linear_phase_out/spacing_out/n)
      factor = b**(1+n//2)
      b,factor = b.astype(data.dtype),factor.astype(data.dtype)
      data_out *= np.cumprod(np.broadcast_to(b,axis_shape(n, axis, data.ndim)),axis=axis) * factor
  else:
    data_out = data

  return spacing, -axis_center, data_out, -additional_linear_phase

def linear_phase_oidentity(spacing, data=None, axis=-1, axis_center=0, additional_linear_phase=0, overwrite=False, multiply_phase=False, np=numpy):
  if data is not None:
    data_out = data if overwrite else np.array(data)

    if multiply_phase:
      n = data.shape[axis]
      b = np.exp(1j*additional_linear_phase/spacing/n)
      factor = b**(1+n//2)
      b,factor = b.astype(data.dtype),factor.astype(data.dtype)
      data_out *= np.cumprod(np.broadcast_to(b,axis_shape(n, axis, data.ndim)),axis=axis) * factor
  else:
    data_out = data

  return spacing, axis_center, data_out, additional_linear_phase

def linear_phase_multioft(spacing, n, data, order=1, axis=-1, axis_center=0, additional_linear_phase=0, multiply_phase=False, overwrite=False, inplace_fft=functools.partial(scipy.fft.fft,overwrite_x=True), inplace_ifft=functools.partial(scipy.fft.ifft,overwrite_x=True), np=numpy):
  order = order % 4
  if order==0: return linear_phase_oidentity(spacing, data, axis=axis, axis_center=axis_center, additional_linear_phase=additional_linear_phase, multiply_phase=multiply_phase, overwrite=overwrite, np=np)
  elif order==1: return linear_phase_oft(spacing, n, data, axis=axis, axis_center=axis_center, additional_linear_phase=additional_linear_phase, multiply_phase=multiply_phase, overwrite=overwrite, inplace_fft=inplace_fft, np=np)
  elif order==2: return linear_phase_odoubleft(spacing, data, axis=axis, axis_center=axis_center, additional_linear_phase=additional_linear_phase, multiply_phase=multiply_phase, overwrite=overwrite, np=np)
  elif order==3: return linear_phase_ioft(spacing, n, data, axis=axis, axis_center=axis_center, additional_linear_phase=additional_linear_phase, multiply_phase=multiply_phase, overwrite=overwrite, inplace_ifft=inplace_ifft, np=np)

def transform(axis, data=None, order=1, phase_coeff=None, return_phase_coeff=False, overwrite=False,
    inplace_fft=functools.partial(scipy.fft.fft,overwrite_x=True),
    inplace_ifft=functools.partial(scipy.fft.ifft,overwrite_x=True),
    np=numpy
  ):
  # TO DO: sign argument

  axis_nr = numpy.argmax(axis.shape)
  n = axis.size
  axis = axis.squeeze()
  axis_center_i = n//2
  axis_center = axis[axis_center_i]
  spacing = axis[1] - axis[0]

  if phase_coeff is None:
    additional_linear_phase = 0
  else:
    additional_linear_phase = phase_coeff

  if not type(order)==int: raise ValueError("order must be an integer")

  spacing_out, axis_center_out, data_out, additional_linear_phase_out = linear_phase_multioft(spacing, n, data, order=order, axis=axis_nr, axis_center=axis_center, additional_linear_phase=additional_linear_phase, overwrite=overwrite, inplace_fft=inplace_fft, inplace_ifft=inplace_ifft, np=np, multiply_phase=not return_phase_coeff)
  axis_out = get_axis(n, spacing_out, axis_nr, data.ndim if data is not None else 1, axis_center_out, np=np)

  if return_phase_coeff:
    if data is not None:
      return axis_out, data_out, additional_linear_phase
    else:
      return axis_out, additional_linear_phase
  else:
    if data is not None:
      return axis_out, data_out
    else:
      return axis_out

def itransform(axis, data=None, order=1, phase_coeff=None, return_phase_coeff=False, overwrite=False,
    inplace_fft=functools.partial(scipy.fft.fft,overwrite_x=True),
    inplace_ifft=functools.partial(scipy.fft.ifft,overwrite_x=True),
    np=numpy):
  return transform(axis, data, -order, phase_coeff, return_phase_coeff, overwrite, inplace_fft, inplace_ifft, np)
