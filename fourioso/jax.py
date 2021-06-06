import functools
import jax

from . import functions

axis_shape = functions.axis_shape

n_spacing = functions.n_spacing

get_axis = functools.partial(functions.get_axis, np=jax.numpy)

transform = functools.partial(functions.transform, inplace_fft=jax.numpy.fft.fft, inplace_ifft=jax.numpy.fft.ifft, np=jax.numpy)

itransform = functools.partial(functions.itransform, inplace_fft=jax.numpy.fft.fft, inplace_ifft=jax.numpy.fft.ifft, np=jax.numpy)
