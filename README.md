User-friendly Fourier transform package.

== Introduction ==

Numerical libraries like numpy & scipy provide fft/ifft functions.
These are fairly low-level if one is looking for a numerical approximation of the Fourier transform - the user has to take care of:

* correct scaling of the temporal and frequency axes
* correct scaling of the input and output amplitude (for the Parsival and Plancherel theorems to apply)
* for FFT, the frequency and time axes start at zero, which is often not wanted (having zero at the center of the axis, with negative frequencies/times on the left and positive frequencies/times at the right, is more convenient)

Once one has figured this out, this is not difficult at all. Let's go step by step.

The usual discrete Fourier transform is obtained with `scipy.fft.fft(data)`. This function implicitly assumes that time and frequency axis begin with zero.

The discrete Fourier transform for zero-centered frequency & time axis is obtained with `scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(data)))`.

For Parsevals theorem to be true, scaling factors need to be added:

```python
def ft(axis, data): return scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(data))) * (axis[1]-axis[0])

def ift(axis, data): return scipy.fft.fftshift(scipy.fft.ifft(scipy.fft.ifftshift(data))) * (axis[1]-axis[0]) * axis.size
```

The corresponding axis can be generated like this:

```python
n_points, t_spacing = 512, 0.1
t = ( np.arange(n_points, dtype=float)-n_points//2 ) * t_spacing

nu_spacing = 1/t_spacing/n_points
nu = ( np.arange(n_points, dtype=float)-n_points//2 ) * nu_spacing
```

Using these functions, the Plancherel theorem is valid:

```python
data = np.exp(-t**2)
data_transformed = ft(t, data)
integral_timedomain = np.trapz(abs(data)**2, t)
integral_frqdomain = np.trapz(abs(data_transformed)**2, nu)
print(integral_timedomain, "==", integral_frqdomain)
```

Make sure that `n_points` does not contain large prime factors, otherwise performance will be bad. The function `scipy.fft.next_fast_len` helps you with that.

You can just copy these simple `ft` and `ift` functions into your code, or you can use this module, which implements a slightly more general version of the Fourier transform, the so-called offset Fourier transform (see <https://doi.org/10.1364/JOSAA.20.000522>).

== This module ==

Installation:

```
pip3 install git+https://gitlab.com/leberwurscht/fourioso.git
```

Example usage:

```python
import numpy as np
import matplotlib.pyplot as plt
import fourioso

t_span = 50
max_tspacing = 0.1

n_points, t_spacing = fourioso.n_spacing(max_tspacing, t_span) # automatically chooses efficient n_points
t = fourioso.get_axis(n_points, t_spacing)

data = np.exp(-t**2)

nu, data_transformed = fourioso.transform(t, data)
# you can also separate this:
#   nu = fourioso.transform(t)
#   data_transformed = fourioso.transform(t, data, return_axis=False)

data_backtransformed = fourioso.transform(nu, data_transformed, return_axis=False)

plt.figure()
plt.plot(t, data, 'x')
plt.plot(t, data_backtransformed, '+')
plt.show()
```
