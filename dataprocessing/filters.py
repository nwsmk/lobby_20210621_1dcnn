import numpy as np


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def cir_filter(x, fc, window_size):
    num_samples = x.shape[0]
    y = np.zeros(x.shape, dtype=complex)
    for i in range(num_samples):
        ht = np.fft.ifft(np.fft.ifftshift(x[i, :])) * np.exp(-2 * 1j * np.pi * fc)
        ht[window_size:] = 0
        y[i, :] = np.fft.fftshift(np.fft.fft(ht))
    return y