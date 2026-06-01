"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

def fft2(data, normalized=True):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = data.contiguous()
    data_complex = torch.view_as_complex(data)
    data_fft = torch.fft.fft2(data_complex, norm='ortho' if normalized else None)
    data_fft_output = torch.view_as_real(data_fft)
    return data_fft_output

def ifft2(data, normalized=True):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = torch.view_as_complex(data)
    data = torch.fft.ifft2(data)
    data = torch.view_as_real(data)
    return data

def r2c(x):
    re, im = torch.chunk(x, 2, 1)
    x = torch.complex(re, im)
    return x

def c2r(x):
    x = torch.cat([torch.real(x), torch.imag(x)], 1)
    return x
