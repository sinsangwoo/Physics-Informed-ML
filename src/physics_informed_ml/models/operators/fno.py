"""Fourier Neural Operator (FNO) implementation.

FNO learns operators between function spaces using spectral methods.
Key features:
- Resolution-invariant: can generalize to different grid resolutions
- Global receptive field via Fourier transforms
- Efficient O(N log N) complexity for convolutions

References:
    Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations"
    ICLR 2021. https://arxiv.org/abs/2010.08895
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution layer using FFT.
    
    Performs convolution in Fourier space by:
    1. FFT: Transform input to frequency domain
    2. Multiply by learnable weights (for selected modes)
    3. IFFT: Transform back to spatial domain
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes: Number of Fourier modes to use (low-frequency modes)
               Higher modes = more detail but more parameters
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Learnable weights for Fourier modes
        # Complex-valued weights in frequency domain
        # Scale initialization following Glorot/Xavier principle
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spectral convolution.
        
        Args:
            x: Input tensor of shape (batch, channels, spatial_dim)
            
        Returns:
            Output tensor of shape (batch, out_channels, spatial_dim)
        """
        batch_size = x.shape[0]
        
        # Compute FFT along spatial dimension
        # rfft = Real FFT (exploits conjugate symmetry for real inputs)
        x_ft = torch.fft.rfft(x)

        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-1) // 2 + 1,  # rfft output size
            dtype=torch.cfloat,
            device=x.device,
        )

        # Multiply relevant Fourier modes
        # Only process the first 'modes' low-frequency components
        # High-frequency modes are implicitly filtered out (truncation)
        out_ft[:, :, : self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, : self.modes], self.weights
        )

        # Inverse FFT to get back spatial domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution layer.
    
    Extends SpectralConv1d to 2D problems (e.g., fluid flow on a plane).
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes1: Number of Fourier modes in first spatial dimension
        modes2: Number of Fourier modes in second spatial dimension
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.rand(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input of shape (batch, channels, height, width)
            
        Returns:
            Output of shape (batch, out_channels, height, width)
        """
        batch_size = x.shape[0]
        
        # 2D FFT
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Multiply Fourier modes in both dimensions
        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights,
        )

        # Inverse 2D FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution layer.
    
    For 3D problems like volumetric flows or 3D elasticity.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes1, modes2, modes3: Fourier modes for each spatial dimension
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.rand(
                in_channels,
                out_channels,
                modes1,
                modes2,
                modes3,
                dtype=torch.cfloat,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input of shape (batch, channels, depth, height, width)
            
        Returns:
            Output of shape (batch, out_channels, depth, height, width)
        """
        batch_size = x.shape[0]
        
        # 3D FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Multiply Fourier modes
        out_ft[
            :, :, : self.modes1, : self.modes2, : self.modes3
        ] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3],
            self.weights,
        )

        # Inverse 3D FFT
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO1d(nn.Module):
    """1D Fourier Neural Operator.
    
    Architecture:
    1. Lifting layer: Project input to higher dimension
    2. Fourier layers: Stack of (spectral conv + nonlinearity)
    3. Projection layer: Project back to output dimension
    
    Args:
        modes: Number of Fourier modes to use
        width: Hidden channel dimension (model capacity)
        in_channels: Input channels (e.g., 1 for scalar field)
        out_channels: Output channels (e.g., 1 for prediction)
        n_layers: Number of Fourier layers (depth)
    """

    def __init__(
        self,
        modes: int,
        width: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        # Lifting layer: Increase dimensionality
        # Why? Richer representation for learning complex functions
        self.fc0 = nn.Linear(in_channels, width)

        # Stack of Fourier layers
        self.conv_layers = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        
        # Residual connections (like ResNet)
        # Why? Helps gradient flow and allows identity mapping
        self.w_layers = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)]
        )

        # Projection layers: Reduce back to output dimension
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input of shape (batch, spatial_dim, in_channels)
            
        Returns:
            Output of shape (batch, spatial_dim, out_channels)
        """
        # Lift to higher dimension
        x = self.fc0(x)  # (batch, spatial, width)
        x = x.permute(0, 2, 1)  # (batch, width, spatial) for Conv1d

        # Fourier layers with residual connections
        for conv, w in zip(self.conv_layers, self.w_layers):
            # Spectral path
            x1 = conv(x)
            # Skip connection path  
            x2 = w(x)
            # Combine and activate
            x = F.gelu(x1 + x2)

        # Project to output dimension
        x = x.permute(0, 2, 1)  # (batch, spatial, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class FNO2d(nn.Module):
    """2D Fourier Neural Operator.
    
    For 2D problems like:
    - Navier-Stokes equations (fluid flow)
    - Heat diffusion on surfaces
    - Wave propagation in 2D
    
    Args:
        modes1, modes2: Fourier modes for each dimension
        width: Hidden channel dimension
        in_channels: Input channels
        out_channels: Output channels
        n_layers: Number of Fourier layers
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers

        self.fc0 = nn.Linear(in_channels, width)

        self.conv_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)]
        )
        self.w_layers = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)]
        )

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input of shape (batch, height, width, in_channels)
            
        Returns:
            Output of shape (batch, height, width, out_channels)
        """
        # Lift
        x = self.fc0(x)  # (batch, h, w, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, h, w)

        # Fourier layers
        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)

        # Project
        x = x.permute(0, 2, 3, 1)  # (batch, h, w, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class FNO3d(nn.Module):
    """3D Fourier Neural Operator.
    
    For 3D volumetric problems like:
    - 3D fluid dynamics
    - Structural mechanics
    - Electromagnetic fields
    
    Args:
        modes1, modes2, modes3: Fourier modes for each dimension
        width: Hidden channel dimension
        in_channels: Input channels
        out_channels: Output channels
        n_layers: Number of Fourier layers
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        modes3: int,
        width: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.n_layers = n_layers

        self.fc0 = nn.Linear(in_channels, width)

        self.conv_layers = nn.ModuleList(
            [
                SpectralConv3d(width, width, modes1, modes2, modes3)
                for _ in range(n_layers)
            ]
        )
        self.w_layers = nn.ModuleList(
            [nn.Conv3d(width, width, 1) for _ in range(n_layers)]
        )

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input of shape (batch, depth, height, width, in_channels)
            
        Returns:
            Output of shape (batch, depth, height, width, out_channels)
        """
        # Lift
        x = self.fc0(x)  # (batch, d, h, w, width)
        x = x.permute(0, 4, 1, 2, 3)  # (batch, width, d, h, w)

        # Fourier layers
        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)

        # Project
        x = x.permute(0, 2, 3, 4, 1)  # (batch, d, h, w, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
