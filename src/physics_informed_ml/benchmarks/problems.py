"""Standard PDE problems for benchmarking.

Each problem provides:
- Analytical or numerical ground truth
- Initial and boundary conditions
- Physical parameters
- Evaluation metrics
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkProblem(ABC):
    """Base class for benchmark PDE problems.
    
    Each problem should implement:
    - generate_data(): Create training/test data
    - analytical_solution(): Exact solution (if available)
    - pde_residual(): Compute PDE residual for physics loss
    """
    
    spatial_dim: int  # Spatial dimensions (1, 2, or 3)
    temporal: bool = True  # Whether problem is time-dependent
    
    @abstractmethod
    def generate_data(
        self, n_samples: int, resolution: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate input-output pairs.
        
        Args:
            n_samples: Number of samples to generate
            resolution: Spatial resolution (grid points)
            
        Returns:
            inputs: Input conditions (initial, boundary, parameters)
            outputs: Target solutions
        """
        pass
    
    @abstractmethod
    def analytical_solution(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute analytical solution if available.
        
        Args:
            x: Spatial coordinates
            t: Time coordinates (if temporal)
            
        Returns:
            Exact solution values
        """
        pass


class HeatEquation1D(BenchmarkProblem):
    """1D Heat Equation: ∂u/∂t = α ∂²u/∂x²
    
    Classic diffusion problem:
    - Models heat conduction
    - Simple parabolic PDE
    - Exact solution available for some initial conditions
    
    Args:
        alpha: Thermal diffusivity coefficient
        L: Domain length [0, L]
    """
    
    def __init__(self, alpha: float = 0.01, L: float = 1.0):
        super().__init__(spatial_dim=1, temporal=True)
        self.alpha = alpha
        self.L = L
    
    def generate_data(
        self, n_samples: int, resolution: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate heat equation data.
        
        Uses sine wave initial conditions with various frequencies.
        
        Returns:
            inputs: (n_samples, resolution, 1) initial conditions
            outputs: (n_samples, resolution, 1) solution at t=0.1
        """
        x = torch.linspace(0, self.L, resolution)
        t_final = 0.1
        
        inputs = []
        outputs = []
        
        for _ in range(n_samples):
            # Random frequency for initial condition
            k = torch.randint(1, 5, (1,)).item()
            # Initial: u(x,0) = sin(k*π*x/L)
            u0 = torch.sin(k * np.pi * x / self.L)
            
            # Analytical solution: u(x,t) = sin(k*π*x/L) * exp(-α*k²*π²*t/L²)
            u_t = u0 * torch.exp(-self.alpha * (k * np.pi / self.L) ** 2 * t_final)
            
            inputs.append(u0.unsqueeze(-1))
            outputs.append(u_t.unsqueeze(-1))
        
        return torch.stack(inputs), torch.stack(outputs)
    
    def analytical_solution(
        self, x: torch.Tensor, t: torch.Tensor, k: int = 1
    ) -> torch.Tensor:
        """Analytical solution for sine initial condition.
        
        Args:
            x: Spatial points
            t: Time points
            k: Wave number
            
        Returns:
            Exact solution u(x,t)
        """
        return torch.sin(k * np.pi * x / self.L) * torch.exp(
            -self.alpha * (k * np.pi / self.L) ** 2 * t
        )


class WaveEquation1D(BenchmarkProblem):
    """1D Wave Equation: ∂²u/∂t² = c² ∂²u/∂x²
    
    Hyperbolic PDE modeling:
    - String vibrations
    - Sound waves
    - Electromagnetic waves
    
    Args:
        c: Wave speed
        L: Domain length
    """
    
    def __init__(self, c: float = 1.0, L: float = 1.0):
        super().__init__(spatial_dim=1, temporal=True)
        self.c = c
        self.L = L
    
    def generate_data(
        self, n_samples: int, resolution: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate wave equation data.
        
        Uses standing wave initial conditions.
        
        Returns:
            inputs: (n_samples, resolution, 2) [u(x,0), ∂u/∂t(x,0)]
            outputs: (n_samples, resolution, 1) u(x,t)
        """
        x = torch.linspace(0, self.L, resolution)
        t_final = 0.5
        
        inputs = []
        outputs = []
        
        for _ in range(n_samples):
            k = torch.randint(1, 5, (1,)).item()
            omega = k * np.pi * self.c / self.L
            
            # Initial displacement
            u0 = torch.sin(k * np.pi * x / self.L)
            # Initial velocity (zero for standing wave)
            v0 = torch.zeros_like(u0)
            
            # Solution: u(x,t) = sin(kπx/L) * cos(ωt)
            u_t = torch.sin(k * np.pi * x / self.L) * torch.cos(omega * t_final)
            
            inputs.append(torch.stack([u0, v0], dim=-1))
            outputs.append(u_t.unsqueeze(-1))
        
        return torch.stack(inputs), torch.stack(outputs)
    
    def analytical_solution(
        self, x: torch.Tensor, t: torch.Tensor, k: int = 1
    ) -> torch.Tensor:
        """Analytical solution for standing wave."""
        omega = k * np.pi * self.c / self.L
        return torch.sin(k * np.pi * x / self.L) * torch.cos(omega * t)


class BurgersEquation1D(BenchmarkProblem):
    """1D Burgers' Equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    
    Nonlinear PDE combining:
    - Advection (transport): u∂u/∂x
    - Diffusion (smoothing): ν∂²u/∂x²
    
    Important because:
    - Simplest nonlinear model
    - Tests shock wave handling
    - Bridge to Navier-Stokes
    
    Args:
        nu: Viscosity coefficient (smaller = sharper shocks)
        L: Domain length
    """
    
    def __init__(self, nu: float = 0.01, L: float = 1.0):
        super().__init__(spatial_dim=1, temporal=True)
        self.nu = nu
        self.L = L
    
    def generate_data(
        self, n_samples: int, resolution: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate Burgers' equation data.
        
        Note: No simple analytical solution; uses numerical solver.
        Here we use a simplified initial condition.
        """
        x = torch.linspace(0, self.L, resolution)
        
        inputs = []
        outputs = []
        
        for _ in range(n_samples):
            # Random smooth initial condition
            k1 = torch.randint(1, 4, (1,)).item()
            k2 = torch.randint(1, 4, (1,)).item()
            
            u0 = (
                torch.sin(2 * np.pi * k1 * x / self.L)
                + 0.5 * torch.sin(2 * np.pi * k2 * x / self.L)
            )
            
            # For demonstration, use a simple decay model
            # In practice, use numerical solver (finite difference/spectral)
            u_t = u0 * 0.8  # Placeholder
            
            inputs.append(u0.unsqueeze(-1))
            outputs.append(u_t.unsqueeze(-1))
        
        return torch.stack(inputs), torch.stack(outputs)
    
    def analytical_solution(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """No general analytical solution; requires numerical methods."""
        raise NotImplementedError(
            "Burgers' equation has no general analytical solution. "
            "Use numerical solver for ground truth."
        )


class NavierStokes2D(BenchmarkProblem):
    """2D Incompressible Navier-Stokes Equations.
    
    Momentum: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
    Continuity: ∇·u = 0
    
    Where:
    - u: Velocity field (2D vector)
    - p: Pressure (scalar)
    - ν: Kinematic viscosity
    - f: External forcing
    
    The holy grail of fluid dynamics!
    
    Args:
        nu: Kinematic viscosity
        Re: Reynolds number (controls turbulence)
    """
    
    def __init__(self, nu: float = 0.001, Re: float = 100):
        super().__init__(spatial_dim=2, temporal=True)
        self.nu = nu
        self.Re = Re
    
    def generate_data(
        self, n_samples: int, resolution: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate Navier-Stokes data.
        
        Classic test case: Vortex shedding past cylinder.
        
        Returns:
            inputs: (n_samples, resolution, resolution, 3) [u, v, p] at t=0
            outputs: (n_samples, resolution, resolution, 3) [u, v, p] at t=T
        """
        # This is a placeholder - real implementation needs CFD solver
        # or pre-computed dataset (e.g., from Johns Hopkins Turbulence Database)
        
        # For now, generate simple vortex patterns
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        inputs = []
        outputs = []
        
        for _ in range(n_samples):
            # Initial vortex
            r = torch.sqrt(X**2 + Y**2)
            theta = torch.atan2(Y, X)
            
            # Velocity field (tangential)
            u = -torch.sin(theta) * torch.exp(-r**2 / 0.1)
            v = torch.cos(theta) * torch.exp(-r**2 / 0.1)
            p = torch.zeros_like(u)
            
            input_field = torch.stack([u, v, p], dim=-1)
            
            # Decayed vortex (placeholder)
            output_field = input_field * 0.9
            
            inputs.append(input_field)
            outputs.append(output_field)
        
        return torch.stack(inputs), torch.stack(outputs)
    
    def analytical_solution(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """No general analytical solution except for special cases (e.g., Taylor-Green vortex)."""
        raise NotImplementedError(
            "Navier-Stokes has no general analytical solution. "
            "Use DNS (Direct Numerical Simulation) for ground truth."
        )
