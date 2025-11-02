#!/usr/bin/env python3
"""
DDM Analyzer - GPU-Accelerated Edition
Differential Dynamic Microscopy with GPU acceleration via PyTorch

Features:
- Multi-frequency data merging
- Configurable averaging (maxNCouples)
- Raw .npy data export
- Comprehensive CSV/plot outputs
- GPU acceleration (Intel XPU, NVIDIA CUDA, or CPU fallback)
- Automatic device detection
- Command-line interface

Author: Enhanced from notebook implementation with GPU acceleration
"""

import sys
import io
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import json
import time

# Force UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np

# PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not found. Install with: pip install torch")
    print("⚠ Falling back to CPU-only mode")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev
from scipy.ndimage import zoom
import cv2
import pandas as pd
import re
from PIL import Image


def get_compute_device(use_gpu=True, verbose=True):
    """
    Automatically detect and return the best available compute device
    
    Priority order:
    1. NVIDIA GPU (CUDA) - fastest for most operations
    2. Intel GPU (XPU) - good performance, integrated
    3. CPU - universal fallback
    
    Parameters:
    -----------
    use_gpu : bool
        If False, force CPU usage
    verbose : bool
        Print device information
    
    Returns:
    --------
    torch.device or None if torch not available
    """
    if not TORCH_AVAILABLE:
        if verbose:
            print("ℹ PyTorch not available, using NumPy CPU mode")
        return None
    
    if not use_gpu:
        if verbose:
            print("ℹ GPU disabled by user, using CPU")
        return torch.device('cpu')
    
    # Try NVIDIA GPU first (CUDA)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ Using NVIDIA GPU: {gpu_name} ({memory_gb:.1f} GB)")
        return device
    
    # Try Intel GPU (XPU)
    try:
        if torch.xpu.is_available():
            device = torch.device('xpu')
            if verbose:
                try:
                    gpu_name = torch.xpu.get_device_name(0)
                    print(f"✓ Using Intel GPU: {gpu_name}")
                except:
                    print(f"✓ Using Intel GPU (XPU)")
            return device
    except AttributeError:
        # Intel extension not installed
        pass
    
    # Fall back to CPU
    if verbose:
        print("ℹ No GPU detected, using CPU")
        print("  For GPU support:")
        print("  - NVIDIA: pip install torch (with CUDA)")
        print("  - Intel: pip install intel-extension-for-pytorch")
    return torch.device('cpu')


@dataclass
class DDMParameters:
    """Store DDM experimental parameters"""
    pixel_size_um: float
    magnification: float
    frame_rate: float
    n_frames: int
    roi_size: int = 512
    
    @property
    def q_min(self):
        """Minimum wave vector (um^-1)"""
        return 2 * np.pi / (self.roi_size * self.pixel_size_um / self.magnification)
    
    @property
    def q_max(self):
        """Maximum wave vector (um^-1)"""
        return np.pi * self.magnification / self.pixel_size_um


def natural_sort_key(s):
    """Sort strings with numbers naturally"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


def log_spaced(L, points_per_decade=15):
    """
    Generate logarithmically spaced integers smaller than L
    Same as notebook's logSpaced function
    """
    if L < 2:
        return np.array([1])
    nbdecades = np.log10(L)
    return np.unique(np.logspace(
        start=0, stop=nbdecades,
        num=int(nbdecades * points_per_decade),
        base=10, endpoint=False
    ).astype(int))


class RadialAverager:
    """
    Azimuthal averaging in Fourier space
    Exactly as implemented in the notebook
    """
    def __init__(self, shape):
        """shape is the image shape (height, width)"""
        self.shape = shape
        h, w = shape
        
        # Create coordinate arrays centered at DC component
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        
        # Distance from center (in pixels)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Bins for radial averaging
        self.rmax = int(min(cx, cy))
        self.bins = np.arange(0, self.rmax + 1)
        self.hd = self.bins[:-1] + 0.5  # bin centers
        
        # Digitize: assign each pixel to a radial bin
        self.indices = np.digitize(r.ravel(), self.bins)
        
    def __call__(self, data):
        """Perform radial averaging on 2D data"""
        # Shift FFT to center DC component
        data_shifted = np.fft.fftshift(data)
        data_flat = data_shifted.ravel()
        
        # Average within each radial bin
        radial_profile = np.zeros(len(self.hd))
        for i in range(len(self.hd)):
            mask = self.indices == (i + 1)
            if np.any(mask):
                radial_profile[i] = np.mean(data_flat[mask])
        
        return radial_profile


class DDMAnalyzer:
    """
    Complete DDM Analyzer with GPU acceleration
    """
    
    def __init__(self, params: DDMParameters, use_gpu=True):
        self.params = params
        self.images = None
        self.lag_times = None
        self.structure_function = None
        self.q_values = None
        self.diff_images = None
        
        # GPU acceleration
        self.device = get_compute_device(use_gpu=use_gpu, verbose=True)
        self.use_gpu = (self.device is not None and 
                       str(self.device) != 'cpu' and 
                       TORCH_AVAILABLE)
        
        # Analysis results
        self.decay_rates = None
        self.amplitudes = None
        self.backgrounds = None
        self.q_fit = None
        self.diffusion_coeff = None
        self.diffusion_std = None
        
    def load_tif_directory(self, directory_path: str, roi=None, max_frames=None):
        """Load sequential TIF files from directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        tif_files = (list(directory.glob("*.tif")) + list(directory.glob("*.tiff")) +
                    list(directory.glob("*.TIF")) + list(directory.glob("*.TIFF")))
        
        if not tif_files:
            raise ValueError(f"No TIF files found in {directory_path}")
        
        tif_files.sort(key=natural_sort_key)
        
        print(f"Found {len(tif_files)} TIF files")
        print(f"First: {tif_files[0].name}, Last: {tif_files[-1].name}")
        
        if max_frames is not None and max_frames < len(tif_files):
            tif_files = tif_files[:max_frames]
            print(f"Loading first {max_frames} frames")
        
        frames = []
        for i, tif_file in enumerate(tif_files):
            img = Image.open(tif_file)
            frame = np.array(img).astype(np.float64)
            
            if len(frame.shape) == 3:
                frame = np.mean(frame, axis=2)
            
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]
            
            frames.append(frame)
            
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(tif_files)} frames...")
        
        self.images = np.array(frames)
        self.params.n_frames = len(frames)
        print(f"✓ Loaded {self.params.n_frames} frames of size {self.images.shape[1:]}")
        
    def load_video(self, video_path: str, roi=None, max_frames=None):
        """Load video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
                
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]
            
            frames.append(frame.astype(np.float64))
            frame_count += 1
            
        cap.release()
        self.images = np.array(frames)
        self.params.n_frames = len(frames)
        print(f"✓ Loaded {self.params.n_frames} frames of size {self.images.shape[1:]}")
    
    def load_images(self, path: str, roi=None, max_frames=None):
        """Smart loader - auto-detects video or TIF directory"""
        path_obj = Path(path)
        
        if path_obj.is_dir():
            print(f"Loading TIF sequence from: {path}")
            self.load_tif_directory(path, roi=roi, max_frames=max_frames)
        elif path_obj.is_file():
            print(f"Loading video file: {path}")
            self.load_video(path, roi=roi, max_frames=max_frames)
        else:
            raise ValueError(f"Path does not exist: {path}")
    
    def spectrum_diff(self, im0, im1):
        """
        Compute squared modulus of 2D FFT of difference
        GPU-accelerated when available, CPU fallback
        """
        if self.use_gpu and TORCH_AVAILABLE:
            # GPU path using PyTorch
            im0_t = torch.from_numpy(im0.astype(np.float32)).to(self.device)
            im1_t = torch.from_numpy(im1.astype(np.float32)).to(self.device)
            
            diff_fft = torch.fft.fft2(im1_t - im0_t)
            power_spectrum = torch.abs(diff_fft) ** 2
            
            return power_spectrum.cpu().numpy()
        else:
            # CPU path using NumPy
            return np.abs(np.fft.fft2(im1 - im0.astype(float)))**2
    
    def time_averaged(self, dt, max_couples=None):
        """
        Time-averaged spectrum difference with GPU acceleration
        Uses batched operations on GPU for efficiency
        
        Parameters:
        -----------
        dt : int
            Time lag in frames
        max_couples : int, optional
            Maximum number of image pairs to average
            If None, uses all available pairs
        """
        available_pairs = len(self.images) - dt
        
        if max_couples is None or max_couples >= available_pairs:
            # Use all available pairs
            increment = 1
            initial_times = np.arange(0, available_pairs)
        else:
            # Sample evenly across available range
            increment = max(available_pairs // max_couples, 1)
            initial_times = np.arange(0, available_pairs, increment)
        
        if self.use_gpu and TORCH_AVAILABLE:
            # GPU-accelerated batch processing
            # Process in batches to manage memory
            batch_size = min(32, len(initial_times))  # Adjust based on GPU memory
            avg_fft = np.zeros(self.images.shape[1:], dtype=np.float32)
            
            for batch_start in range(0, len(initial_times), batch_size):
                batch_end = min(batch_start + batch_size, len(initial_times))
                batch_times = initial_times[batch_start:batch_end]
                
                # Create batches of image pairs
                im0_batch = torch.from_numpy(
                    self.images[batch_times].astype(np.float32)
                ).to(self.device)
                im1_batch = torch.from_numpy(
                    self.images[batch_times + dt].astype(np.float32)
                ).to(self.device)
                
                # Compute FFTs for entire batch
                diff_fft_batch = torch.fft.fft2(im1_batch - im0_batch)
                power_spectrum_batch = torch.abs(diff_fft_batch) ** 2
                
                # Sum and move back to CPU
                avg_fft += power_spectrum_batch.sum(dim=0).cpu().numpy()
            
            return avg_fft / len(initial_times), len(initial_times)
        else:
            # CPU path - original implementation
            avg_fft = np.zeros(self.images.shape[1:])
            for t in initial_times:
                avg_fft += self.spectrum_diff(self.images[t], self.images[t + dt])
            
            return avg_fft / len(initial_times), len(initial_times)
    
    def compute_structure_function(self, lag_times=None, points_per_decade=15, 
                                   max_couples=None):
        """
        Compute DDM structure function D(q,t)
        Combines notebook's ddm() function with radial averaging
        
        Parameters:
        -----------
        lag_times : array-like, optional
            Specific lag times. If None, uses log spacing
        points_per_decade : int
            Points per decade for log spacing (default: 15, as in notebook)
        max_couples : int, optional
            Maximum couples to average per lag (default: all available)
        """
        if lag_times is None:
            self.lag_times = log_spaced(self.params.n_frames, points_per_decade)
        else:
            self.lag_times = np.array(lag_times)
        
        print(f"\nComputing structure function...")
        print(f"  Lag times: {len(self.lag_times)} points")
        if max_couples:
            print(f"  Max couples per lag: {max_couples}")
        else:
            print(f"  Using all available couples")
        
        # Initialize radial averager
        ra = RadialAverager(self.images.shape[1:])
        self.q_values = ra.hd
        
        # Convert pixel q-values to physical units (um^-1)
        pixel_q = 2 * np.pi * ra.hd / self.images.shape[1]
        self.q_values = pixel_q * self.params.magnification / self.params.pixel_size_um
        
        # Compute structure function for each lag time
        self.structure_function = {}
        n_couples_used = {}
        
        for i, dt in enumerate(self.lag_times):
            avg_spectrum, n_couples = self.time_averaged(dt, max_couples)
            radial_avg = ra(avg_spectrum)
            self.structure_function[dt] = radial_avg
            n_couples_used[dt] = n_couples
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i + 1}/{len(self.lag_times)} lags completed")
        
        print(f"✓ Structure function computed")
        print(f"  Q-range: {self.q_values[0]:.3f} to {self.q_values[-1]:.3f} µm⁻¹")
        
        # Store metadata
        self.n_couples_used = n_couples_used
    
    def fit_exponential_growth(self, q_range=None):
        """
        Fit exponential growth model to D(q,t)
        Model: D(q,t) = A(q) * (1 - exp(-Γ(q)*t)) + B(q)
        """
        if self.structure_function is None:
            raise ValueError("Must compute structure function first")
        
        # Determine q-range to fit
        if q_range is None:
            # Use middle 70% of q-range (avoid edge artifacts)
            q_start = int(0.15 * len(self.q_values))
            q_end = int(0.85 * len(self.q_values))
        else:
            q_start, q_end = q_range
        
        self.q_fit = self.q_values[q_start:q_end]
        
        # Prepare time array in seconds
        time_sec = self.lag_times / self.params.frame_rate
        
        # Fit each q-value
        print(f"\nFitting exponential growth...")
        print(f"  Fitting {len(self.q_fit)} q-values")
        
        self.decay_rates = np.zeros(len(self.q_fit))
        self.amplitudes = np.zeros(len(self.q_fit))
        self.backgrounds = np.zeros(len(self.q_fit))
        
        def exponential_model(t, A, gamma, B):
            return A * (1 - np.exp(-gamma * t)) + B
        
        for i, q_idx in enumerate(range(q_start, q_end)):
            # Extract D(q,t) time series
            y_data = np.array([self.structure_function[lag][q_idx] 
                              for lag in self.lag_times])
            
            # Initial guess
            A_guess = y_data[-1] - y_data[0]
            gamma_guess = 1.0 / time_sec[len(time_sec)//2]
            B_guess = y_data[0]
            
            try:
                popt, _ = curve_fit(
                    exponential_model, time_sec, y_data,
                    p0=[A_guess, gamma_guess, B_guess],
                    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                    maxfev=5000
                )
                self.amplitudes[i] = popt[0]
                self.decay_rates[i] = popt[1]
                self.backgrounds[i] = popt[2]
            except:
                self.amplitudes[i] = np.nan
                self.decay_rates[i] = np.nan
                self.backgrounds[i] = np.nan
        
        print(f"✓ Fitting complete")
        
    def compute_diffusion_coefficient(self, fit_range=None):
        """
        Extract diffusion coefficient from Γ(q) = D*q²
        Returns D and standard deviation
        """
        if self.decay_rates is None:
            raise ValueError("Must fit exponential growth first")
        
        # Remove NaN values
        valid = ~np.isnan(self.decay_rates)
        q_valid = self.q_fit[valid]
        gamma_valid = self.decay_rates[valid]
        
        if len(q_valid) == 0:
            raise ValueError("No valid fits found")
        
        # Fit Γ vs q² to extract D
        q_squared = q_valid**2
        
        # Linear fit: Γ = D*q² + offset
        coeffs = np.polyfit(q_squared, gamma_valid, 1)
        self.diffusion_coeff = coeffs[0]
        
        # Compute standard deviation
        D_apparent = gamma_valid / q_squared
        self.diffusion_std = np.std(D_apparent)
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Diffusion coefficient: D = {self.diffusion_coeff:.4f} ± {self.diffusion_std:.4f} µm²/s")
        print(f"Hydrodynamic radius: R_h = {self.compute_hydrodynamic_radius():.2f} nm")
        print(f"{'='*60}\n")
        
        return self.diffusion_coeff, self.diffusion_std
    
    def compute_hydrodynamic_radius(self, temperature_C=25, viscosity_Pa_s=0.001):
        """
        Compute hydrodynamic radius using Stokes-Einstein equation
        R_h = kT / (6πηD)
        """
        if self.diffusion_coeff is None:
            return np.nan
        
        k_B = 1.380649e-23  # Boltzmann constant (J/K)
        T = temperature_C + 273.15  # Temperature in Kelvin
        
        # Convert D from µm²/s to m²/s
        D_SI = self.diffusion_coeff * 1e-12
        
        # Stokes-Einstein equation
        R_h_m = k_B * T / (6 * np.pi * viscosity_Pa_s * D_SI)
        
        # Convert to nanometers
        return R_h_m * 1e9
    
    def export_npy(self, output_path):
        """
        Export raw DDM data as .npy files (notebook-style)
        Saves: structure function, q values, lag times, parameters
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert structure function dict to 2D array
        sf_array = np.zeros((len(self.lag_times), len(self.q_values)))
        for i, lag in enumerate(self.lag_times):
            sf_array[i, :] = self.structure_function[lag]
        
        # Save arrays (notebook style)
        np.save(output_path / 'DDM.npy', sf_array)
        np.save(output_path / 'dt.npy', self.lag_times)
        np.save(output_path / 'q_values.npy', self.q_values)
        
        # Save fitted parameters if available
        if self.decay_rates is not None:
            np.save(output_path / 'gamma.npy', self.decay_rates)
            np.save(output_path / 'q_fit.npy', self.q_fit)
            np.save(output_path / 'amplitudes.npy', self.amplitudes)
            np.save(output_path / 'backgrounds.npy', self.backgrounds)
        
        # Save experimental parameters as JSON
        params_dict = {
            'pixel_size_um': self.params.pixel_size_um,
            'magnification': self.params.magnification,
            'frame_rate': self.params.frame_rate,
            'n_frames': self.params.n_frames,
            'roi_size': self.params.roi_size,
            'diffusion_coeff': float(self.diffusion_coeff) if self.diffusion_coeff else None,
            'diffusion_std': float(self.diffusion_std) if self.diffusion_std else None,
        }
        
        with open(output_path / 'parameters.json', 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        print(f"✓ Exported .npy files to {output_path}/")
        print(f"  - DDM.npy (structure function)")
        print(f"  - dt.npy (lag times)")
        print(f"  - q_values.npy")
        print(f"  - parameters.json")
        if self.decay_rates is not None:
            print(f"  - gamma.npy (decay rates)")
            print(f"  - q_fit.npy, amplitudes.npy, backgrounds.npy")
    
    def export_csv(self, output_path):
        """Export fit results to CSV"""
        if self.decay_rates is None:
            print("Warning: No fit results to export")
            return
        
        valid = ~np.isnan(self.decay_rates)
        
        results_df = pd.DataFrame({
            'q_um-1': self.q_fit[valid],
            'q2_um-2': self.q_fit[valid]**2,
            'Gamma_Hz': self.decay_rates[valid],
            'D_app_um2s-1': self.decay_rates[valid] / (self.q_fit[valid]**2),
            'Amplitude': self.amplitudes[valid],
            'Background': self.backgrounds[valid]
        })
        
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ Exported CSV to {output_path}")
    
    def export_summary(self, output_path, elapsed_time=None):
        """Export text summary with GPU and timing info"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("DDM Analysis Summary (GPU-Accelerated)\n")
            f.write("="*60 + "\n\n")
            
            if self.diffusion_coeff:
                f.write(f"Diffusion Coefficient:\n")
                f.write(f"  D = {self.diffusion_coeff:.4f} ± {self.diffusion_std:.4f} µm²/s\n\n")
                f.write(f"Hydrodynamic Radius:\n")
                f.write(f"  R_h = {self.compute_hydrodynamic_radius():.2f} nm\n\n")
            
            f.write(f"Experimental Parameters:\n")
            f.write(f"  Frames: {self.params.n_frames}\n")
            f.write(f"  Frame rate: {self.params.frame_rate} Hz\n")
            f.write(f"  Pixel size: {self.params.pixel_size_um} µm\n")
            f.write(f"  Magnification: {self.params.magnification}x\n")
            f.write(f"  ROI size: {self.params.roi_size} px\n\n")
            
            f.write(f"Analysis Details:\n")
            f.write(f"  Lag times analyzed: {len(self.lag_times)}\n")
            f.write(f"  Q-range: {self.q_values[0]:.3f} to {self.q_values[-1]:.3f} µm⁻¹\n")
            if hasattr(self, 'n_couples_used'):
                min_couples = min(self.n_couples_used.values())
                max_couples = max(self.n_couples_used.values())
                f.write(f"  Couples averaged: {min_couples} to {max_couples} per lag\n")
            
            # GPU information
            f.write(f"\nCompute Device:\n")
            if self.use_gpu and TORCH_AVAILABLE:
                device_name = str(self.device)
                if 'cuda' in device_name:
                    f.write(f"  GPU: NVIDIA {torch.cuda.get_device_name(0)}\n")
                elif 'xpu' in device_name:
                    try:
                        f.write(f"  GPU: Intel {torch.xpu.get_device_name(0)}\n")
                    except:
                        f.write(f"  GPU: Intel XPU\n")
                else:
                    f.write(f"  Device: {device_name}\n")
            else:
                f.write(f"  Device: CPU\n")
            
            if elapsed_time is not None:
                f.write(f"\nComputation Time:\n")
                hours = int(elapsed_time // 3600)
                minutes = int((elapsed_time % 3600) // 60)
                seconds = elapsed_time % 60
                if hours > 0:
                    f.write(f"  Total time: {hours}h {minutes}m {seconds:.1f}s ({elapsed_time:.1f} seconds)\n")
                elif minutes > 0:
                    f.write(f"  Total time: {minutes}m {seconds:.1f}s ({elapsed_time:.1f} seconds)\n")
                else:
                    f.write(f"  Total time: {elapsed_time:.2f} seconds\n")
        
        print(f"✓ Exported summary to {output_path}")
    
    def plot_results(self, save_path=None):
        """Create comprehensive analysis plot"""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Sample images
        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(self.images[0], cmap='gray')
        ax1.set_title('First Frame')
        ax1.axis('off')
        
        ax2 = plt.subplot(3, 4, 2)
        ax2.imshow(self.images[-1], cmap='gray')
        ax2.set_title('Last Frame')
        ax2.axis('off')
        
        ax3 = plt.subplot(3, 4, 3)
        diff = self.images[-1] - self.images[0]
        ax3.imshow(diff, cmap='gray')
        ax3.set_title('Difference (Last - First)')
        ax3.axis('off')
        
        # 4. FFT spectrum example
        ax4 = plt.subplot(3, 4, 4)
        fft_example = self.spectrum_diff(self.images[0], self.images[10])
        ax4.imshow(np.log10(np.fft.fftshift(fft_example) + 1), cmap='hot')
        ax4.set_title('FFT Spectrum (log scale)')
        ax4.axis('off')
        
        # 5. Radial profile at one lag
        ax5 = plt.subplot(3, 4, 5)
        mid_lag = self.lag_times[len(self.lag_times)//2]
        ax5.plot(self.q_values, self.structure_function[mid_lag])
        ax5.set_xlabel('q (µm⁻¹)')
        ax5.set_ylabel('D(q,t)')
        ax5.set_title(f'Radial Profile (t={mid_lag/self.params.frame_rate:.3f}s)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Structure function time evolution
        ax6 = plt.subplot(3, 4, 6)
        time_sec = self.lag_times / self.params.frame_rate
        q_indices = [len(self.q_values)//4, len(self.q_values)//2, 3*len(self.q_values)//4]
        for idx in q_indices:
            if idx < len(self.q_values):
                y_data = [self.structure_function[lag][idx] for lag in self.lag_times]
                ax6.plot(time_sec, y_data, 'o-', label=f'q = {self.q_values[idx]:.2f}')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('D(q,t)')
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        ax6.legend(fontsize=8)
        ax6.set_title('Structure Function Growth')
        ax6.grid(True, alpha=0.3)
        
        # 7. Structure function vs q at different times
        ax7 = plt.subplot(3, 4, 7)
        time_indices = [0, len(self.lag_times)//4, len(self.lag_times)//2, -1]
        for idx in time_indices:
            lag = self.lag_times[idx]
            ax7.plot(self.q_values, self.structure_function[lag],
                    label=f't={lag/self.params.frame_rate:.4f}s')
        ax7.set_xlabel('q (µm⁻¹)')
        ax7.set_ylabel('D(q,t)')
        ax7.legend(fontsize=7)
        ax7.set_title('D(q,t) vs q')
        ax7.grid(True, alpha=0.3)
        
        if self.decay_rates is not None:
            # 8. Gamma vs q (log-log)
            ax8 = plt.subplot(3, 4, 8)
            valid = ~np.isnan(self.decay_rates)
            ax8.loglog(self.q_fit[valid], self.decay_rates[valid], 'o', label='Data')
            q_theory = np.linspace(self.q_fit[valid].min(), self.q_fit[valid].max(), 100)
            gamma_theory = self.diffusion_coeff * q_theory**2
            ax8.loglog(q_theory, gamma_theory, 'r--', 
                      label=f'D={self.diffusion_coeff:.3f} µm²/s')
            ax8.set_xlabel('q (µm⁻¹)')
            ax8.set_ylabel('Γ (Hz)')
            ax8.legend(fontsize=8)
            ax8.set_title('Decay Rate vs q')
            ax8.grid(True, alpha=0.3)
            
            # 9. Gamma vs q² (linear)
            ax9 = plt.subplot(3, 4, 9)
            q_squared = self.q_fit[valid]**2
            ax9.plot(q_squared, self.decay_rates[valid], 'o', label='Data')
            slope, intercept = np.polyfit(q_squared, self.decay_rates[valid], 1)
            ax9.plot(q_squared, slope*q_squared + intercept, 'r--',
                    label=f'D={slope:.4f} µm²/s')
            ax9.set_xlabel('q² (µm⁻²)')
            ax9.set_ylabel('Γ (Hz)')
            ax9.legend(fontsize=8)
            ax9.set_title('Linearized: Γ vs q²')
            ax9.grid(True, alpha=0.3)
            
            # 10. Apparent D vs q² (quality check)
            ax10 = plt.subplot(3, 4, 10)
            D_app = self.decay_rates[valid] / (self.q_fit[valid]**2)
            ax10.plot(q_squared, D_app, 'o')
            ax10.axhline(y=self.diffusion_coeff, color='r', linestyle='--',
                        label=f'Mean D={self.diffusion_coeff:.4f}')
            ax10.set_xlabel('q² (µm⁻²)')
            ax10.set_ylabel('D_app (µm²/s)')
            ax10.legend(fontsize=8)
            ax10.set_title('Apparent Diffusion Coefficient')
            ax10.grid(True, alpha=0.3)
        
        # 11. Number of couples used
        ax11 = plt.subplot(3, 4, 11)
        if hasattr(self, 'n_couples_used'):
            couples = [self.n_couples_used[lag] for lag in self.lag_times]
            ax11.plot(self.lag_times, couples, 'o-')
            ax11.set_xlabel('Lag time (frames)')
            ax11.set_ylabel('Number of couples averaged')
            ax11.set_title('Averaging Statistics')
            ax11.grid(True, alpha=0.3)
        
        # 12. Summary text
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        summary = f"""DDM Analysis Summary
        
Parameters:
  Frames: {self.params.n_frames}
  Frame rate: {self.params.frame_rate} Hz
  Pixel size: {self.params.pixel_size_um} µm
  Magnification: {self.params.magnification}x
  
Results:"""
        
        if self.diffusion_coeff:
            summary += f"""
  D = {self.diffusion_coeff:.4f} ± {self.diffusion_std:.4f} µm²/s
  R_h = {self.compute_hydrodynamic_radius():.1f} nm
  
Analysis:
  Lag times: {len(self.lag_times)}
  Q-range: {self.q_values[0]:.3f} - {self.q_values[-1]:.3f} µm⁻¹"""
        
        ax12.text(0.1, 0.5, summary, fontsize=9, verticalalignment='center',
                 fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.close()


def merge_ddm_datasets(analyzers: List[DDMAnalyzer], frequencies: List[float],
                      overlap_points: int = 5) -> DDMAnalyzer:
    """
    Merge DDM results from multiple acquisition frequencies
    Implements notebook's multi-frequency merging capability
    
    Parameters:
    -----------
    analyzers : list of DDMAnalyzer
        Analyzers in order from highest to lowest frequency
    frequencies : list of float
        Corresponding frame rates (Hz)
    overlap_points : int
        Number of points for overlap region analysis
        
    Returns:
    --------
    merged : DDMAnalyzer
        New analyzer with merged data
    """
    print(f"\n{'='*60}")
    print(f"MULTI-FREQUENCY MERGING")
    print(f"{'='*60}")
    print(f"Merging {len(analyzers)} datasets at frequencies: {frequencies} Hz")
    
    # Convert lag times to seconds
    time_arrays = []
    for analyzer, freq in zip(analyzers, frequencies):
        times_sec = analyzer.lag_times / freq
        time_arrays.append(times_sec)
    
    # Find merge boundaries
    merge_indices = []
    
    for i in range(len(analyzers) - 1):
        high_times = time_arrays[i]
        low_times = time_arrays[i + 1]
        
        # Low frequency starts being good after overlap_points
        low_start_time = low_times[overlap_points]
        
        # Find where high frequency reaches this time
        boundary_idx = np.searchsorted(high_times, low_start_time)
        
        print(f"\nOverlap {i+1}:")
        print(f"  High freq ends: {high_times[boundary_idx]:.4f} s (frame {analyzers[i].lag_times[boundary_idx]})")
        print(f"  Low freq starts: {low_times[overlap_points]:.4f} s (frame {analyzers[i+1].lag_times[overlap_points]})")
        
        merge_indices.append((boundary_idx, overlap_points))
    
    # Build merged dataset
    merged_lags = []
    merged_times = []
    merged_sf = []
    
    # First dataset up to boundary
    boundary, _ = merge_indices[0]
    for j, lag in enumerate(analyzers[0].lag_times[:boundary]):
        merged_lags.append(lag)
        merged_times.append(time_arrays[0][j])
        merged_sf.append(analyzers[0].structure_function[lag])
    
    # Middle datasets
    for i in range(1, len(analyzers) - 1):
        _, start = merge_indices[i-1]
        end, _ = merge_indices[i]
        for j, lag in enumerate(analyzers[i].lag_times[start:end], start):
            merged_lags.append(lag)
            merged_times.append(time_arrays[i][j])
            merged_sf.append(analyzers[i].structure_function[lag])
    
    # Last dataset
    _, start = merge_indices[-1]
    for j, lag in enumerate(analyzers[-1].lag_times[start:], start):
        merged_lags.append(lag)
        merged_times.append(time_arrays[-1][j])
        merged_sf.append(analyzers[-1].structure_function[lag])
    
    # Create merged analyzer
    merged = DDMAnalyzer(analyzers[0].params, use_gpu=analyzers[0].use_gpu)
    merged.lag_times = np.array(merged_lags)
    merged.structure_function = {lag: sf for lag, sf in zip(merged_lags, merged_sf)}
    merged.q_values = analyzers[0].q_values
    merged.images = analyzers[0].images  # Keep first dataset's images for plotting
    merged.device = analyzers[0].device  # Keep device settings
    
    decades = np.log10(merged_times[-1] / merged_times[0])
    
    print(f"\n✓ Merge complete:")
    print(f"  Total points: {len(merged.lag_times)}")
    print(f"  Time range: {merged_times[0]:.4f} to {merged_times[-1]:.2f} s")
    print(f"  Decades covered: {decades:.2f}")
    print(f"{'='*60}\n")
    
    return merged


def run_ddm_analysis(image_path: str, pixel_size_um: float, magnification: float,
                    frame_rate: float, roi=None, output_dir=None, max_frames=None,
                    points_per_decade=15, max_couples=None, use_gpu=True):
    """
    Complete DDM analysis pipeline with GPU acceleration
    
    Parameters:
    -----------
    image_path : str
        Path to video or TIF directory
    pixel_size_um : float
        Physical pixel size (µm)
    magnification : float
        Microscope magnification
    frame_rate : float
        Frame rate (Hz)
    roi : tuple, optional
        Region of interest (x, y, width, height)
    output_dir : str, optional
        Output directory for results
    max_frames : int, optional
        Maximum frames to load
    points_per_decade : int
        Lag time density (default: 15, as in notebook)
    max_couples : int, optional
        Maximum couples to average per lag
    use_gpu : bool
        Enable GPU acceleration (default: True)
    """
    # Start timing
    start_time = time.time()
    
    params = DDMParameters(
        pixel_size_um=pixel_size_um,
        magnification=magnification,
        frame_rate=frame_rate,
        n_frames=0
    )
    
    analyzer = DDMAnalyzer(params, use_gpu=use_gpu)
    
    print("="*60)
    print("DDM ANALYSIS - GPU-Accelerated Edition")
    print("="*60)
    
    print("\nStep 1: Loading images...")
    analyzer.load_images(image_path, roi=roi, max_frames=max_frames)
    
    print("\nStep 2: Computing structure function...")
    analyzer.compute_structure_function(
        points_per_decade=points_per_decade,
        max_couples=max_couples
    )
    
    print("\nStep 3: Fitting exponential growth...")
    analyzer.fit_exponential_growth()
    
    print("\nStep 4: Computing diffusion coefficient...")
    analyzer.compute_diffusion_coefficient()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Export results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\nStep 5: Exporting results...")
        
        # .npy files (notebook-style)
        analyzer.export_npy(output_path)
        
        # CSV and summary (with timing)
        analyzer.export_csv(output_path / 'ddm_results.csv')
        analyzer.export_summary(output_path / 'ddm_summary.txt', elapsed_time=elapsed_time)
        
        # Plot
        analyzer.plot_results(save_path=str(output_path / 'ddm_analysis.png'))
        
        # Format and print elapsed time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        
        if hours > 0:
            print(f"Total computation time: {hours}h {minutes}m {seconds:.1f}s")
        elif minutes > 0:
            print(f"Total computation time: {minutes}m {seconds:.1f}s")
        else:
            print(f"Total computation time: {seconds:.2f} seconds")
        
        print(f"\nResults saved to: {output_dir}")
        print("  - DDM.npy, dt.npy, q_values.npy (raw data)")
        print("  - ddm_results.csv (fitted parameters)")
        print("  - ddm_summary.txt (text summary)")
        print("  - ddm_analysis.png (comprehensive plot)")
    else:
        # Still print timing even if not saving
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        
        if hours > 0:
            print(f"Total computation time: {hours}h {minutes}m {seconds:.1f}s")
        elif minutes > 0:
            print(f"Total computation time: {minutes}m {seconds:.1f}s")
        else:
            print(f"Total computation time: {seconds:.2f} seconds")
    
    return analyzer


def main():
    parser = argparse.ArgumentParser(
        description='DDM Analysis - GPU-Accelerated Edition with PyTorch support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single frequency analysis (auto-detect GPU)
  python pyDDMAnalyzer_GPU.py data/video.mp4 --pixel-size 0.108 --mag 100 --fps 400 -o results/

  # Multi-frequency merging
  python pyDDMAnalyzer_GPU.py --merge data/400Hz/ data/4Hz/ --frequencies 400 4 \\
         --pixel-size 0.108 --mag 100 -o results_merged/

  # Force CPU mode (disable GPU)
  python pyDDMAnalyzer_GPU.py data/test/ --pixel-size 0.108 --mag 100 --fps 400 \\
         --no-gpu -o test_results/

  # Fast evaluation mode
  python pyDDMAnalyzer_GPU.py data/test/ --pixel-size 0.108 --mag 100 --fps 400 \\
         --max-couples 10 --max-frames 500 -o test_results/
        """
    )
    
    # Input
    parser.add_argument('input', nargs='+', help='Video file or TIF directory (or multiple for --merge)')
    parser.add_argument('--merge', action='store_true', 
                       help='Merge multiple frequency datasets')
    parser.add_argument('--frequencies', nargs='+', type=float,
                       help='Frame rates for each input (required for --merge)')
    
    # Parameters
    parser.add_argument('--pixel-size', type=float, required=True,
                       help='Pixel size in micrometers')
    parser.add_argument('--mag', '--magnification', type=float, required=True,
                       help='Microscope magnification')
    parser.add_argument('--fps', '--frame-rate', type=float,
                       help='Frame rate (Hz) - not needed for --merge')
    
    # Optional parameters
    parser.add_argument('--roi', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'),
                       help='Region of interest: x y width height')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to load (for testing)')
    parser.add_argument('--max-couples', type=int,
                       help='Maximum couples to average per lag (default: all)')
    parser.add_argument('--points-per-decade', type=int, default=15,
                       help='Lag time density (default: 15)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration (use CPU only)')
    
    # Output
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.merge:
        if not args.frequencies:
            parser.error("--frequencies required for --merge")
        if len(args.input) != len(args.frequencies):
            parser.error("Number of inputs must match number of frequencies")
        if len(args.input) < 2:
            parser.error("--merge requires at least 2 inputs")
    else:
        if not args.fps:
            parser.error("--fps required for single-file analysis")
        if len(args.input) != 1:
            parser.error("Single analysis mode requires exactly 1 input")
    
    try:
        use_gpu = not args.no_gpu
        
        if args.merge:
            # Multi-frequency analysis
            merge_start_time = time.time()
            
            print("MULTI-FREQUENCY MODE (GPU-Accelerated)")
            print("="*60)
            
            analyzers = []
            for path, freq in zip(args.input, args.frequencies):
                print(f"\nAnalyzing: {path} at {freq} Hz")
                analyzer = run_ddm_analysis(
                    image_path=path,
                    pixel_size_um=args.pixel_size,
                    magnification=args.mag,
                    frame_rate=freq,
                    roi=tuple(args.roi) if args.roi else None,
                    output_dir=None,  # Don't save individual results
                    max_frames=args.max_frames,
                    points_per_decade=args.points_per_decade,
                    max_couples=args.max_couples,
                    use_gpu=use_gpu
                )
                analyzers.append(analyzer)
            
            # Merge
            merged = merge_ddm_datasets(analyzers, args.frequencies)
            
            # Fit and export merged results
            print("Fitting merged dataset...")
            merged.fit_exponential_growth()
            merged.compute_diffusion_coefficient()
            
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            merged.export_npy(output_path)
            merged.export_csv(output_path / 'ddm_results_merged.csv')
            
            # Calculate total elapsed time for merged analysis
            merge_elapsed_time = time.time() - merge_start_time
            merged.export_summary(output_path / 'ddm_summary_merged.txt', elapsed_time=merge_elapsed_time)
            
            merged.plot_results(save_path=str(output_path / 'ddm_analysis_merged.png'))
            
            # Format and print elapsed time
            hours = int(merge_elapsed_time // 3600)
            minutes = int((merge_elapsed_time % 3600) // 60)
            seconds = merge_elapsed_time % 60
            
            print(f"\n{'='*60}")
            print("MERGED ANALYSIS COMPLETE!")
            print(f"{'='*60}")
            
            if hours > 0:
                print(f"Total computation time: {hours}h {minutes}m {seconds:.1f}s")
            elif minutes > 0:
                print(f"Total computation time: {minutes}m {seconds:.1f}s")
            else:
                print(f"Total computation time: {seconds:.2f} seconds")
            
        else:
            # Single frequency analysis
            run_ddm_analysis(
                image_path=args.input[0],
                pixel_size_um=args.pixel_size,
                magnification=args.mag,
                frame_rate=args.fps,
                roi=tuple(args.roi) if args.roi else None,
                output_dir=args.output,
                max_frames=args.max_frames,
                points_per_decade=args.points_per_decade,
                max_couples=args.max_couples,
                use_gpu=use_gpu
            )
    
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()