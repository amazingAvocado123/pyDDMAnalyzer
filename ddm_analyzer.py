"""
DDM Analyzer - Core Module
Differential Dynamic Microscopy for measuring particle diffusion
Supports video files AND sequential TIF image directories
"""

import sys
import io

# Force UTF-8 encoding for console output (Windows compatibility)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
# Configure matplotlib for UTF-8
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
import cv2
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import re
from PIL import Image


@dataclass
class DDMParameters:
    """Store DDM experimental parameters"""
    pixel_size_um: float  # Physical size of one pixel in micrometers
    magnification: float
    frame_rate: float  # frames per second
    n_frames: int
    roi_size: int = 512  # ROI size in pixels
    
    @property
    def q_min(self):
        """Minimum wave vector (um^-1)"""
        return 2 * np.pi / (self.roi_size * self.pixel_size_um / self.magnification)
    
    @property
    def q_max(self):
        """Maximum wave vector (um^-1)"""
        return np.pi * self.magnification / self.pixel_size_um


def natural_sort_key(s):
    """
    Sort strings containing numbers in natural order
    Example: ['img1.tif', 'img2.tif', 'img10.tif'] instead of ['img1.tif', 'img10.tif', 'img2.tif']
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


class DDMAnalyzer:
    """
    Differential Dynamic Microscopy Analyzer
    
    Measures diffusion coefficients from microscope videos or TIF sequences
    """
    
    def __init__(self, params: DDMParameters):
        self.params = params
        self.images = None
        self.structure_function = None
        self.q_values = None
        
    def load_tif_directory(self, directory_path: str, roi=None, max_frames=None):
        """
        Load sequential TIF files from a directory
        
        Parameters:
        -----------
        directory_path : str
            Path to directory containing TIF files
        roi : tuple (x, y, width, height), optional
            Region of interest to extract
        max_frames : int, optional
            Maximum number of frames to load (useful for testing)
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all TIF files
        tif_files = list(directory.glob("*.tif")) + list(directory.glob("*.tiff")) + \
                    list(directory.glob("*.TIF")) + list(directory.glob("*.TIFF"))
        
        if not tif_files:
            raise ValueError(f"No TIF files found in {directory_path}")
        
        # Sort files naturally (img1, img2, img10 instead of img1, img10, img2)
        tif_files.sort(key=natural_sort_key)
        
        print(f"Found {len(tif_files)} TIF files")
        print(f"First file: {tif_files[0].name}")
        print(f"Last file: {tif_files[-1].name}")
        
        # Limit number of frames if specified
        if max_frames is not None and max_frames < len(tif_files):
            tif_files = tif_files[:max_frames]
            print(f"Loading first {max_frames} frames")
        
        frames = []
        for i, tif_file in enumerate(tif_files):
            # Load TIF file
            img = Image.open(tif_file)
            frame = np.array(img).astype(np.float64)
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame = np.mean(frame, axis=2)  # Simple RGB to grayscale
            
            # Extract ROI if specified
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]
            
            frames.append(frame)
            
            # Progress update every 100 frames
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(tif_files)} frames...")
        
        self.images = np.array(frames)
        self.params.n_frames = len(frames)
        print(f"[OK] Loaded {self.params.n_frames} frames of size {self.images.shape[1:]}")
        
    def load_video(self, video_path: str, roi=None):
        """
        Load video file and extract frames
        
        Parameters:
        -----------
        video_path : str
            Path to video file
        roi : tuple (x, y, width, height), optional
            Region of interest to extract
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract ROI if specified
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]
            
            frames.append(frame.astype(np.float64))
            
        cap.release()
        self.images = np.array(frames)
        self.params.n_frames = len(frames)
        print(f"Loaded {self.params.n_frames} frames of size {self.images.shape[1:]}")
    
    def load_images(self, path: str, roi=None, max_frames=None):
        """
        Smart loader - detects if path is a video file or directory of TIFs
        
        Parameters:
        -----------
        path : str
            Path to video file OR directory containing TIF files
        roi : tuple (x, y, width, height), optional
            Region of interest to extract
        max_frames : int, optional
            Maximum number of frames to load
        """
        path_obj = Path(path)
        
        if path_obj.is_dir():
            # It's a directory - load TIF sequence
            print(f"Loading TIF sequence from directory: {path}")
            self.load_tif_directory(path, roi=roi, max_frames=max_frames)
        elif path_obj.is_file():
            # It's a file - load as video
            print(f"Loading video file: {path}")
            self.load_video(path, roi=roi)
        else:
            raise ValueError(f"Path does not exist: {path}")
        
    def compute_difference_images(self, lag_times=None):
        """
        Compute difference images for various lag times
        
        Parameters:
        -----------
        lag_times : array-like, optional
            Lag times in frame indices. If None, uses logarithmic spacing
        """
        if lag_times is None:
            # Logarithmic spacing as mentioned in the paper
            lag_times = np.unique(np.logspace(0, 
                                             np.log10(self.params.n_frames // 4), 
                                             num=50).astype(int))
        
        self.lag_times = lag_times
        self.diff_images = {}
        
        print("Computing difference images...")
        for lag in lag_times:
            diffs = []
            for i in range(self.params.n_frames - lag):
                diff = self.images[i + lag] - self.images[i]
                diffs.append(diff)
            self.diff_images[lag] = np.array(diffs)
            
        print(f"[OK] Computed difference images for {len(lag_times)} lag times")
        
    def compute_structure_function(self):
        """
        Compute the image structure function |F(q,t)|^2 = |D(q,t)|^2
        
        This is the core DDM calculation
        """
        # Get image dimensions
        ny, nx = self.images.shape[1:3]
        
        # Create frequency arrays
        qx = 2 * np.pi * np.fft.fftfreq(nx, 
                                         d=self.params.pixel_size_um/self.params.magnification)
        qy = 2 * np.pi * np.fft.fftfreq(ny, 
                                         d=self.params.pixel_size_um/self.params.magnification)
        Qx, Qy = np.meshgrid(qx, qy)
        Q = np.sqrt(Qx**2 + Qy**2)
        
        # Compute structure function for each lag time
        self.structure_function = {}
        
        print("Computing structure function...")
        for lag in self.lag_times:
            # Compute 2D FFT of each difference image
            fft_stack = np.fft.fft2(self.diff_images[lag])
            
            # Compute |F(q,t)|^2 and average over all pairs
            power_spectrum = np.abs(fft_stack)**2
            avg_power = np.mean(power_spectrum, axis=0)
            
            # Azimuthal averaging
            # Create bins for different q magnitudes
            q_bins = np.linspace(0, np.max(Q), 100)
            q_centers = (q_bins[:-1] + q_bins[1:]) / 2
            
            structure_func = np.zeros(len(q_centers))
            for i, (q_low, q_high) in enumerate(zip(q_bins[:-1], q_bins[1:])):
                mask = (Q >= q_low) & (Q < q_high)
                if np.any(mask):
                    structure_func[i] = np.mean(avg_power[mask])
            
            self.structure_function[lag] = structure_func
        
        self.q_values = q_centers
        print(f"[OK] Computed structure function for q range: {q_centers[0]:.3f} to {q_centers[-1]:.3f} um^-1")
        
    def fit_exponential_growth(self, q_range=None):
        """
        Fit structure function to exponential growth model:
        D(q,t) = A(q)[1 - exp(-Gamma(q)t)] + B(q)
        
        Parameters:
        -----------
        q_range : tuple (q_min, q_max), optional
            Range of q values to analyze
        """
        def structure_model(t, A, Gamma, B):
            """Model: A(1 - exp(-Gammat)) + B"""
            return A * (1 - np.exp(-Gamma * t)) + B
        
        # Select q range
        if q_range is not None:
            q_mask = (self.q_values >= q_range[0]) & (self.q_values <= q_range[1])
        else:
            # Use middle 80% of q range (avoid artifacts at extremes)
            n_q = len(self.q_values)
            q_mask = np.zeros(n_q, dtype=bool)
            q_mask[int(0.1*n_q):int(0.9*n_q)] = True
        
        q_selected = self.q_values[q_mask]
        
        # Convert lag times to seconds
        lag_times_sec = np.array(self.lag_times) / self.params.frame_rate
        
        # Fit for each q value
        self.decay_rates = np.zeros(len(q_selected))
        self.amplitudes = np.zeros(len(q_selected))
        self.backgrounds = np.zeros(len(q_selected))
        
        print("Fitting exponential growth...")
        for i, q_idx in enumerate(np.where(q_mask)[0]):
            # Get structure function values for this q at all lag times
            y_data = np.array([self.structure_function[lag][q_idx] 
                              for lag in self.lag_times])
            
            # Initial parameter guesses
            A_guess = np.max(y_data) - np.min(y_data)
            B_guess = np.min(y_data)
            Gamma_guess = 1.0 / np.median(lag_times_sec)
            
            try:
                popt, _ = curve_fit(structure_model, lag_times_sec, y_data,
                                   p0=[A_guess, Gamma_guess, B_guess],
                                   bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                                   maxfev=5000)
                
                self.amplitudes[i] = popt[0]
                self.decay_rates[i] = popt[1]
                self.backgrounds[i] = popt[2]
            except RuntimeError:
                print(f"Warning: Fit failed for q = {q_selected[i]:.3f}")
                self.decay_rates[i] = np.nan
        
        self.q_fit = q_selected
        print(f"[OK] Fitted {np.sum(~np.isnan(self.decay_rates))} q values successfully")
        
    def compute_diffusion_coefficient(self):
        """
        Compute diffusion coefficient from D = Gamma(q) / q^2
        
        Returns:
        --------
        D : float
            Diffusion coefficient in um^2/s
        D_std : float
            Standard deviation of D measurements
        """
        # Remove NaN values
        valid = ~np.isnan(self.decay_rates)
        q_valid = self.q_fit[valid]
        gamma_valid = self.decay_rates[valid]
        
        # Compute D for each q
        D_values = gamma_valid / (q_valid**2)
        
        # Fit Gamma vs q^2 to get D from slope
        # This is more robust than averaging D values
        popt, pcov = np.polyfit(q_valid**2, gamma_valid, 1, cov=True)
        D = popt[0]
        D_std = np.sqrt(pcov[0, 0])
        
        self.diffusion_coeff = D
        self.diffusion_std = D_std
        
        print(f"\n[OK] Diffusion Coefficient: D = {D:.3f} +/- {D_std:.3f} um^2/s")
        
        return D, D_std
    
    def compute_hydrodynamic_radius(self, temperature=298, viscosity=0.001):
        """
        Compute hydrodynamic radius using Stokes-Einstein equation
        
        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin (default 298 K = 25 degC)
        viscosity : float
            Solvent viscosity in Pa*s (default 0.001 = water at 25 degC)
        
        Returns:
        --------
        R_h : float
            Hydrodynamic radius in nm
        """
        k_B = 1.38064852e-23  # Boltzmann constant (J/K)
        
        # Convert D from um^2/s to m^2/s
        D_SI = self.diffusion_coeff * 1e-12
        
        # Stokes-Einstein: R_h = k_B*T / (6*pi*eta*D)
        R_h = k_B * temperature / (6 * np.pi * viscosity * D_SI)
        
        # Convert to nm
        R_h_nm = R_h * 1e9
        
        print(f"[OK] Hydrodynamic Radius: R_h = {R_h_nm:.1f} nm")
        
        return R_h_nm
    
    def plot_results(self, save_path=None):
        """
        Create comprehensive plots of DDM analysis results
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Sample images
        ax1 = plt.subplot(3, 3, 1)
        ax1.imshow(self.images[0], cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # 2. Difference image
        ax2 = plt.subplot(3, 3, 2)
        lag_mid = self.lag_times[len(self.lag_times)//2]
        ax2.imshow(self.diff_images[lag_mid][0], cmap='RdBu')
        ax2.set_title(f'Difference Image (Deltat = {lag_mid} frames)')
        ax2.axis('off')
        
        # 3. 2D FFT of difference image
        ax3 = plt.subplot(3, 3, 3)
        fft_img = np.fft.fft2(self.diff_images[lag_mid][0])
        ax3.imshow(np.log10(np.abs(np.fft.fftshift(fft_img)) + 1), cmap='hot')
        ax3.set_title('2D FFT (log scale)')
        ax3.axis('off')
        
        # 4. Structure function vs lag time for selected q
        ax4 = plt.subplot(3, 3, 4)
        q_indices = [len(self.q_fit)//4, len(self.q_fit)//2, 3*len(self.q_fit)//4]
        lag_times_sec = np.array(self.lag_times) / self.params.frame_rate
        
        for idx in q_indices:
            if idx < len(self.q_fit):
                q_val = self.q_fit[idx]
                # Find corresponding index in q_values
                q_idx = np.argmin(np.abs(self.q_values - q_val))
                y_data = [self.structure_function[lag][q_idx] for lag in self.lag_times]
                ax4.plot(lag_times_sec, y_data, 'o-', label=f'q = {q_val:.2f} um^-1')
        
        ax4.set_xlabel('Lag time (s)')
        ax4.set_ylabel('|D(q,t)|^2')
        ax4.set_xscale('log')
        ax4.legend()
        ax4.set_title('Structure Function Growth')
        ax4.grid(True, alpha=0.3)
        
        # 5. Structure function vs q for selected lag times
        ax5 = plt.subplot(3, 3, 5)
        lag_indices = [0, len(self.lag_times)//4, len(self.lag_times)//2, -1]
        
        for idx in lag_indices:
            lag = self.lag_times[idx]
            ax5.plot(self.q_values, self.structure_function[lag], 
                    label=f't = {lag/self.params.frame_rate:.3f} s')
        
        ax5.set_xlabel('q (um^-1)')
        ax5.set_ylabel('|D(q,t)|^2')
        ax5.legend()
        ax5.set_title('Structure Function vs q')
        ax5.grid(True, alpha=0.3)
        
        # 6. Decay rate vs q
        ax6 = plt.subplot(3, 3, 6)
        valid = ~np.isnan(self.decay_rates)
        ax6.loglog(self.q_fit[valid], self.decay_rates[valid], 'o', label='Data')
        
        # Plot theoretical q^2 dependence
        q_theory = np.linspace(self.q_fit[valid].min(), self.q_fit[valid].max(), 100)
        gamma_theory = self.diffusion_coeff * q_theory**2
        ax6.loglog(q_theory, gamma_theory, 'r--', label=f'D = {self.diffusion_coeff:.2f} um^2/s')
        
        ax6.set_xlabel('q (um^-1)')
        ax6.set_ylabel('Gamma (Hz)')
        ax6.legend()
        ax6.set_title('Decay Rate vs q')
        ax6.grid(True, alpha=0.3)
        
        # 7. Gamma vs q^2 (linearized)
        ax7 = plt.subplot(3, 3, 7)
        q_squared = self.q_fit[valid]**2
        ax7.plot(q_squared, self.decay_rates[valid], 'o', label='Data')
        
        # Linear fit
        slope, intercept = np.polyfit(q_squared, self.decay_rates[valid], 1)
        ax7.plot(q_squared, slope*q_squared + intercept, 'r--', 
                label=f'Fit: Gamma = {slope:.3f}q^2 + {intercept:.3f}')
        
        ax7.set_xlabel('q^2 (um^-2)')
        ax7.set_ylabel('Gamma (Hz)')
        ax7.legend()
        ax7.set_title('Linearized: Gamma vs q^2')
        ax7.grid(True, alpha=0.3)
        
        # 8. Apparent D vs q^2 (should be flat for Brownian motion)
        ax8 = plt.subplot(3, 3, 8)
        D_app = self.decay_rates[valid] / (self.q_fit[valid]**2)
        ax8.plot(q_squared, D_app, 'o')
        ax8.axhline(y=self.diffusion_coeff, color='r', linestyle='--', 
                   label=f'D = {self.diffusion_coeff:.3f} um^2/s')
        
        ax8.set_xlabel('q^2 (um^-2)')
        ax8.set_ylabel('D_app (um^2/s)')
        ax8.legend()
        ax8.set_title('Apparent Diffusion Coefficient')
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary text
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_text = f"""
        DDM Analysis Summary
        --------------------
        
        Experimental Parameters:
        - Frames: {self.params.n_frames}
        - Frame rate: {self.params.frame_rate} fps
        - Pixel size: {self.params.pixel_size_um} um
        - Magnification: {self.params.magnification}x
        
        Results:
        - D = {self.diffusion_coeff:.3f} +/- {self.diffusion_std:.3f} um^2/s
        - R_h = {self.compute_hydrodynamic_radius():.1f} nm
        
        Q-range analyzed:
        - q_min = {self.q_fit.min():.3f} um^-1
        - q_max = {self.q_fit.max():.3f} um^-1
        """
        
        ax9.text(0.1, 0.5, summary_text, fontsize=10, 
                verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.close()
        
    def export_results(self, output_path):
        """
        Export analysis results to CSV file
        """
        # Create DataFrame with results
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
        print(f"Exported results to {output_path}")
        
        # Also save summary
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"DDM Analysis Summary\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Diffusion Coefficient: D = {self.diffusion_coeff:.4f} +/- {self.diffusion_std:.4f} um^2/s\n")
            f.write(f"Hydrodynamic Radius: R_h = {self.compute_hydrodynamic_radius():.2f} nm\n")
            f.write(f"\nExperimental Parameters:\n")
            f.write(f"  Frames: {self.params.n_frames}\n")
            f.write(f"  Frame rate: {self.params.frame_rate} fps\n")
            f.write(f"  Pixel size: {self.params.pixel_size_um} um\n")
            f.write(f"  Magnification: {self.params.magnification}x\n")
        
        print(f"Exported summary to {summary_path}")


def run_ddm_analysis(image_path, pixel_size_um, magnification, frame_rate,
                     roi=None, output_dir=None, max_frames=None):
    """
    Complete DDM analysis pipeline
    Supports both video files and directories of TIF images
    
    Parameters:
    -----------
    image_path : str
        Path to video file OR directory containing TIF sequence
    pixel_size_um : float
        Physical pixel size in micrometers
    magnification : float
        Microscope magnification
    frame_rate : float
        Frame rate in fps (or effective frame rate for TIF sequences)
    roi : tuple (x, y, width, height), optional
        Region of interest
    output_dir : str, optional
        Directory to save results
    max_frames : int, optional
        Maximum frames to load (useful for testing)
    
    Returns:
    --------
    analyzer : DDMAnalyzer
        Analyzer object with all results
    """
    # Create parameters object
    params = DDMParameters(
        pixel_size_um=pixel_size_um,
        magnification=magnification,
        frame_rate=frame_rate,
        n_frames=0  # Will be set when loading images
    )
    
    # Initialize analyzer
    analyzer = DDMAnalyzer(params)
    
    # Run analysis pipeline
    print("="*60)
    print("DDM Analysis - Differential Dynamic Microscopy")
    print("="*60)
    print("\nStep 1: Loading images...")
    analyzer.load_images(image_path, roi=roi, max_frames=max_frames)
    
    print("\nStep 2: Computing difference images...")
    analyzer.compute_difference_images()
    
    print("\nStep 3: Computing structure function...")
    analyzer.compute_structure_function()
    
    print("\nStep 4: Fitting exponential growth...")
    analyzer.fit_exponential_growth()
    
    print("\nStep 5: Computing diffusion coefficient...")
    D, D_std = analyzer.compute_diffusion_coefficient()
    R_h = analyzer.compute_hydrodynamic_radius()
    
    # Save results
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_path = Path(output_dir) / "ddm_analysis.png"
        csv_path = Path(output_dir) / "ddm_results.csv"
        
        print("\nSaving results...")
        analyzer.plot_results(save_path=str(plot_path))
        analyzer.export_results(str(csv_path))
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {output_dir}")
        print(f"  - ddm_analysis.png")
        print(f"  - ddm_results.csv")
        print(f"  - ddm_results_summary.txt")
    
    return analyzer