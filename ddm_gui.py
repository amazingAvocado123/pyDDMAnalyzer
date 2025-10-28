"""
DDM Analyzer - Simple GUI Version
Easy-to-use interface for Differential Dynamic Microscopy analysis
Supports video files AND sequential TIF directories
"""

import sys
import io

# Force UTF-8 encoding for console output (Windows compatibility)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path

# Import the DDM analyzer
from ddm_analyzer import run_ddm_analysis


class DDMGUI:
    @staticmethod
    def sanitize_text(text):
        """Remove any non-ASCII characters that might cause encoding issues"""
        try:
            # Try to encode as ASCII, replacing bad characters
            return str(text).encode('ascii', errors='replace').decode('ascii')
        except:
            return "Error message contains special characters"
    
    def __init__(self, root):
        self.root = root
        self.root.title("DDM Analyzer - Differential Dynamic Microscopy")
        self.root.geometry("700x900")
        
        # Variables
        self.image_path = tk.StringVar()
        self.input_type = tk.StringVar(value="tif_directory")  # "tif_directory" or "video"
        self.output_dir = tk.StringVar(value="results")
        self.pixel_size = tk.DoubleVar(value=6.5)
        self.magnification = tk.DoubleVar(value=50.0)
        self.frame_rate = tk.DoubleVar(value=125.0)
        self.use_roi = tk.BooleanVar(value=False)
        self.roi_x = tk.IntVar(value=0)
        self.roi_y = tk.IntVar(value=0)
        self.roi_width = tk.IntVar(value=512)
        self.roi_height = tk.IntVar(value=512)
        self.max_frames = tk.IntVar(value=0)
        self.use_max_frames = tk.BooleanVar(value=False)
        
        self.analyzer = None
        self.is_running = False
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # ========================================
        # Title
        # ========================================
        title_label = ttk.Label(main_frame, text="DDM Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        subtitle_label = ttk.Label(main_frame, 
                                   text="Measure particle diffusion from microscope images",
                                   font=('Arial', 9, 'italic'))
        subtitle_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1
        
        # ========================================
        # Input Type Selection
        # ========================================
        ttk.Label(main_frame, text="1. Select Input Type:", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, 
                                                  columnspan=3, sticky=tk.W, pady=(10, 5))
        row += 1
        
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        ttk.Radiobutton(input_frame, text="Directory of TIF files (sequential)", 
                       variable=self.input_type, value="tif_directory",
                       command=self.update_browse_button).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(input_frame, text="Video file (.avi, .mp4, etc.)", 
                       variable=self.input_type, value="video",
                       command=self.update_browse_button).pack(side=tk.LEFT)
        
        # ========================================
        # Image/Video Path Selection
        # ========================================
        ttk.Label(main_frame, text="2. Select Images:", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, 
                                                  columnspan=3, sticky=tk.W, pady=(10, 5))
        row += 1
        
        ttk.Entry(main_frame, textvariable=self.image_path, width=50).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 5))
        self.browse_button = ttk.Button(main_frame, text="Browse Directory...", 
                                       command=self.browse_images)
        self.browse_button.grid(row=row, column=2)
        row += 1
        
        # Max frames option
        max_frames_frame = ttk.Frame(main_frame)
        max_frames_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        ttk.Checkbutton(max_frames_frame, text="Limit number of frames (for testing):", 
                       variable=self.use_max_frames,
                       command=self.toggle_max_frames).pack(side=tk.LEFT)
        self.max_frames_entry = ttk.Entry(max_frames_frame, textvariable=self.max_frames, 
                                          width=10, state='disabled')
        self.max_frames_entry.pack(side=tk.LEFT, padx=5)
        
        # ========================================
        # Microscope Parameters
        # ========================================
        ttk.Separator(main_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        row += 1
        
        ttk.Label(main_frame, text="3. Microscope Parameters:", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, 
                                                  columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Pixel size
        ttk.Label(main_frame, text="Camera Pixel Size (um):").grid(
            row=row, column=0, sticky=tk.W, pady=5)
        pixel_entry = ttk.Entry(main_frame, textvariable=self.pixel_size, width=15)
        pixel_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(main_frame, text="(typically 3-7 um)", 
                 font=('Arial', 8, 'italic'), foreground='gray').grid(
            row=row, column=2, sticky=tk.W, pady=5)
        row += 1
        
        # Magnification
        ttk.Label(main_frame, text="Objective Magnification:").grid(
            row=row, column=0, sticky=tk.W, pady=5)
        mag_entry = ttk.Entry(main_frame, textvariable=self.magnification, width=15)
        mag_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(main_frame, text="(e.g., 10, 20, 50, 100)", 
                 font=('Arial', 8, 'italic'), foreground='gray').grid(
            row=row, column=2, sticky=tk.W, pady=5)
        row += 1
        
        # Frame rate
        ttk.Label(main_frame, text="Frame Rate (fps):").grid(
            row=row, column=0, sticky=tk.W, pady=5)
        fps_entry = ttk.Entry(main_frame, textvariable=self.frame_rate, width=15)
        fps_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(main_frame, text="(time between images)", 
                 font=('Arial', 8, 'italic'), foreground='gray').grid(
            row=row, column=2, sticky=tk.W, pady=5)
        row += 1
        
        # Calculated effective pixel size
        self.effective_pixel_label = ttk.Label(main_frame, text="", 
                                              foreground='blue')
        self.effective_pixel_label.grid(row=row, column=0, columnspan=3, 
                                       sticky=tk.W, pady=10)
        self.update_effective_pixel()
        row += 1
        
        # ========================================
        # Region of Interest (Optional)
        # ========================================
        ttk.Separator(main_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        row += 1
        
        ttk.Label(main_frame, text="4. Region of Interest (Optional):", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, 
                                                  columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1
        
        roi_check = ttk.Checkbutton(main_frame, text="Use specific region (ROI)", 
                                     variable=self.use_roi,
                                     command=self.toggle_roi)
        roi_check.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        # ROI frame (initially disabled)
        self.roi_frame = ttk.Frame(main_frame)
        self.roi_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # ROI inputs
        roi_labels = ["X:", "Y:", "Width:", "Height:"]
        roi_vars = [self.roi_x, self.roi_y, self.roi_width, self.roi_height]
        
        for i, (label, var) in enumerate(zip(roi_labels, roi_vars)):
            ttk.Label(self.roi_frame, text=label).grid(row=i//2, column=(i%2)*2, 
                                                       sticky=tk.W, padx=5, pady=2)
            entry = ttk.Entry(self.roi_frame, textvariable=var, width=10)
            entry.grid(row=i//2, column=(i%2)*2+1, sticky=tk.W, padx=5, pady=2)
            entry.config(state='disabled')
        
        # ========================================
        # Output Directory
        # ========================================
        ttk.Separator(main_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        row += 1
        
        ttk.Label(main_frame, text="5. Output Directory:", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, 
                                                  columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1
        
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(main_frame, text="Browse...", 
                  command=self.browse_output).grid(row=row, column=2)
        row += 1
        
        # ========================================
        # Run Button
        # ========================================
        ttk.Separator(main_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        row += 1
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        self.run_button = ttk.Button(button_frame, text="> Run Analysis", 
                                     command=self.run_analysis,
                                     style='Accent.TButton')
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="[Stop] Stop", 
                                      command=self.stop_analysis,
                                      state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear Log", 
                  command=self.clear_log).pack(side=tk.LEFT, padx=5)
        
        # ========================================
        # Progress Bar
        # ========================================
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=3, 
                          sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # ========================================
        # Status/Log Display
        # ========================================
        ttk.Label(main_frame, text="Status Log:", 
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, 
                                                  columnspan=3, sticky=tk.W, pady=(10, 5))
        row += 1
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=12, 
                                                  wrap=tk.WORD,
                                                  font=('Courier', 9))
        self.log_text.grid(row=row, column=0, columnspan=3, 
                          sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(row, weight=1)
        row += 1
        
        # ========================================
        # Results Frame
        # ========================================
        self.results_frame = ttk.LabelFrame(main_frame, text="Results", 
                                           padding="10")
        self.results_frame.grid(row=row, column=0, columnspan=3, 
                               sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        self.results_label = ttk.Label(self.results_frame, 
                                      text="Run analysis to see results...",
                                      font=('Arial', 9))
        self.results_label.pack()
        
        # Initial log message
        self.log("Welcome to DDM Analyzer!")
        self.log("1. Select input type (TIF directory or video)")
        self.log("2. Browse and select your images")
        self.log("3. Enter microscope parameters")
        self.log("4. Click 'Run Analysis'\n")
        
        # Bind parameter changes
        self.pixel_size.trace('w', lambda *args: self.update_effective_pixel())
        self.magnification.trace('w', lambda *args: self.update_effective_pixel())
    
    def update_browse_button(self):
        """Update browse button text based on input type"""
        if self.input_type.get() == "tif_directory":
            self.browse_button.config(text="Browse Directory...")
        else:
            self.browse_button.config(text="Browse Video...")
    
    def toggle_max_frames(self):
        """Enable/disable max frames entry"""
        state = 'normal' if self.use_max_frames.get() else 'disabled'
        self.max_frames_entry.config(state=state)
    
    def update_effective_pixel(self):
        """Update the calculated effective pixel size display"""
        try:
            pixel = self.pixel_size.get()
            mag = self.magnification.get()
            effective = pixel / mag
            self.effective_pixel_label.config(
                text=f"=> Effective pixel size in sample: {effective:.3f} um/pixel")
        except:
            self.effective_pixel_label.config(text="")
    
    def toggle_roi(self):
        """Enable/disable ROI input fields"""
        state = 'normal' if self.use_roi.get() else 'disabled'
        for widget in self.roi_frame.winfo_children():
            if isinstance(widget, ttk.Entry):
                widget.config(state=state)
    
    def browse_images(self):
        """Open dialog to select directory or video"""
        if self.input_type.get() == "tif_directory":
            # Browse for directory
            directory = filedialog.askdirectory(title="Select TIF Directory")
            if directory:
                self.image_path.set(directory)
                # Count TIF files
                path = Path(directory)
                tif_count = len(list(path.glob("*.tif"))) + len(list(path.glob("*.tiff")))
                self.log(f"Selected directory: {Path(directory).name}")
                self.log(f"Found {tif_count} TIF files")
        else:
            # Browse for video file
            filename = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.avi *.mp4 *.mov *.mkv"),
                    ("AVI files", "*.avi"),
                    ("MP4 files", "*.mp4"),
                    ("All files", "*.*")
                ]
            )
            if filename:
                self.image_path.set(filename)
                self.log(f"Selected video: {Path(filename).name}")
    
    def browse_output(self):
        """Open dialog to select output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            self.log(f"Output directory: {directory}")
    
    def log(self, message):
        """Add message to log display with UTF-8 error handling"""
        try:
            self.log_text.insert(tk.END, message + "\n")
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fallback: replace problematic characters
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            self.log_text.insert(tk.END, safe_message + "\n")
        self.log_text.see(tk.END)
        self.log_text.update()
    
    def clear_log(self):
        """Clear the log display"""
        self.log_text.delete(1.0, tk.END)
    
    def validate_inputs(self):
        """Check if all required inputs are valid"""
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select images/video")
            return False
        
        path_obj = Path(self.image_path.get())
        if self.input_type.get() == "tif_directory":
            if not path_obj.is_dir():
                messagebox.showerror("Error", "Selected path is not a directory")
                return False
            tif_count = len(list(path_obj.glob("*.tif"))) + len(list(path_obj.glob("*.tiff")))
            if tif_count == 0:
                messagebox.showerror("Error", "No TIF files found in directory")
                return False
        else:
            if not path_obj.is_file():
                messagebox.showerror("Error", "Video file does not exist")
                return False
        
        try:
            pixel = self.pixel_size.get()
            if pixel <= 0:
                raise ValueError("Pixel size must be positive")
        except:
            messagebox.showerror("Error", "Invalid pixel size")
            return False
        
        try:
            mag = self.magnification.get()
            if mag <= 0:
                raise ValueError("Magnification must be positive")
        except:
            messagebox.showerror("Error", "Invalid magnification")
            return False
        
        try:
            fps = self.frame_rate.get()
            if fps <= 0:
                raise ValueError("Frame rate must be positive")
        except:
            messagebox.showerror("Error", "Invalid frame rate")
            return False
        
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory")
            return False
        
        return True
    
    def run_analysis(self):
        """Run DDM analysis in a separate thread"""
        if not self.validate_inputs():
            return
        
        if self.is_running:
            messagebox.showwarning("Warning", "Analysis is already running")
            return
        
        # Disable controls
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress.start()
        self.is_running = True
        
        # Clear previous results
        self.results_label.config(text="Analysis in progress...")
        
        # Run in separate thread
        thread = threading.Thread(target=self._run_analysis_thread, daemon=True)
        thread.start()
    
    def _run_analysis_thread(self):
        """Actual analysis code running in separate thread"""
        try:
            self.log("\n" + "="*50)
            self.log("Starting DDM Analysis...")
            self.log("="*50)
            
            # Get parameters
            image_path = self.image_path.get()
            pixel_size = self.pixel_size.get()
            magnification = self.magnification.get()
            frame_rate = self.frame_rate.get()
            output_dir = self.output_dir.get()
            
            # Get ROI if enabled
            roi = None
            if self.use_roi.get():
                roi = (self.roi_x.get(), self.roi_y.get(), 
                      self.roi_width.get(), self.roi_height.get())
                self.log(f"Using ROI: {roi}")
            
            # Get max frames if enabled
            max_frames = None
            if self.use_max_frames.get():
                max_frames = self.max_frames.get()
                self.log(f"Limiting to {max_frames} frames")
            
            # Redirect stdout to log
            import sys
            old_stdout = sys.stdout
            sys.stdout = LogRedirector(self.log)
            
            try:
                # Run analysis
                self.analyzer = run_ddm_analysis(
                    image_path=image_path,
                    pixel_size_um=pixel_size,
                    magnification=magnification,
                    frame_rate=frame_rate,
                    roi=roi,
                    output_dir=output_dir,
                    max_frames=max_frames
                )
                
                # Display results
                D = self.analyzer.diffusion_coeff
                D_std = self.analyzer.diffusion_std
                R_h = self.analyzer.compute_hydrodynamic_radius()
                
                results_text = f"""
Diffusion Coefficient:
  D = {D:.4f} +/- {D_std:.4f} um^2/s

Hydrodynamic Radius:
  R_h = {R_h:.2f} nm

Files saved to: {output_dir}
  - ddm_analysis.png
  - ddm_results.csv
  - ddm_results_summary.txt
                """
                
                # Update results display on main thread
                self.root.after(0, lambda: self.results_label.config(
                    text=results_text, font=('Courier', 9), foreground='green'))
                
                self.log("\n[OK] Success! Check the output directory for detailed results.")
                
                # Ask if user wants to open results folder
                self.root.after(0, lambda: self._ask_open_folder(output_dir))
                
            finally:
                sys.stdout = old_stdout
            
        except Exception as e:
            error_msg = self.sanitize_text(str(e))
            self.log(f"\n[ERROR] {error_msg}")
            import traceback
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", 
                                                            f"An error occurred:\n{error_msg}"))
            self.root.after(0, lambda: self.results_label.config(
                text="Analysis failed. See log for details.", foreground='red'))
        
        finally:
            # Re-enable controls
            self.root.after(0, self._analysis_complete)
    
    def _ask_open_folder(self, folder_path):
        """Ask user if they want to open the results folder"""
        if messagebox.askyesno("Analysis Complete", 
                              "Would you like to open the results folder?"):
            import os
            import platform
            
            folder_path = str(Path(folder_path).absolute())
            
            if platform.system() == "Windows":
                os.startfile(folder_path)
            elif platform.system() == "Darwin":  # macOS
                os.system(f'open "{folder_path}"')
            else:  # Linux
                os.system(f'xdg-open "{folder_path}"')
    
    def _analysis_complete(self):
        """Reset UI after analysis completes"""
        self.is_running = False
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress.stop()
    
    def stop_analysis(self):
        """Stop the current analysis"""
        messagebox.showinfo("Stop", 
                           "Analysis will stop after current step completes.")
        self.is_running = False


class LogRedirector:
    """Redirect print statements to GUI log"""
    def __init__(self, log_func):
        self.log_func = log_func
    
    def write(self, message):
        if message.strip():
            # Ensure proper encoding handling
            try:
                self.log_func(message.rstrip())
            except UnicodeEncodeError:
                # Fallback: replace problematic characters
                self.log_func(message.rstrip().encode('utf-8', errors='replace').decode('utf-8'))
    
    def flush(self):
        pass


def main():
    """Launch the GUI application"""
    root = tk.Tk()
    
    # Apply a style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = DDMGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()