import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_style_transfer import load_image, blend_images, save_image
import tempfile

class ImageStyleTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Style Transfer using SVD")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Initialize variables
        self.content_image_path = tk.StringVar()
        self.style_image_path = tk.StringVar()
        self.output_path = tk.StringVar(value="output.jpg")
        self.content_k = tk.IntVar(value=50)
        self.style_k = tk.IntVar(value=30)
        self.alpha = tk.DoubleVar(value=0.7)
        self.size = tk.IntVar(value=400)
        
        self.content_image = None
        self.style_image = None
        self.blended_image = None
        
        # === Create welcome frame first ===
        self.create_welcome_frame()
        
        # Main UI frame (hidden initially)
        self.main_frame = ttk.Frame(self.root, padding="10")

    def create_welcome_frame(self):
        self.welcome_frame = tk.Frame(self.root)
        self.welcome_frame.pack(fill=tk.BOTH, expand=True)
        
        welcome_label = tk.Label(
            self.welcome_frame,
            text="Welcome to Image Style Transfer!\nBlend images using SVD.",
            font=("Helvetica", 18, "bold"),
            justify="center",
            pady=20
        )
        welcome_label.pack(pady=60)
            
        start_button = tk.Button(
            self.welcome_frame,
            text="Start",
            font=("Helvetica", 14),
            command=self.show_main_ui
        )
        start_button.pack(pady=20)
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ✅ Welcome message at top
        welcome_frame = ttk.Frame(main_frame, padding="10")
        welcome_frame.pack(fill=tk.X, pady=(0, 10))
        
        welcome_msg = (
            "Welcome to Image Style Transfer!\n"
            "This app uses SVD to blend a content image with a style image. "
            "Adjust parameters like Content K, Style K, and Alpha to fine-tune the result."
        )
        
        ttk.Label(
            welcome_frame,
            text=welcome_msg,
            wraplength=800,
            justify="center",
            font=("Helvetica", 12)
        ).pack()
        
        # ✅ Create main content layout
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(content_frame, text="Controls", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Image selection
        ttk.Label(control_frame, text="Content Image:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.content_image_path, width=30).grid(row=0, column=1, pady=5)
        ttk.Button(control_frame, text="Browse...", command=self.browse_content_image).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Style Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.style_image_path, width=30).grid(row=1, column=1, pady=5)
        ttk.Button(control_frame, text="Browse...", command=self.browse_style_image).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Output Path:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.output_path, width=30).grid(row=2, column=1, pady=5)
        ttk.Button(control_frame, text="Browse...", command=self.browse_output_path).grid(row=2, column=2, padx=5, pady=5)
        
        # Parameters
        param_frame = ttk.LabelFrame(control_frame, text="Parameters", padding="10")
        param_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        ttk.Label(param_frame, text="Content K:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Scale(param_frame, from_=1, to=200, variable=self.content_k, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky=tk.EW, pady=5)
        ttk.Label(param_frame, textvariable=self.content_k).grid(row=0, column=2, padx=5)
        
        ttk.Label(param_frame, text="Style K:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Scale(param_frame, from_=1, to=200, variable=self.style_k, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=tk.EW, pady=5)
        ttk.Label(param_frame, textvariable=self.style_k).grid(row=1, column=2, padx=5)
        
        ttk.Label(param_frame, text="Alpha:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Scale(param_frame, from_=0.0, to=1.0, variable=self.alpha, orient=tk.HORIZONTAL).grid(row=2, column=1, sticky=tk.EW, pady=5)
        ttk.Label(param_frame, textvariable=self.alpha).grid(row=2, column=2, padx=5)
        
        ttk.Label(param_frame, text="Image Size:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Scale(param_frame, from_=100, to=800, variable=self.size, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky=tk.EW, pady=5)
        ttk.Label(param_frame, textvariable=self.size).grid(row=3, column=2, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Blend Images", command=self.blend_images_action).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        # Right panel for image display
        display_frame = ttk.LabelFrame(content_frame, text="Image Preview", padding="10")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure for image display
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize the plot
        self.init_plot()
    
    def show_main_ui(self):
        self.welcome_frame.pack_forget()  # Hide the welcome message
        self.create_widgets()             # Show main image blending UI

    def init_plot(self):
        self.fig.clear()
        self.fig.suptitle("Image Style Transfer using SVD")
        
        # Create empty subplots
        self.ax1 = self.fig.add_subplot(131)
        self.ax1.set_title("Content Image")
        self.ax1.axis('off')
        
        self.ax2 = self.fig.add_subplot(132)
        self.ax2.set_title("Style Image")
        self.ax2.axis('off')
        
        self.ax3 = self.fig.add_subplot(133)
        self.ax3.set_title("Blended Image")
        self.ax3.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def browse_content_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Content Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.content_image_path.set(file_path)
            try:
                self.content_image = load_image(file_path, size=(self.size.get(), self.size.get()))
                self.update_plot()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def browse_style_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Style Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.style_image_path.set(file_path)
            try:
                self.style_image = load_image(file_path, size=(self.size.get(), self.size.get()))
                self.update_plot()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def browse_output_path(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Output Image As",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.output_path.set(file_path)
    
    def blend_images_action(self):
        if self.content_image is None or self.style_image is None:
            messagebox.showwarning("Warning", "Please load both content and style images first.")
            return
        
        try:
            # Show loading cursor
            self.root.config(cursor="wait")
            self.root.update()
            
            # Resize images if size has changed
            size = (self.size.get(), self.size.get())
            content = cv2.resize(self.content_image, size) if self.content_image.shape[:2] != size else self.content_image
            style = cv2.resize(self.style_image, size) if self.style_image.shape[:2] != size else self.style_image
            
            # Blend images
            self.blended_image = blend_images(
                content, style,
                self.content_k.get(),
                self.style_k.get(),
                self.alpha.get()
            )
            
            # Update plot
            self.update_plot()
            
            # Restore cursor
            self.root.config(cursor="")
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Error", f"Failed to blend images: {str(e)}")
    
    def update_plot(self):
        self.fig.clear()
        self.fig.suptitle("Image Style Transfer using SVD")
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(131)
        self.ax1.set_title("Content Image")
        if self.content_image is not None:
            self.ax1.imshow(self.content_image)
        self.ax1.axis('off')
        
        self.ax2 = self.fig.add_subplot(132)
        self.ax2.set_title("Style Image")
        if self.style_image is not None:
            self.ax2.imshow(self.style_image)
        self.ax2.axis('off')
        
        self.ax3 = self.fig.add_subplot(133)
        self.ax3.set_title("Blended Image")
        if self.blended_image is not None:
            self.ax3.imshow(self.blended_image)
        self.ax3.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_result(self):
        if self.blended_image is None:
            messagebox.showwarning("Warning", "No blended image to save. Please blend images first.")
            return
        
        output_path = self.output_path.get()
        if not output_path:
            output_path = filedialog.asksaveasfilename(
                title="Save Output Image As",
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
            )
            if not output_path:
                return
            self.output_path.set(output_path)
        
        try:
            save_image(self.blended_image, output_path)
            messagebox.showinfo("Success", f"Image saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
            
    def load_sample_images(self):
        """Generate and load sample content and style images"""
        try:
            # Show loading cursor
            self.root.config(cursor="wait")
            self.root.update()
            
            # Create a temporary directory to store sample images
            temp_dir = tempfile.mkdtemp()
            
            # Generate sample content image (gradient)
            size = (self.size.get(), self.size.get())
            x = np.linspace(0, 1, size[0])
            y = np.linspace(0, 1, size[1])
            xx, yy = np.meshgrid(x, y)
            
            content = np.zeros((*size, 3), dtype=np.uint8)
            content[:, :, 0] = (xx * 255).astype(np.uint8)  # Red channel
            content[:, :, 1] = (yy * 255).astype(np.uint8)  # Green channel
            content[:, :, 2] = ((1 - xx) * (1 - yy) * 255).astype(np.uint8)  # Blue channel
            
            # Generate sample style image (pattern)
            style = np.zeros((*size, 3), dtype=np.uint8)
            freq = 20
            style[:, :, 0] = ((np.sin(xx * freq) + 1) * 127.5).astype(np.uint8)  # Red channel
            style[:, :, 1] = ((np.cos(yy * freq) + 1) * 127.5).astype(np.uint8)  # Green channel
            style[:, :, 2] = ((np.sin(xx * freq + yy * freq) + 1) * 127.5).astype(np.uint8)  # Blue channel
            
            # Save sample images to temporary files
            content_path = os.path.join(temp_dir, "sample_content.jpg")
            style_path = os.path.join(temp_dir, "sample_style.jpg")
            
            # Convert from RGB to BGR for OpenCV
            cv2.imwrite(content_path, cv2.cvtColor(content, cv2.COLOR_RGB2BGR))
            cv2.imwrite(style_path, cv2.cvtColor(style, cv2.COLOR_RGB2BGR))
            
            # Set paths and load images
            self.content_image_path.set(content_path)
            self.style_image_path.set(style_path)
            self.content_image = content
            self.style_image = style
            
            # Update the plot
            self.update_plot()
            
            # Show a message
            messagebox.showinfo("Sample Images", "Sample content and style images have been loaded. Click 'Blend Images' to see the result.")
            
            # Restore cursor
            self.root.config(cursor="")
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Error", f"Failed to generate sample images: {str(e)}")
    
    def show_welcome_prompt(self):
        """Show a welcome dialog that asks users what images they want to add"""
        welcome_dialog = tk.Toplevel(self.root)
        welcome_dialog.title("Welcome")
        welcome_dialog.transient(self.root)
        welcome_dialog.grab_set()
        
        # Remove fixed geometry - let it auto-size based on content
        welcome_dialog.resizable(True, True)

        # Create a main container frame with scrollbars
        container = ttk.Frame(welcome_dialog)
        container.pack(fill=tk.BOTH, expand=True)

        # Add a canvas with scrollbars for content
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Frame inside canvas for content
        frame = ttk.Frame(canvas, padding="20")
        canvas.create_window((0, 0), window=frame, anchor="nw")

        # Welcome header
        ttk.Label(frame, text="Image Style Transfer", font=("Helvetica", 16, "bold")).pack(pady=10)
        ttk.Label(frame, text="Blend images using SVD.", wraplength=400).pack(pady=5)

        # Image description frame
        desc_frame = ttk.LabelFrame(frame, text="Select images to blend", padding="10")
        desc_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Content image description
        content_frame = ttk.Frame(desc_frame)
        content_frame.pack(fill=tk.X, pady=5)
        ttk.Label(content_frame, text="Content:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
        content_desc = ttk.Label(content_frame, text="Photo with structure (selfie, landscape)", wraplength=300)
        content_desc.pack(side=tk.LEFT, padx=5)

        # Style image description
        style_frame = ttk.Frame(desc_frame)
        style_frame.pack(fill=tk.X, pady=5)
        ttk.Label(style_frame, text="Style:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)
        style_desc = ttk.Label(style_frame, text="Artistic image (painting, pattern)", wraplength=300)
        style_desc.pack(side=tk.LEFT, padx=5)

        # Additional information
        additional_info_frame = ttk.LabelFrame(frame, text="How Image Style Transfer Works", padding="10")
        additional_info_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        info_label = ttk.Label(
            additional_info_frame,
            text=("This application uses Singular Value Decomposition (SVD) to blend images. "
                "SVD decomposes an image into components representing structure and texture. "
                "By combining the structure from one image with the texture from another, "
                "we can create artistic effects similar to neural style transfer but using linear algebra instead of neural networks.\n\n"
                "Adjust the 'Content K' and 'Style K' parameters to control how much detail from each image is preserved. "
                "The 'Alpha' parameter controls the blending ratio between content and style."),
            wraplength=500,
            justify="left"
        )
        info_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Buttons frame
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=15)

        ttk.Button(
            button_frame, 
            text="Load Content Image", 
            command=lambda: [self.browse_content_image(), welcome_dialog.destroy()]
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            button_frame, 
            text="Load Style Image", 
            command=lambda: [self.browse_style_image(), welcome_dialog.destroy()]
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            button_frame, 
            text="Use Sample Images", 
            command=lambda: [self.load_sample_images(), welcome_dialog.destroy()]
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            button_frame, 
            text="Skip", 
            command=welcome_dialog.destroy
        ).pack(side=tk.LEFT, padx=10)

        # Update scroll region dynamically
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        frame.bind("<Configure>", on_configure)

        # Center the dialog after content is loaded
        welcome_dialog.update_idletasks()
        width = welcome_dialog.winfo_reqwidth()
        height = welcome_dialog.winfo_reqheight()
        x = (welcome_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (welcome_dialog.winfo_screenheight() // 2) - (height // 2)
        welcome_dialog.geometry(f"+{x}+{y}")

        # Wait until closed
        self.root.wait_window(welcome_dialog)

def main():
    root = tk.Tk()
    root.title("Image Style Transfer using SVD")

    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Define size for main window
    window_width = 1000
    window_height = 700

    # Calculate center position
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)

    # Apply geometry
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    app = ImageStyleTransferApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()