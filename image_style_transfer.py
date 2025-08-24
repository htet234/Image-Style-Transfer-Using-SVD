import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse

def load_image(image_path, size=None):
    """
    Load an image from the specified path and preprocess it.
    
    Args:
        image_path (str): Path to the image file
        size (tuple, optional): Target size (width, height) for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert from BGR to RGB (OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if size is specified
    if size is not None:
        img = cv2.resize(img, size)
    
    return img

def compute_svd(image):
    """
    Compute the SVD for each channel of the image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        tuple: Lists of U, Sigma, and V components for each channel
    """
    # Split the image into channels
    channels = cv2.split(image)
    
    # Initialize lists to store SVD components
    U_list, sigma_list, Vt_list = [], [], []
    
    # Compute SVD for each channel
    for channel in channels:
        U, sigma, Vt = np.linalg.svd(channel, full_matrices=False)
        U_list.append(U)
        sigma_list.append(sigma)
        Vt_list.append(Vt)
    
    return U_list, sigma_list, Vt_list

def low_rank_approximation(U, sigma, Vt, k):
    """
    Create a low-rank approximation of a matrix using the top k singular values.
    
    Args:
        U (numpy.ndarray): U matrix from SVD
        sigma (numpy.ndarray): Singular values
        Vt (numpy.ndarray): V transpose matrix from SVD
        k (int): Number of singular values to use
        
    Returns:
        numpy.ndarray: Low-rank approximation of the original matrix
    """
    # Ensure k is not larger than the number of singular values
    k = min(k, len(sigma))
    
    # Create the low-rank approximation
    return U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]

def blend_images(content_image, style_image, content_k, style_k, alpha=0.5):
    """
    Blend two images using their SVD components.
    
    Args:
        content_image (numpy.ndarray): Content image
        style_image (numpy.ndarray): Style image
        content_k (int): Number of singular values to use from content image
        style_k (int): Number of singular values to use from style image
        alpha (float): Blending factor (0.0 to 1.0)
        
    Returns:
        numpy.ndarray: Blended image
    """
    # Ensure images have the same size
    if content_image.shape != style_image.shape:
        style_image = cv2.resize(style_image, (content_image.shape[1], content_image.shape[0]))
    
    # Compute SVD for both images
    content_U, content_sigma, content_Vt = compute_svd(content_image)
    style_U, style_sigma, style_Vt = compute_svd(style_image)
    
    # Initialize list to store blended channels
    blended_channels = []
    
    # Process each channel
    for i in range(len(content_U)):
        # Create low-rank approximations
        content_approx = low_rank_approximation(
            content_U[i], content_sigma[i], content_Vt[i], content_k
        )
        style_approx = low_rank_approximation(
            style_U[i], style_sigma[i], style_Vt[i], style_k
        )
        
        # Blend the approximations
        blended = alpha * content_approx + (1 - alpha) * style_approx
        
        # Clip values to valid range [0, 255]
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        blended_channels.append(blended)
    
    # Merge channels back into an image
    blended_image = cv2.merge(blended_channels)
    
    return blended_image

def visualize_results(content_image, style_image, blended_image, title="Image Style Transfer using SVD"):
    """
    Visualize the original images and the blended result.
    
    Args:
        content_image (numpy.ndarray): Content image
        style_image (numpy.ndarray): Style image
        blended_image (numpy.ndarray): Blended image
        title (str): Title for the plot
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(content_image)
    plt.title("Content Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(style_image)
    plt.title("Style Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(blended_image)
    plt.title("Blended Image")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def save_image(image, output_path):
    """
    Save an image to the specified path.
    
    Args:
        image (numpy.ndarray): Image to save
        output_path (str): Path to save the image
    """
    # Convert from RGB to BGR (OpenCV expects BGR format for saving)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save the image
    cv2.imwrite(output_path, image_bgr)
    print(f"Image saved to {output_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Image Style Transfer using SVD")
    parser.add_argument("--content", required=True, help="Path to content image")
    parser.add_argument("--style", required=True, help="Path to style image")
    parser.add_argument("--output", default="output.jpg", help="Path to save the output image")
    parser.add_argument("--content-k", type=int, default=50, help="Number of singular values to use from content image")
    parser.add_argument("--style-k", type=int, default=30, help="Number of singular values to use from style image")
    parser.add_argument("--alpha", type=float, default=0.7, help="Blending factor (0.0 to 1.0)")
    parser.add_argument("--size", type=int, default=400, help="Size to resize images (square)")
    args = parser.parse_args()
    
    # Load images
    print("Loading images...")
    content_image = load_image(args.content, size=(args.size, args.size))
    style_image = load_image(args.style, size=(args.size, args.size))
    
    # Blend images
    print("Blending images using SVD...")
    blended_image = blend_images(
        content_image, style_image, args.content_k, args.style_k, args.alpha
    )
    
    # Visualize results
    visualize_results(content_image, style_image, blended_image)
    
    # Save the blended image
    save_image(blended_image, args.output)

if __name__ == "__main__":
    main()