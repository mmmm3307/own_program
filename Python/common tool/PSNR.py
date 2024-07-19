import cv2
import numpy as np

def calculate_psnr(img1, img2):
    # Ensure the images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # Convert images to float32 for precise computation
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Compute Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    
    # If MSE is zero, the PSNR is infinite (images are identical)
    if mse == 0:
        return float('inf')
    
    # Compute PSNR
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

# Example usage
if __name__ == "__main__":
    # Load images
    img1 = cv2.imread('image1.png')
    img2 = cv2.imread('image2.png')
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("One of the images was not found.")
    
    # Convert images to grayscale (optional, depending on the use case)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate PSNR
    psnr_value = calculate_psnr(img1_gray, img2_gray)
    print(f"PSNR: {psnr_value} dB")
