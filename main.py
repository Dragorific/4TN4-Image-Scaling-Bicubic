import numpy as np
import cv2

def downsample_image(image, downsample_factor):
    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply a Gaussian filter to the image
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Downsample the image
    rows, cols, _ = image.shape
    downsampled_image = cv2.resize(image, (cols//downsample_factor, rows//downsample_factor), interpolation=cv2.INTER_AREA)
    
    return downsampled_image

def upsample_image(image, upsample_factor):
    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Upsample the image using bicubic interpolation
    rows, cols, _ = image.shape
    new_rows = int(rows * upsample_factor)
    new_cols = int(cols * upsample_factor)
    upsampled_image = cv2.resize(image, (new_cols, new_rows), interpolation=cv2.INTER_CUBIC)
    
    return upsampled_image

# Load a base image
image = cv2.imread('hotdawg.jpg')

# Downsample the image by a factor of 4
downsampled_image = downsample_image(image, 4)

# Save the downsampled image
cv2.imwrite('downsampled_hotdawg.jpg', cv2.cvtColor(downsampled_image, cv2.COLOR_RGB2BGR))


# Load the downsampled image
image = cv2.imread('downsampled_hotdawg.jpg')

# Upsample the image by a factor of 4
upsampled_image = upsample_image(image, 4)

# Save the upsampled image
cv2.imwrite('upsampled_image.jpg', cv2.cvtColor(upsampled_image, cv2.COLOR_RGB2BGR))
