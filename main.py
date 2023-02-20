# Muhammad Umar Khan
# 400167784 | khanm214

import cv2
import numpy as np

def rgb_to_yuv(rgb_image):
    # Get the red, green, and blue channels of the image
    r = rgb_image[:,:,0]
    g = rgb_image[:,:,1]
    b = rgb_image[:,:,2]

    # Calculate the Y, U, and V channels
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.14713*r - 0.28886*g + 0.436*b
    v = 0.615*r - 0.51498*g - 0.10001*b

    # Combine the Y, U, and V channels into a single image
    yuv_image = np.zeros(rgb_image.shape, dtype=np.uint8)
    yuv_image[:,:,0] = y
    yuv_image[:,:,1] = u
    yuv_image[:,:,2] = v

    return yuv_image

def yuv_to_rgb(yuv_image):
    # Get the Y, U, and V channels of the image
    y = yuv_image[:,:,0]
    u = yuv_image[:,:,1]
    v = yuv_image[:,:,2]

    # Convert YUV to RGB
    r = y + 1.13983*v
    g = y - 0.39465*u - 0.5806*v
    b = y + 2.03211*u

    # Combine the R, G, and B channels into a single image
    rgb_image = np.zeros(yuv_image.shape)
    rgb_image[:,:,0] = r
    rgb_image[:,:,1] = g
    rgb_image[:,:,2] = b

    return rgb_image

def downsize_image(input_file, output_file, factor):
    # Decode the image into a list of RGB tuples
    data = cv2.imread(input_file)
    data = rgb_to_yuv(data)
    print(data[:,:,0])
    print(data[:,:,1])
    print(data[:,:,2])
    
    # Calculate the new size of the image
    new_width = int(len(data[0]) / factor)
    new_height = int(len(data) / factor)

    # Create a new list to store the resized image
    new_pixels = [[(0, 0, 0) for j in range(new_width)] for i in range(new_height)]

    # Copy pixels from the original image to the new image
    for i in range(new_height):
        for j in range(new_width):
            new_pixels[i][j] = data[i * factor][j * factor]

    # Convert the list of tuples to a 3D Numpy array
    downsampled_image = np.array(new_pixels, dtype=np.uint8)

    # Write the binary data to the output file
    cv2.imwrite(output_file, downsampled_image)

def bicubic_interpolation(x, a):
    if x < 0:
        x = -x
    if x <= 1:
        return (a + 2) * x**3 - (a + 3) * x**2 + 1
    elif x < 2:
        return a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a
    else:
        return 0

def bicubic_upsample(input_file, output_file, factor):
    # Decode the image into a list of RGB tuples
    pixels = cv2.imread(input_file)
    
    # Calculate the new size of the image
    height, width = len(pixels), len(pixels[0])
    new_height, new_width = height * factor, width * factor

    # Create a new list to store the upsampled image
    new_pixels = [[(0, 0, 0) for j in range(new_width)] for i in range(new_height)]

    # Set the parameter 'a' for bicubic interpolation
    a = -0.5

    # Upsample the image using bicubic interpolation
    for i in range(new_height):
        for j in range(new_width):
            x, y = i / factor, j / factor
            x0, y0 = int(x), int(y)
            dx, dy = x - x0, y - y0

            # Check if the original pixel indices are within the bounds
            if x0 >= 0 and x0 < height - 1 and y0 >= 0 and y0 < width - 1:
                # Perform bicubic interpolation on the red channel
                red = (
                    min(pixels[x0 - 1][y0 - 1][0] * bicubic_interpolation(dx + 1, a) * bicubic_interpolation(dy + 1, a) +
                    pixels[x0 - 1][y0][0] * bicubic_interpolation(dx + 1, a) * bicubic_interpolation(dy, a) +
                    pixels[x0 - 1][y0 + 1][0] * bicubic_interpolation(dx + 1, a) * bicubic_interpolation(1 - dy, a) +
                    pixels[x0][y0 - 1][0] * bicubic_interpolation(dx, a) * bicubic_interpolation(dy + 1, a) +
                    pixels[x0][y0][0] * bicubic_interpolation(dx, a) * bicubic_interpolation(dy, a) +
                    pixels[x0][y0 + 1][0] * bicubic_interpolation(dx, a) * bicubic_interpolation(1 - dy, a) +
                    pixels[x0 + 1][y0 - 1][0] * bicubic_interpolation(1 - dx, a) * bicubic_interpolation(dy + 1, a) +
                    pixels[x0 + 1][y0][0] * bicubic_interpolation(1 - dx, a) * bicubic_interpolation(dy, a) +
                    pixels[x0 + 1][y0 + 1][0] * bicubic_interpolation(1 - dx, a) * bicubic_interpolation(1 - dy, a), 255)
                )

                # Perform bicubic interpolation on the green channel
                green = (
                    min(pixels[x0 - 1][y0 - 1][1] * bicubic_interpolation(dx + 1, a) * bicubic_interpolation(dy + 1, a) +
                    pixels[x0 - 1][y0][1] * bicubic_interpolation(dx + 1, a) * bicubic_interpolation(dy, a) +
                    pixels[x0 - 1][y0 + 1][1] * bicubic_interpolation(dx + 1, a) * bicubic_interpolation(1 - dy, a) +
                    pixels[x0][y0 - 1][1] * bicubic_interpolation(dx, a) * bicubic_interpolation(dy + 1, a) +
                    pixels[x0][y0][1] * bicubic_interpolation(dx, a) * bicubic_interpolation(dy, a) +
                    pixels[x0][y0 + 1][1] * bicubic_interpolation(dx, a) * bicubic_interpolation(1 - dy, a) +
                    pixels[x0 + 1][y0 - 1][1] * bicubic_interpolation(1 - dx, a) * bicubic_interpolation(dy + 1, a) +
                    pixels[x0 + 1][y0][1] * bicubic_interpolation(1 - dx, a) * bicubic_interpolation(dy, a) +
                    pixels[x0 + 1][y0 + 1][1] * bicubic_interpolation(1 - dx, a) * bicubic_interpolation(1 - dy, a), 255)
                )

                # Perform bicubic interpolation on the blue channel
                blue = (
                    min(pixels[x0 - 1][y0 - 1][2] * bicubic_interpolation(dx + 1, a) * bicubic_interpolation(dy + 1, a) +
                    pixels[x0 - 1][y0][2] * bicubic_interpolation(dx + 1, a) * bicubic_interpolation(dy, a) +
                    pixels[x0 - 1][y0 + 1][2] * bicubic_interpolation(dx + 1, a) * bicubic_interpolation(1 - dy, a) +
                    pixels[x0][y0 - 1][2] * bicubic_interpolation(dx, a) * bicubic_interpolation(dy + 1, a) +
                    pixels[x0][y0][2] * bicubic_interpolation(dx, a) * bicubic_interpolation(dy, a) +
                    pixels[x0][y0 + 1][2] * bicubic_interpolation(dx, a) * bicubic_interpolation(1 - dy, a) +
                    pixels[x0 + 1][y0 - 1][2] * bicubic_interpolation(1 - dx, a) * bicubic_interpolation(dy + 1, a) +
                    pixels[x0 + 1][y0][2] * bicubic_interpolation(1 - dx, a) * bicubic_interpolation(dy, a) +
                    pixels[x0 + 1][y0 + 1][2] * bicubic_interpolation(1 - dx, a) * bicubic_interpolation(1 - dy, a), 255)
                )

            # Combine the red, green, and blue channels to form the final pixel value
            new_pixels[i][j] = (red, green, blue)
    
    
    # Convert the list of tuples to a 3D Numpy array
    upsampled_image = np.array(new_pixels, dtype=np.uint8)
    
    # Convert from YCbCr to RGB
    upsampled_image = yuv_to_rgb(upsampled_image)

    # Save the upsampled image as a new image file
    cv2.imwrite(output_file, upsampled_image)

# Example usage, bicubic upsample takes a long time, downsize is much faster, dont use both at the same time
downsize_image("plant-based-food.jpg", "downsampled_plant-based-food.jpg", 2)
# bicubic_upsample("downsampled_plant-based-food.jpg", "upsampled_plant-based-food.jpg", 4)
