import cv2
import numpy as np

def downsize_image(input_file, output_file, factor):
    # Decode the image into a list of RGB tuples
    data = cv2.imread(input_file)
    
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

    # Save the upsampled image as a new image file
    cv2.imwrite(output_file, upsampled_image)

# Example usage, bicubic upsample takes a long time, downsize is much faster, dont use both at the same time
downsize_image("hotdawg.jpg", "downsampled_hotdawg.jpg", 4)
# bicubic_upsample("downsampled_hotdawg.jpg", "upsampled_hotdawg.jpg", 4)
