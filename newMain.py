import cv2
import numpy as np

# Transform colour space from RGB to Y, U and V, with seperate channels for each, into a greyscale
def split_color_channels(image_path):
    # Load the image using PIL
    image = cv2.imread(image_path)
    # Get image size
    width, height = len(image[0]), len(image)

    # Create numpy arrays for each grayscale channel
    Yy = np.zeros((height, width), dtype=np.uint8)
    Cb = np.zeros((height, width), dtype=np.uint8)
    Cr = np.zeros((height, width), dtype=np.uint8)

    # Loop through the pixels of the image and compute the channel values
    for x in range(height):
        for y in range(width):
            # Get the color channels for the current pixel
            b, g, r = image[x, y]

            # Compute the grayscale values for the current pixel
            Yy[x, y] = int(0.299*r + 0.587*g + 0.114*b)
            Cb[x, y] = int(-0.14713*r - 0.28886*g + 0.436*b)
            Cr[x, y] = int(0.615*r - 0.51498*g - 0.10001*b)

    # Return the channel arrays
    return Yy, Cb, Cr

def combine_color_channels(y, u, v, height, width):
    new_pixels = [[(0, 0, 0) for j in range(width)] for i in range(height)]
    # Copy pixels from the channels to the new image
    for i in range(height):
        for j in range(width):
            Yy = y[i][j]
            Uu = u[i][j]
            Vv = v[i][j]
            r = max(min(int(Yy + 1.13983*Vv), 255), 0)
            g = max(min(int(Yy - 0.39465*Uu - 0.5806*Vv), 255), 0)
            b = max(min(int(Yy + 2.03211*Uu), 255), 0)
            new_pixels[i][j] = (b, g, r)

        print("Completed copying row " + str(i))
    
    new_image = np.array(new_pixels, dtype=np.uint8)
    return new_image

def downsample_channel(channel, factor):
    # Get channel size
    width, height = len(channel[0]), len(channel)

    # Compute the new size of the downsampled channel
    new_height = int(height/factor)+1
    new_width = int(width/factor)+1

    # Create a new numpy array for the downsampled channel
    downsampled = np.zeros((new_height, new_width), dtype=np.uint8)

    # Loop through the pixels of the downsampled channel and compute the average value
    for y in range(0, height, factor):
        for x in range(0, width, factor):
            # Get the values for the current block
            block = channel[y:y+factor, x:x+factor]

            # Compute the average value for the current block
            avg_value = int(np.mean(block))

            # Set the corresponding value in the downsampled channel
            downsampled[int(y/factor), int(x/factor)] = avg_value

    return downsampled

def bicubic_interpolation(x, a):
    if x < 0:
        x = -x
    if x <= 1:
        return (a + 2) * x**3 - (a + 3) * x**2 + 1
    elif x < 2:
        return a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a
    else:
        return 0
    
def bicubic_upsample(channel, factor, a):
    # Get channel size
    width, height = len(channel[0]), len(channel)

    # Compute the new size of the upsampled channel
    new_height = int(height*factor)+1
    new_width = int(width*factor)+1

    # Create a new numpy array for the upsampled channel
    upsampled = np.zeros((new_height, new_width), dtype=np.uint8)

    # Loop through the pixels of the upsampled channel and compute the bicubic interpolation
    for y in range(new_height):
        for x in range(new_width):
            # Compute the corresponding position in the original channel
            orig_y = (y + 0.5) / factor - 0.5
            orig_x = (x + 0.5) / factor - 0.5

            # Compute the indices and weights for the surrounding pixels
            y0 = int(np.floor(orig_y))
            x0 = int(np.floor(orig_x))
            indices = range(y0-1, y0+3)
            indices = [(i, j) for i in indices for j in range(x0-1, x0+3)]
            indices = [(i, j) for i, j in indices if i >= 0 and j >= 0 and i < height and j < width]
            weights = [bicubic_interpolation(y-y0-1, a) * bicubic_interpolation(x-x0-1, a) for y, x in indices]

            # Compute the interpolated value for the current pixel
            interpolated = sum([channel[i, j] * weights[n] for n, (i, j) in enumerate(indices)])

            # Set the corresponding value in the upsampled channel
            upsampled[y, x] = int(interpolated)
        print("Row " + str(y) + " is completed interpolation.")

    return upsampled

def bilinear_upsample(channel, factor):
    # Get channel size
    height, width = len(channel), len(channel[0])

    # Compute the new size of the upsampled channel
    new_height = int(height * factor)
    new_width = int(width * factor)

    # Create a new numpy array for the upsampled channel
    upsampled = np.zeros((new_height, new_width), dtype=np.uint8)

    # Loop through the pixels of the upsampled channel and compute the bilinear interpolation
    for y in range(new_height):
        for x in range(new_width):
            # Compute the corresponding position in the original channel
            orig_y = (y + 0.5) / factor - 0.5
            orig_x = (x + 0.5) / factor - 0.5

            # Compute the indices for the surrounding pixels
            y0 = int(orig_y)
            x0 = int(orig_x)
            y1 = y0 + 1
            x1 = x0 + 1

            # Check if the surrounding pixels are within the bounds of the original channel
            if y0 >= 0 and x0 >= 0 and y1 < height and x1 < width:
                # Compute the weights for the surrounding pixels
                wy0 = (y1 - orig_y) / (y1 - y0)
                wx0 = (x1 - orig_x) / (x1 - x0)
                wy1 = (orig_y - y0) / (y1 - y0)
                wx1 = (orig_x - x0) / (x1 - x0)

                # Compute the interpolated value for the current pixel
                interpolated = wy0 * wx0 * channel[y0, x0] + wy0 * wx1 * channel[y0, x1] + wy1 * wx0 * channel[y1, x0] + wy1 * wx1 * channel[y1, x1]

                # Set the corresponding value in the upsampled channel
                upsampled[y, x] = int(interpolated)
        print("Row " + str(y) + " is completed interpolation.")

    return upsampled

image_path = "hotdawg.jpg"
image = cv2.imread(image_path)
width, height = len(image[0]), len(image)
y, cb, cr = split_color_channels(image_path)

down_y = downsample_channel(y, 2)
down_cb = downsample_channel(cb, 16)
down_cr = downsample_channel(cr, 16)

up_y = bilinear_upsample(down_y, 2)
up_cb = bilinear_upsample(down_cb, 16)
up_cr = bilinear_upsample(down_cr, 16)

new_image = combine_color_channels(y, cb, cr, height, width)

cv2.imwrite("sampled_hotdawg.jpg", new_image)
