% Muhammad Umar Khan
% 400167784 | khanm214

% read in your image
img = imread('hotdawg.jpg');

% manually convert the image from RGB to YUV
redChannel = double(img(:,:,1));
greenChannel = double(img(:,:,2));
blueChannel = double(img(:,:,3));

% compute YUV channels
[h, w] = size(redChannel);
yChannel = double(zeros(h,w));
uChannel = double(zeros(h,w));
vChannel = double(zeros(h,w));

for i = 1:h
    for j = 1:w 
        yChannel(i,j) = 0.299*redChannel(i,j) + 0.587*greenChannel(i,j) + 0.114*blueChannel(i,j);
        uChannel(i,j) = -0.14713*redChannel(i,j) - 0.28886*greenChannel(i,j) + 0.436*blueChannel(i,j);
        vChannel(i,j) = 0.615*redChannel(i,j) - 0.51499*greenChannel(i,j) - 0.10001*blueChannel(i,j);
    end
end

% ------------------- set the downsampling factor for the y channel only
% ---------------------------------------------------------------------------------------------------------
downsampleFactor = 2;

% manually downsample each channel and cast to uint8
[h, w] = size(yChannel);
downsampledY = cast(zeros(ceil(h/downsampleFactor), ceil(w/downsampleFactor)), 'double');

for i = 1:downsampleFactor:h
    for j = 1:downsampleFactor:w
        downsampledY(ceil((i+1)/downsampleFactor), ceil((j+1)/downsampleFactor)) = yChannel(i,j);
    end
end

% ------------------- set the downsampling factor for the u and v channels
% ---------------------------------------------------------------------------------------------------------
downsampleFactor = 32;

% manually downsample each channel and cast to uint8
[h, w] = size(yChannel);
downsampledU = cast(zeros(ceil(h/downsampleFactor), ceil(w/downsampleFactor)), 'double');
downsampledV = cast(zeros(ceil(h/downsampleFactor), ceil(w/downsampleFactor)), 'double');

for i = 1:downsampleFactor:h
    for j = 1:downsampleFactor:w
        downsampledU(ceil((i+1)/downsampleFactor), ceil((j+1)/downsampleFactor)) = uChannel(i,j);
        downsampledV(ceil((i+1)/downsampleFactor), ceil((j+1)/downsampleFactor)) = vChannel(i,j);
    end
end

% ------------------- Define the upsampling factor for the y channel
% ---------------------------------------------------------------------------------------------------------
upsampleFactorY = 2;

% Get the height and width of the original image
[h, w] = size(downsampledY);

% Compute the new height and width of the upsampled image
newHeight = upsampleFactorY*h;
newWidth = upsampleFactorY*w;

upsampledY = double(zeros(newHeight, newWidth));

% Compute the scaling factor in x and y directions
scaleX = (w - 1) / (newWidth - 1);
scaleY = (h - 1) / (newHeight - 1);

% Perform bilinear upsampling for the y channel
for j = 1:newWidth
    % Compute the x-coordinate in the original image corresponding to the current x-coordinate in the upsampled image
    x = (j-1)*scaleX + 1;
    x1 = floor(x);
    x2 = ceil(x);

    for i = 1:newHeight
        % Compute the y-coordinate in the original image corresponding to the current y-coordinate in the upsampled image
        y = (i-1)*scaleY + 1;
        y1 = floor(y);
        y2 = ceil(y);

        % Compute the weights for each of the four surrounding pixels
        w1 = (x2 - x) * (y2 - y);
        w2 = (x - x1) * (y2 - y);
        w3 = (x2 - x) * (y - y1);
        w4 = (x - x1) * (y - y1);

        % Compute the new pixel value using the weighted sum of the four surrounding pixels
        upsampledY(i, j) = w1*downsampledY(y1, x1) + w2*downsampledY(y1, x2) + w3*downsampledY(y2, x1) + w4*downsampledY(y2, x2);
    end
end

% ------------------- Define the upsampling factor for the u and v channel
% ---------------------------------------------------------------------------------------------------------
upsampleFactorUV = 32;

% Get the height and width of the original image
[h, w] = size(downsampledU);

% Compute the new height and width of the upsampled image
newHeight = upsampleFactorUV*h;
newWidth = upsampleFactorUV*w;

upsampledU = double(zeros(newHeight, newWidth));
upsampledV = double(zeros(newHeight, newWidth));

% Compute the scaling factor in x and y directions
scaleX = (w - 1) / (newWidth - 1);
scaleY = (h - 1) / (newHeight - 1);

% Perform bilinear upsampling for the u and v channels
for j = 1:newWidth
    % Compute the x-coordinate in the original image corresponding to the current x-coordinate in the upsampled image
    x = (j-1)*scaleX + 1;
    x1 = floor(x);
    x2 = ceil(x);

    for i = 1:newHeight
        % Compute the y-coordinate in the original image corresponding to the current y-coordinate in the upsampled image
        y = (i-1)*scaleY + 1;
        y1 = floor(y);
        y2 = ceil(y);

        % Compute the weights for each of the four surrounding pixels
        w1 = (x2 - x) * (y2 - y);
        w2 = (x - x1) * (y2 - y);
        w3 = (x2 - x) * (y - y1);
        w4 = (x - x1) * (y - y1);

        % Compute the new pixel value using the weighted sum of the four surrounding pixels
        upsampledU(i, j) = w1*downsampledU(y1, x1) + w2*downsampledU(y1, x2) + w3*downsampledU(y2, x1) + w4*downsampledU(y2, x2);
        upsampledV(i, j) = w1*downsampledV(y1, x1) + w2*downsampledV(y1, x2) + w3*downsampledV(y2, x1) + w4*downsampledV(y2, x2);
    end
end

% manually convert the downsampled YUV image to RGB
% ---------------------------------------------------------------------------------------------------------
[h, w] = size(upsampledY);
upsampledRed = double(zeros(h,w));
upsampledGreen = double(zeros(h,w));
upsampledBlue = double(zeros(h,w));

for i = 1:h
    for j = 1:w
        upsampledRed(i,j) = upsampledY(i,j) + 1.13983*upsampledV(i,j);
        upsampledGreen(i,j) = upsampledY(i,j) - 0.39465*upsampledU(i,j) - 0.58060*upsampledV(i,j);
        upsampledBlue(i,j) = upsampledY(i,j) + 2.03211*upsampledU(i,j);
    end
end

% Compute the mean squared error (MSE) between the original and upsampled image
mse = 0;
originalImg = cat(3, redChannel, greenChannel, blueChannel);
upsampledImg = cat(3, upsampledRed, upsampledGreen, upsampledBlue);
[h, w, numChannels] = size(originalImg);
for c = 1:numChannels
    for i = 1:h
        for j = 1:w
            mse = mse + (originalImg(i,j,c) - upsampledImg(i,j,c))^2;
        end
    end
end
mse = mse / (h*w*numChannels);

% Compute the peak signal-to-noise ratio (PSNR) between the original and upsampled image
psnr = 10*log(255/mse);

% Display the MSE and PSNR values

% cast the downsampled RGB channels to uint8
upsampledRed = cast(upsampledRed, 'uint8');
upsampledGreen = cast(upsampledGreen, 'uint8');
upsampledBlue = cast(upsampledBlue, 'uint8');

% clip the color channels to [0,255]
upsampledRed(upsampledRed < 0) = 0;
upsampledGreen(upsampledGreen < 0) = 0;
upsampledBlue(upsampledBlue < 0) = 0;
upsampledRed(upsampledRed > 255) = 255;
upsampledGreen(upsampledGreen > 255) = 255;
upsampledBlue(upsampledBlue > 255) = 255;

% combine the downsampled channels into an RGB image
upsampledImgRGB = cat(3, upsampledRed, upsampledGreen, upsampledBlue);

% display the original and downsampled images
figure;
subplot(1,2,1);
imshow(img);
title('Original Image');
subplot(1,2,2);
imshow(upsampledImgRGB);
title('Upsampled Image');

clc
fprintf('Compression factor of Y channel: %d \n', upsampleFactorY);
fprintf('Compression factor of U and V channels: %d \n', upsampleFactorUV);
fprintf('The mean squared error (MSE) between the original and upsampled image is: %0.2f\n', mse);
fprintf('PSNR: %0.3f dB \n', psnr);
