# HMM Textract Documentation

## Concepts

### Ground Truth Label
These are the true or correct annotations or labels associated with the data, in this case it is the text present in the image.

This is typically done by human annotators who carefully examine the images and mark the boundaries of text regions associating them with the corresponding text.

### Preprocessing Techniques

#### Resizing

Its modifying the image size through its dimensions (width & height) of an image while maintaining its aspect ratio or altering it as needed.

When downscaling an image sometimes its needed anti-aliasing techniques such as jagged edges or pixelation.

Interpolation its used to estimate the pixel values in the resized image based on the values of neighboring pixels in the original image. There are also many interpolation techniques.

#### Interpolation Techniques

##### Nearest-Neighbor

##### Bilinear

##### Cubic interpolation

#### Noise Removal

This is a technique done to reduce unwanted random variations or disturbances in an image which is what we refer to as "noise". This can be introduced because due to factors such as poor lighting conditions, sensor limitations, compression artifacts, or transimission errors.

There are different types of noise such as Gaussian noise, which follows a Gaussian distribution and appears as random variations in brightness. Salt-and-pepper noise manifests as random white and black pixels scattered throughout the image.

There are reduction filters or techniques to reduce noise from an image:
- Gaussian Blur: This filters applies a weighted average to each pixel neighborhood effectively blurring the image and smoothing out small-scaled variations.
- Median Filter: Replaces each pixel value with the median value of its neighboring pixels, useful for salt-and-pepper noise without blurring the edges.
- Bilateral Filter: Combination of spatial and intensity weights to preserve edges while reducing noise.

#### Contrast Adjustment

Enchances the quality and distinguishability of an image by expanding or compressing its range of pixel intensities. It aims to improve the perception of details.

1. Histogram: First step is to analyze the histogram, with this, we can get an understanding of the overall contrast present in the image.

2. Intensity Scaling: Linear mapping the pixel intensities of the image to a new range.
- We find the minimum and maximum values.
- Then define the desired minimum and maximum intensities (0 and 255 for example).
- Finally we use a linear mapping function to rescale it.
```
new_intensity = (old_intensity - min_intensity) * ((new_max - new_min) / (max_intensity - min_intensity)) + new_min
```
3. Histogram Equalization: It redistributes the pixel intensities in the image to make the histogram as uniformly distributed as possible.

#### Binarization

Converts a grayscale image or color image into a binary image where each pixel is represented by only two values: black or white. This is commonly applied in text extraction tasks to separate the foreground from the background.

1. Thresholding: This involves selecting a treshold value to divide the pixel intensities into two groups, those below and above or equal to the treshold.

2. Global Tresholding: Uses a single treshold value that is applied to all pixels in the image.
- Otsu's Tresholding: Otsu's method automatically calculates the optimal trehsold value, maximizing the between-class variance of the pixel intensities.

3. Adaptive tresholding: this methods take into account local variations in image intensity and adjust the treshold value in order to it. This is useful when the ilumination conditions or contrast levels vary across different regions of the image.

## Libraries

### CV2

#### imread

```py
cv2.imread(filename, flag)
```
Where:  
```filename```: path to the image file.  
```flag```: way how image is read.    
- cv2.IMREAD_COLOR - It specifies to load a **color image**. Any transparency of image will be neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.

- cv2.IMREAD_GRAYSCALE – It specifies to load an image in **grayscale mode**. Alternatively, we can pass integer value 0 for this flag. 

- cv2.IMREAD_UNCHANGED – It specifies to load an image as such including **alpha channel**. Alternatively, we can pass integer value -1 for this flag.

Returns a **NumPy array** if the image is loaded successfully.

