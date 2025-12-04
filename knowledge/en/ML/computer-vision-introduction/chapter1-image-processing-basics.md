---
title: "Chapter 1: Fundamentals of Image Processing"
chapter_title: "Chapter 1: Fundamentals of Image Processing"
subtitle: First Steps in Computer Vision - Understanding Digital Image Representation and Basic Operations
reading_time: 30-35 minutes
difficulty: Beginner
code_examples: 10
exercises: 5
---

This chapter covers the fundamentals of Fundamentals of Image Processing, which image representation. You will learn digital image representation methods (pixels, basic image processing operations such as resizing, and filtering techniques including smoothing.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand digital image representation methods (pixels, channels)
  * ✅ Explain differences and applications of color spaces such as RGB, HSV, and LAB
  * ✅ Read, save, and display images using OpenCV and PIL
  * ✅ Implement basic image processing operations such as resizing, rotation, and color conversion
  * ✅ Apply filtering techniques including smoothing and edge detection
  * ✅ Build image data preprocessing pipelines for machine learning

* * *

## 1.1 Image Representation

### Digital Images and Pixels

**Digital images** are represented as a collection of discrete points (pixels). Each pixel holds color information (intensity values).

> "Images are treated as two-dimensional arrays of pixels, forming a data structure that enables numerical computation."

#### Basic Image Structure

  * **Height** : Number of pixels in the vertical direction
  * **Width** : Number of pixels in the horizontal direction
  * **Channels** : Number of color components (Grayscale=1, RGB=3)

Image shape is typically represented in one of the following formats:

Format | Dimension Order | Used by Libraries  
---|---|---  
**HWC** | (Height, Width, Channels) | OpenCV, PIL, matplotlib  
**CHW** | (Channels, Height, Width) | PyTorch, Caffe  
**NHWC** | (Batch, Height, Width, Channels) | TensorFlow, Keras  
**NCHW** | (Batch, Channels, Height, Width) | PyTorch  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: Image shape is typically represented in one of the following
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create a grayscale image (height 5, width 5)
    gray_image = np.array([
        [0, 50, 100, 150, 200],
        [50, 100, 150, 200, 250],
        [100, 150, 200, 250, 255],
        [150, 200, 250, 255, 200],
        [200, 250, 255, 200, 150]
    ], dtype=np.uint8)
    
    # Create an RGB image (height 3, width 3, channels 3)
    rgb_image = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb_image[0, 0] = [255, 0, 0]      # Red
    rgb_image[0, 1] = [0, 255, 0]      # Green
    rgb_image[0, 2] = [0, 0, 255]      # Blue
    rgb_image[1, 1] = [255, 255, 0]    # Yellow
    rgb_image[2, 2] = [255, 255, 255]  # White
    
    print("=== Grayscale Image ===")
    print(f"Shape: {gray_image.shape}")
    print(f"Data type: {gray_image.dtype}")
    print(f"Min value: {gray_image.min()}, Max value: {gray_image.max()}")
    print(f"\nImage data:\n{gray_image}")
    
    print("\n=== RGB Image ===")
    print(f"Shape: {rgb_image.shape} (Height, Width, Channels)")
    print(f"Top-left pixel value (R,G,B): {rgb_image[0, 0]}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(gray_image, cmap='gray')
    axes[0].set_title('Grayscale Image (5×5)')
    axes[0].axis('off')
    
    axes[1].imshow(rgb_image)
    axes[1].set_title('RGB Image (3×3)')
    axes[1].axis('off')
    
    plt.tight_layout()
    print("\nImages visualized")
    

### Color Spaces

Color spaces are methods for representing colors numerically. Selecting an appropriate color space based on the application is important.

#### RGB (Red, Green, Blue)

The most common color space, based on additive color mixing (primary colors of light).

  * Value range per channel: 0-255 (8-bit integer) or 0.0-1.0 (floating-point)
  * Applications: Digital cameras, displays, image storage
  * Characteristics: Intuitive but does not align with human color perception

#### HSV (Hue, Saturation, Value)

Represents colors using hue, saturation, and value (brightness).

  * **Hue** : 0-179 degrees (0-180 in OpenCV), type of color
  * **Saturation** : 0-255, vividness of color
  * **Value** : 0-255, brightness of color
  * Applications: Color-based object detection, image segmentation

#### LAB (L*a*b*)

A color space closer to human vision.

  * **L** : Lightness (0-100)
  * **a** : Green-red axis (-128 to 127)
  * **b** : Blue-yellow axis (-128 to 127)
  * Applications: Color difference calculation, image correction

    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: A color space closer to human vision.
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create sample image (red, green, blue blocks)
    sample = np.zeros((100, 300, 3), dtype=np.uint8)
    sample[:, 0:100] = [255, 0, 0]     # Red
    sample[:, 100:200] = [0, 255, 0]   # Green
    sample[:, 200:300] = [0, 0, 255]   # Blue
    
    # Convert from BGR to RGB (OpenCV uses BGR order)
    sample_rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    
    # Convert to various color spaces
    sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
    sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original image (RGB)
    axes[0, 0].imshow(sample_rgb)
    axes[0, 0].set_title('Original (RGB)')
    axes[0, 0].axis('off')
    
    # HSV (display each channel separately)
    axes[0, 1].imshow(sample_hsv[:, :, 0], cmap='hsv')
    axes[0, 1].set_title('HSV - Hue')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sample_hsv[:, :, 1], cmap='gray')
    axes[0, 2].set_title('HSV - Saturation')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(sample_hsv[:, :, 2], cmap='gray')
    axes[0, 3].set_title('HSV - Value')
    axes[0, 3].axis('off')
    
    # LAB (display each channel separately)
    axes[1, 0].imshow(sample_lab[:, :, 0], cmap='gray')
    axes[1, 0].set_title('LAB - L (Lightness)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sample_lab[:, :, 1], cmap='RdYlGn_r')
    axes[1, 1].set_title('LAB - a (Green-Red)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(sample_lab[:, :, 2], cmap='RdYlBu_r')
    axes[1, 2].set_title('LAB - b (Blue-Yellow)')
    axes[1, 2].axis('off')
    
    # Grayscale
    axes[1, 3].imshow(sample_gray, cmap='gray')
    axes[1, 3].set_title('Grayscale')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    print("Converted to various color spaces and visualized")
    
    # Check numerical values
    print(f"\nPixel values at center of red block (position [50, 50]):")
    print(f"  RGB: {sample_rgb[50, 50]}")
    print(f"  HSV: {sample_hsv[50, 50]}")
    print(f"  LAB: {sample_lab[50, 50]}")
    print(f"  Gray: {sample_gray[50, 50]}")
    

### Loading and Saving Images

#### Using OpenCV
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Using OpenCV
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    
    # Load image (create dummy image if not exists)
    try:
        image = cv2.imread('sample.jpg')
        if image is None:
            raise FileNotFoundError
    except:
        # Create dummy image (gradient)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            for j in range(640):
                image[i, j] = [i * 255 // 480, j * 255 // 640, 128]
        print("Created dummy image")
    
    print(f"Image shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    
    # Load as grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Save images
    cv2.imwrite('output_color.jpg', image)
    cv2.imwrite('output_gray.jpg', gray)
    
    print("\nImages saved:")
    print("  - output_color.jpg (color)")
    print("  - output_gray.jpg (grayscale)")
    
    # Display image information
    print(f"\nColor image:")
    print(f"  Shape: {image.shape}")
    print(f"  Memory size: {image.nbytes / 1024:.2f} KB")
    
    print(f"\nGrayscale image:")
    print(f"  Shape: {gray.shape}")
    print(f"  Memory size: {gray.nbytes / 1024:.2f} KB")
    

#### Using PIL
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pillow>=10.0.0
    
    """
    Example: Using PIL
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    from PIL import Image
    import numpy as np
    
    # Create PIL Image (gradient)
    width, height = 640, 480
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            array[i, j] = [i * 255 // height, j * 255 // width, 128]
    
    pil_image = Image.fromarray(array)
    
    print(f"PIL image size: {pil_image.size}")  # (width, height)
    print(f"PIL image mode: {pil_image.mode}")
    
    # Convert to grayscale
    pil_gray = pil_image.convert('L')
    
    # Save
    pil_image.save('output_pil_color.png')
    pil_gray.save('output_pil_gray.png')
    
    # PIL → NumPy array
    np_array = np.array(pil_image)
    print(f"\nConverted to NumPy array:")
    print(f"  Shape: {np_array.shape}")
    
    # NumPy array → PIL
    pil_from_numpy = Image.fromarray(np_array)
    print(f"\nConverted to PIL image:")
    print(f"  Size: {pil_from_numpy.size}")
    
    print("\nSaved images using PIL:")
    print("  - output_pil_color.png")
    print("  - output_pil_gray.png")
    

* * *

## 1.2 Basic Image Processing

### Resizing and Cropping

**Resizing** changes the size of an image. Quality varies depending on the interpolation method.

Interpolation Method | Characteristics | Applications  
---|---|---  
**NEAREST** | Nearest neighbor, fast but low quality | Integer scaling  
**LINEAR** | Linear interpolation, good balance | General downscaling  
**CUBIC** | Cubic interpolation, high quality but slow | Upscaling  
**LANCZOS** | Highest quality, slowest | When high quality is required  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Resizingchanges the size of an image. Quality varies dependi
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create sample image (checkerboard)
    def create_checkerboard(size=200, square_size=20):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = [255, 255, 255]
        return image
    
    original = create_checkerboard(200, 20)
    
    # Resize with various interpolation methods
    resized_nearest = cv2.resize(original, (400, 400), interpolation=cv2.INTER_NEAREST)
    resized_linear = cv2.resize(original, (400, 400), interpolation=cv2.INTER_LINEAR)
    resized_cubic = cv2.resize(original, (400, 400), interpolation=cv2.INTER_CUBIC)
    resized_lanczos = cv2.resize(original, (400, 400), interpolation=cv2.INTER_LANCZOS4)
    
    # Downscale
    resized_small = cv2.resize(original, (100, 100), interpolation=cv2.INTER_AREA)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    images = [
        (cv2.cvtColor(original, cv2.COLOR_BGR2RGB), "Original (200×200)"),
        (cv2.cvtColor(resized_nearest, cv2.COLOR_BGR2RGB), "NEAREST (400×400)"),
        (cv2.cvtColor(resized_linear, cv2.COLOR_BGR2RGB), "LINEAR (400×400)"),
        (cv2.cvtColor(resized_cubic, cv2.COLOR_BGR2RGB), "CUBIC (400×400)"),
        (cv2.cvtColor(resized_lanczos, cv2.COLOR_BGR2RGB), "LANCZOS (400×400)"),
        (cv2.cvtColor(resized_small, cv2.COLOR_BGR2RGB), "AREA (100×100 downscale)"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("Compared resizing with various interpolation methods")
    
    # Crop (trim)
    print("\n=== Cropping Example ===")
    height, width = original.shape[:2]
    x, y, w, h = 50, 50, 100, 100  # (x, y, width, height)
    
    cropped = original[y:y+h, x:x+w]
    print(f"Original image: {original.shape}")
    print(f"After cropping: {cropped.shape}")
    print(f"Crop region: x={x}, y={y}, width={w}, height={h}")
    

### Rotation and Flipping
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Rotation and Flipping
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create test image (arrow pattern)
    def create_arrow(size=200):
        image = np.ones((size, size, 3), dtype=np.uint8) * 255
        # Draw arrow
        pts = np.array([[100, 50], [150, 100], [125, 100], [125, 150],
                        [75, 150], [75, 100], [50, 100]], np.int32)
        cv2.fillPoly(image, [pts], (0, 0, 255))
        return image
    
    original = create_arrow()
    
    # Rotation (90, 180, 270 degrees)
    rotated_90 = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(original, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Arbitrary angle rotation (45 degrees)
    height, width = original.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_45 = cv2.warpAffine(original, rotation_matrix, (width, height))
    
    # Flipping
    flipped_horizontal = cv2.flip(original, 1)  # Horizontal flip
    flipped_vertical = cv2.flip(original, 0)    # Vertical flip
    flipped_both = cv2.flip(original, -1)       # Both directions
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    images = [
        (cv2.cvtColor(original, cv2.COLOR_BGR2RGB), "Original"),
        (cv2.cvtColor(rotated_90, cv2.COLOR_BGR2RGB), "Rotated 90°"),
        (cv2.cvtColor(rotated_180, cv2.COLOR_BGR2RGB), "Rotated 180°"),
        (cv2.cvtColor(rotated_270, cv2.COLOR_BGR2RGB), "Rotated 270°"),
        (cv2.cvtColor(rotated_45, cv2.COLOR_BGR2RGB), "Rotated 45°"),
        (cv2.cvtColor(flipped_horizontal, cv2.COLOR_BGR2RGB), "Horizontal Flip"),
        (cv2.cvtColor(flipped_vertical, cv2.COLOR_BGR2RGB), "Vertical Flip"),
        (cv2.cvtColor(flipped_both, cv2.COLOR_BGR2RGB), "Both Directions Flip"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("Visualized rotation and flipping operations")
    
    # Rotation matrix details
    print("\n=== 45-Degree Rotation Transform Matrix ===")
    print(rotation_matrix)
    

### Color Conversion and Histograms
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Color Conversion and Histograms
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create sample image (gradient + noise)
    def create_sample_image():
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        for i in range(300):
            for j in range(400):
                image[i, j] = [
                    int(i * 255 / 300),
                    int(j * 255 / 400),
                    128
                ]
        # Add noise
        noise = np.random.randint(-30, 30, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return image
    
    image = create_sample_image()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Calculate histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # RGB histogram
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        axes[0, 1].plot(hist, color=color, label=f'{color.upper()} channel')
    axes[0, 1].set_title('RGB Histogram')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    axes[1, 0].imshow(equalized, cmap='gray')
    axes[1, 0].set_title('After Histogram Equalization')
    axes[1, 0].axis('off')
    
    # Compare histograms before and after equalization
    hist_before = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    axes[1, 1].plot(hist_before, 'b-', label='Before Equalization', alpha=0.7)
    axes[1, 1].plot(hist_after, 'r-', label='After Equalization', alpha=0.7)
    axes[1, 1].set_title('Grayscale Histogram')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("Visualized histograms and equalization")
    
    # Statistical information
    print("\n=== Image Statistics ===")
    print(f"Mean value (before equalization): {gray.mean():.2f}")
    print(f"Mean value (after equalization): {equalized.mean():.2f}")
    print(f"Standard deviation (before equalization): {gray.std():.2f}")
    print(f"Standard deviation (after equalization): {equalized.std():.2f}")
    

* * *

## 1.3 Filtering

### Smoothing Filters

**Smoothing** removes noise from images or creates blurring effects.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Smoothingremoves noise from images or creates blurring effec
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create image with noise
    def create_noisy_image(size=200):
        # Clean image (circle)
        image = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(image, (size//2, size//2), size//3, 255, -1)
    
        # Add Gaussian noise
        noise = np.random.normal(0, 25, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    
        # Add salt & pepper noise
        salt_pepper = noisy.copy()
        salt = np.random.random(image.shape) < 0.02
        pepper = np.random.random(image.shape) < 0.02
        salt_pepper[salt] = 255
        salt_pepper[pepper] = 0
    
        return image, noisy, salt_pepper
    
    clean, gaussian_noisy, sp_noisy = create_noisy_image()
    
    # Apply various smoothing filters
    # Mean Filter
    mean_blur = cv2.blur(gaussian_noisy, (5, 5))
    
    # Gaussian Filter
    gaussian_blur = cv2.GaussianBlur(gaussian_noisy, (5, 5), 0)
    
    # Median Filter - effective for salt & pepper noise
    median_blur = cv2.medianBlur(sp_noisy, 5)
    
    # Bilateral Filter - smooths while preserving edges
    bilateral = cv2.bilateralFilter(gaussian_noisy, 9, 75, 75)
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    images = [
        (clean, "Clean Image"),
        (gaussian_noisy, "Gaussian Noise"),
        (mean_blur, "Mean Filter"),
        (gaussian_blur, "Gaussian Filter"),
        (sp_noisy, "Salt & Pepper Noise"),
        (median_blur, "Median Filter"),
        (bilateral, "Bilateral Filter"),
        (gaussian_noisy - gaussian_blur, "Noise Component (Difference)"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("Compared effects of various smoothing filters")
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    def calculate_psnr(original, filtered):
        mse = np.mean((original - filtered) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    print("\n=== Noise Removal Performance (PSNR, higher is better) ===")
    print(f"Noisy image: {calculate_psnr(clean, gaussian_noisy):.2f} dB")
    print(f"Mean filter: {calculate_psnr(clean, mean_blur):.2f} dB")
    print(f"Gaussian filter: {calculate_psnr(clean, gaussian_blur):.2f} dB")
    print(f"Bilateral filter: {calculate_psnr(clean, bilateral):.2f} dB")
    

### Edge Detection

**Edge detection** identifies rapid changes in intensity within an image.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Edge detectionidentifies rapid changes in intensity within a
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create test image (multiple shapes)
    def create_shapes_image(size=300):
        image = np.ones((size, size), dtype=np.uint8) * 200
        # Rectangle
        cv2.rectangle(image, (50, 50), (120, 120), 50, -1)
        # Circle
        cv2.circle(image, (220, 80), 40, 100, -1)
        # Triangle
        pts = np.array([[150, 200], [100, 280], [200, 280]], np.int32)
        cv2.fillPoly(image, [pts], 150)
        return image
    
    image = create_shapes_image()
    
    # Sobel filter (X direction, Y direction)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
    
    # Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Canny edge detection
    canny = cv2.Canny(image, 50, 150)
    
    # Scharr filter (more accurate than Sobel)
    scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr_combined = np.uint8(scharr_combined / scharr_combined.max() * 255)
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    images = [
        (image, "Original Image"),
        (sobel_x, "Sobel X"),
        (sobel_y, "Sobel Y"),
        (sobel_combined, "Sobel Combined"),
        (laplacian, "Laplacian"),
        (canny, "Canny"),
        (scharr_combined, "Scharr Combined"),
        (image, "Original Image (Reference)"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("Compared various edge detection filters")
    
    # Analyze edge strength
    print("\n=== Edge Detection Result Statistics ===")
    print(f"Sobel: Mean intensity = {sobel_combined.mean():.2f}")
    print(f"Laplacian: Mean intensity = {laplacian.mean():.2f}")
    print(f"Canny: Edge pixel count = {np.sum(canny > 0)}")
    print(f"Scharr: Mean intensity = {scharr_combined.mean():.2f}")
    

### Morphological Operations

**Morphological operations** are shape processing techniques for binary images.

Operation | Effect | Applications  
---|---|---  
**Dilation** | Expands white regions | Fill holes, connect broken parts  
**Erosion** | Shrinks white regions | Remove noise, eliminate thin lines  
**Opening** | Erosion → Dilation | Remove small noise  
**Closing** | Dilation → Erosion | Fill holes and gaps  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Morphological operationsare shape processing techniques for 
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create noisy binary image
    def create_noisy_binary_image(size=200):
        image = np.zeros((size, size), dtype=np.uint8)
        # Main shape (rectangle)
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
        # Add small holes
        for _ in range(10):
            x, y = np.random.randint(60, 140, 2)
            cv2.circle(image, (x, y), 3, 0, -1)
        # Add small noise
        for _ in range(20):
            x, y = np.random.randint(0, size, 2)
            cv2.circle(image, (x, y), 2, 255, -1)
        return image
    
    binary = create_noisy_binary_image()
    
    # Create kernel (structuring element)
    kernel = np.ones((5, 5), np.uint8)
    
    # Morphological operations
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    images = [
        (binary, "Original Image (with noise)"),
        (erosion, "Erosion"),
        (dilation, "Dilation"),
        (opening, "Opening"),
        (closing, "Closing"),
        (gradient, "Gradient"),
        (tophat, "Top Hat"),
        (blackhat, "Black Hat"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("Visualized morphological operations")
    
    # Compare pixel counts
    print("\n=== Changes in White Pixel Count ===")
    print(f"Original image: {np.sum(binary == 255):,} pixels")
    print(f"After erosion: {np.sum(erosion == 255):,} pixels ({np.sum(erosion == 255) / np.sum(binary == 255) * 100:.1f}%)")
    print(f"After dilation: {np.sum(dilation == 255):,} pixels ({np.sum(dilation == 255) / np.sum(binary == 255) * 100:.1f}%)")
    print(f"After opening: {np.sum(opening == 255):,} pixels")
    print(f"After closing: {np.sum(closing == 255):,} pixels")
    

* * *

## 1.4 Feature Extraction

### Corner Detection

**Corners** are important feature points in images, used for object recognition and tracking.

#### Harris Corner Detection
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Harris Corner Detection
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create checkerboard pattern
    def create_checkerboard_complex(size=400):
        image = np.zeros((size, size), dtype=np.uint8)
        square_size = 40
        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = 255
        # Additional shape
        cv2.circle(image, (300, 300), 50, 128, -1)
        return image
    
    image = create_checkerboard_complex()
    
    # Harris corner detection
    harris = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)  # Enhance results
    
    # Apply threshold to detect corners
    image_harris = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_harris[harris > 0.01 * harris.max()] = [0, 0, 255]
    
    # Shi-Tomasi corner detection (Good Features to Track)
    corners = cv2.goodFeaturesToTrack(image, maxCorners=100, qualityLevel=0.01, minDistance=10)
    image_shi = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image_shi, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(image_harris, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Harris Corner Detection')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(image_shi, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Shi-Tomasi ({len(corners) if corners is not None else 0} corners)')
    axes[2].axis('off')
    
    plt.tight_layout()
    print("Compared corner detection algorithms")
    
    if corners is not None:
        print(f"\nNumber of corners detected: {len(corners)}")
        print(f"First 5 corner coordinates:")
        for i, corner in enumerate(corners[:5]):
            x, y = corner.ravel()
            print(f"  Corner{i+1}: ({x:.1f}, {y:.1f})")
    

### SIFT / ORB Features

**SIFT (Scale-Invariant Feature Transform)** is a feature descriptor invariant to scale and rotation.

> Note: In some versions of OpenCV, SIFT is included in opencv-contrib due to patent issues. Here we mainly use ORB (Oriented FAST and Rotated BRIEF).
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Note: In some versions of OpenCV, SIFT is included in opencv
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create distinctive image
    def create_feature_image(size=400):
        image = np.ones((size, size), dtype=np.uint8) * 200
        # Multiple shapes
        cv2.rectangle(image, (50, 50), (150, 150), 50, -1)
        cv2.circle(image, (300, 100), 50, 100, -1)
        cv2.rectangle(image, (100, 250), (200, 350), 150, 3)
        pts = np.array([[250, 250], [350, 280], [320, 350]], np.int32)
        cv2.fillPoly(image, [pts], 80)
        return image
    
    image = create_feature_image()
    
    # Create ORB feature detector
    orb = cv2.ORB_create(nfeatures=100)
    
    # Detect keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    # Draw keypoints
    image_keypoints = cv2.drawKeypoints(
        image, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(image_keypoints)
    axes[1].set_title(f'ORB Feature Points ({len(keypoints)} points)')
    axes[1].axis('off')
    
    plt.tight_layout()
    print(f"Detected ORB features: {len(keypoints)} keypoints")
    
    # Feature details
    print("\n=== ORB Feature Details ===")
    print(f"Number of keypoints: {len(keypoints)}")
    if descriptors is not None:
        print(f"Descriptor shape: {descriptors.shape}")
        print(f"  Each keypoint is described by a {descriptors.shape[1]}-dimensional vector")
    
    # First 5 keypoint information
    print("\nFirst 5 keypoints:")
    for i, kp in enumerate(keypoints[:5]):
        print(f"  Point {i+1}: Position=({kp.pt[0]:.1f}, {kp.pt[1]:.1f}), "
              f"Size={kp.size:.1f}, Angle={kp.angle:.1f}°")
    

### HOG (Histogram of Oriented Gradients)

**HOG** uses histograms of gradient orientations as features. Widely used for pedestrian detection and other applications.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: HOGuses histograms of gradient orientations as features. Wid
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from skimage.feature import hog
    from skimage import exposure
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create simple person silhouette
    def create_person_silhouette(size=128):
        image = np.zeros((size, size), dtype=np.uint8)
        # Head
        cv2.circle(image, (size//2, size//4), size//8, 255, -1)
        # Body
        cv2.rectangle(image, (size//2 - size//10, size//4 + size//10),
                      (size//2 + size//10, size//2 + size//6), 255, -1)
        # Arms
        cv2.line(image, (size//2 - size//10, size//4 + size//6),
                 (size//2 - size//4, size//2), 255, size//20)
        cv2.line(image, (size//2 + size//10, size//4 + size//6),
                 (size//2 + size//4, size//2), 255, size//20)
        # Legs
        cv2.line(image, (size//2 - size//20, size//2 + size//6),
                 (size//2 - size//10, size - size//8), 255, size//20)
        cv2.line(image, (size//2 + size//20, size//2 + size//6),
                 (size//2 + size//10, size - size//8), 255, size//20)
        return image
    
    image = create_person_silhouette()
    
    # Calculate HOG features
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
    
    # Enhance HOG image contrast
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image (Person Silhouette)')
    axes[0].axis('off')
    
    axes[1].imshow(hog_image_rescaled, cmap='gray')
    axes[1].set_title('HOG Feature Visualization')
    axes[1].axis('off')
    
    # Display part of HOG feature vector
    axes[2].bar(range(min(100, len(fd))), fd[:100])
    axes[2].set_title('HOG Feature Vector (First 100 Dimensions)')
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Value')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("Calculated HOG features")
    
    print(f"\n=== HOG Feature Details ===")
    print(f"Feature vector dimension: {len(fd)}")
    print(f"Mean value: {fd.mean():.4f}")
    print(f"Standard deviation: {fd.std():.4f}")
    print(f"Max value: {fd.max():.4f}")
    print(f"Min value: {fd.min():.4f}")
    

* * *

## 1.5 Image Data Preprocessing

### Normalization and Standardization

In machine learning, proper scaling of image data is important.

Method | Formula | Applications  
---|---|---  
**Min-Max Normalization** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Convert range to [0, 1]  
**Standardization (Z-score)** | $x' = \frac{x - \mu}{\sigma}$ | Convert to mean 0, variance 1  
**ImageNet Normalization** | Standardize per channel | Using pretrained models  
      
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    """
    Example: In machine learning, proper scaling of image data is importa
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create sample image
    np.random.seed(42)
    image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    
    print("=== Original Image Statistics ===")
    print(f"Shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Value range: [{image.min()}, {image.max()}]")
    print(f"Mean value: {image.mean():.2f}")
    print(f"Standard deviation: {image.std():.2f}")
    
    # Min-Max normalization [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    print("\n=== After Min-Max Normalization ===")
    print(f"Data type: {normalized.dtype}")
    print(f"Value range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"Mean value: {normalized.mean():.3f}")
    
    # Standardization (Z-score)
    mean = image.mean(axis=(0, 1), keepdims=True)
    std = image.std(axis=(0, 1), keepdims=True)
    standardized = (image.astype(np.float32) - mean) / (std + 1e-7)
    
    print("\n=== After Standardization (Z-score) ===")
    print(f"Mean value: {standardized.mean():.6f} (≈ 0)")
    print(f"Standard deviation: {standardized.std():.6f} (≈ 1)")
    print(f"Value range: [{standardized.min():.3f}, {standardized.max():.3f}]")
    
    # ImageNet standardization (for pretrained models)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    imagenet_normalized = (normalized - imagenet_mean) / imagenet_std
    
    print("\n=== After ImageNet Standardization ===")
    print(f"R channel mean: {imagenet_normalized[:,:,0].mean():.3f}")
    print(f"G channel mean: {imagenet_normalized[:,:,1].mean():.3f}")
    print(f"B channel mean: {imagenet_normalized[:,:,2].mean():.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image\n[50, 200]')
    axes[0].axis('off')
    
    axes[1].imshow(normalized)
    axes[1].set_title('Min-Max Normalization\n[0, 1]')
    axes[1].axis('off')
    
    # Adjust standardized image for visualization
    standardized_vis = (standardized - standardized.min()) / (standardized.max() - standardized.min())
    axes[2].imshow(standardized_vis)
    axes[2].set_title('Standardization\nMean≈0, Variance≈1')
    axes[2].axis('off')
    
    # Adjust ImageNet normalization
    imagenet_vis = (imagenet_normalized - imagenet_normalized.min()) / \
                   (imagenet_normalized.max() - imagenet_normalized.min())
    axes[3].imshow(imagenet_vis)
    axes[3].set_title('ImageNet Standardization')
    axes[3].axis('off')
    
    plt.tight_layout()
    print("\nVisualized effects of normalization and standardization")
    

### Data Augmentation

**Data augmentation** generates diverse variations from limited training data to improve model generalization performance.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    # - pillow>=10.0.0
    
    """
    Example: Data augmentationgenerates diverse variations from limited t
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image, ImageEnhance
    
    # Create sample object image
    def create_sample_object(size=128):
        image = np.ones((size, size, 3), dtype=np.uint8) * 255
        # Arrow-shaped object
        pts = np.array([[size//2, size//4], [3*size//4, size//2],
                        [5*size//8, size//2], [5*size//8, 3*size//4],
                        [3*size//8, 3*size//4], [3*size//8, size//2],
                        [size//4, size//2]], np.int32)
        cv2.fillPoly(image, [pts], (30, 144, 255))
        return image
    
    original = create_sample_object()
    
    # Apply various data augmentation techniques
    # 1. Rotation
    rotated = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)
    
    # 2. Horizontal flip
    flipped = cv2.flip(original, 1)
    
    # 3. Random crop & resize
    h, w = original.shape[:2]
    crop_size = 96
    x, y = np.random.randint(0, w - crop_size), np.random.randint(0, h - crop_size)
    cropped = original[y:y+crop_size, x:x+crop_size]
    cropped_resized = cv2.resize(cropped, (128, 128))
    
    # 4. Brightness adjustment
    pil_img = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    brightness = ImageEnhance.Brightness(pil_img).enhance(1.5)
    brightness = cv2.cvtColor(np.array(brightness), cv2.COLOR_RGB2BGR)
    
    # 5. Contrast adjustment
    contrast = ImageEnhance.Contrast(pil_img).enhance(1.5)
    contrast = cv2.cvtColor(np.array(contrast), cv2.COLOR_RGB2BGR)
    
    # 6. Gaussian noise
    noisy = original.copy().astype(np.float32)
    noise = np.random.normal(0, 10, original.shape)
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    
    # 7. Blur
    blurred = cv2.GaussianBlur(original, (5, 5), 0)
    
    # 8. Hue shift (operate in HSV space)
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 30) % 180  # Shift hue
    hue_shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Visualization
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    images = [
        (cv2.cvtColor(original, cv2.COLOR_BGR2RGB), "Original Image"),
        (cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), "Rotation (90°)"),
        (cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB), "Horizontal Flip"),
        (cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB), "Random Crop"),
        (cv2.cvtColor(brightness, cv2.COLOR_BGR2RGB), "Brightness Adjustment (1.5x)"),
        (cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB), "Contrast Adjustment"),
        (cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB), "Gaussian Noise"),
        (cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB), "Blur"),
        (cv2.cvtColor(hue_shifted, cv2.COLOR_BGR2RGB), "Hue Shift"),
    ]
    
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    print("Visualized various data augmentation techniques")
    print("\nBy combining these techniques, hundreds to thousands of variations can be generated from a single image")
    

### Preprocessing Pipeline with PyTorch Transforms
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - pillow>=10.0.0
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    """
    Example: Preprocessing Pipeline with PyTorch Transforms
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    
    # Create sample image
    sample_np = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    sample_pil = Image.fromarray(sample_np)
    
    print("=== Preprocessing Pipeline with PyTorch Transforms ===\n")
    
    # Training transformation pipeline
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transformation pipeline
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    train_tensor = train_transform(sample_pil)
    val_tensor = val_transform(sample_pil)
    
    print("Training transformation:")
    print(f"  Input: PIL Image {sample_pil.size}")
    print(f"  Output: Tensor {train_tensor.shape}")
    print(f"  Data type: {train_tensor.dtype}")
    print(f"  Value range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
    
    print("\nValidation transformation:")
    print(f"  Input: PIL Image {sample_pil.size}")
    print(f"  Output: Tensor {val_tensor.shape}")
    print(f"  Value range: [{val_tensor.min():.3f}, {val_tensor.max():.3f}]")
    
    # Simulate batch processing
    batch_size = 4
    batch_tensors = [train_transform(sample_pil) for _ in range(batch_size)]
    batch = torch.stack(batch_tensors)
    
    print(f"\nBatch processing:")
    print(f"  Batch size: {batch_size}")
    print(f"  Batch tensor shape: {batch.shape}")
    print(f"  → [Batch, Channels, Height, Width]")
    
    # Individual transformation examples
    print("\n=== Individual Transformation Details ===")
    
    # ToTensor only
    to_tensor = transforms.ToTensor()
    tensor_only = to_tensor(sample_pil)
    print(f"\n1. ToTensor:")
    print(f"   PIL (H, W, C) → Tensor (C, H, W)")
    print(f"   Value range: [0, 255] → [0.0, 1.0]")
    print(f"   Shape change: {sample_pil.size} → {tensor_only.shape}")
    
    # Effect of Normalize
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalized = normalize(tensor_only)
    print(f"\n2. Normalize (mean=0.5, std=0.5):")
    print(f"   Mean before transformation: {tensor_only.mean():.3f}")
    print(f"   Mean after transformation: {normalized.mean():.3f}")
    print(f"   Value range: [0.0, 1.0] → [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    print("\nPreprocessing pipeline construction complete")
    

* * *

## Summary

In this chapter, we learned the fundamentals of image processing.

### Key Points

  * **Digital images** are represented as arrays of pixels, handled in HWC or CHW format
  * **Color spaces** (RGB, HSV, LAB) should be selected based on application
  * **Basic operations** (resizing, rotation, color conversion) are important for image analysis preprocessing
  * **Filtering** (smoothing, edge detection) extracts image features
  * **Features** (corners, SIFT, HOG) form the foundation of object recognition
  * **Preprocessing and data augmentation** are essential for machine learning model performance improvement

### Preview of Next Chapter

Chapter 2 will cover the following topics:

  * Fundamentals of Convolutional Neural Networks (CNN)
  * Mechanisms of convolutional and pooling layers
  * Representative CNN architectures (LeNet, AlexNet)
  * CNN implementation and image classification with PyTorch

* * *

## Exercises

**Exercise 1: Understanding Color Spaces**

**Problem** : Convert an RGB image to HSV space and extract regions of a specific color (e.g., red).

**Hint** :

  * In HSV space, colors can be specified more easily by hue
  * Red hue range: 0-10 and 170-180 (OpenCV)

**Solution Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Solution Example:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    
    # Create RGB image (red, green, blue regions)
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:, 0:100] = [0, 0, 255]    # Red (BGR)
    image[:, 100:200] = [0, 255, 0]  # Green
    image[:, 200:300] = [255, 0, 0]  # Blue
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define red color range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply mask
    red_only = cv2.bitwise_and(image, image, mask=red_mask)
    
    print(f"Number of red region pixels: {np.sum(red_mask > 0)}")
    

**Exercise 2: Comparing Interpolation Methods**

**Problem** : When upscaling a 100×100 image to 500×500, compare the processing time and visual quality of four interpolation methods: NEAREST, LINEAR, CUBIC, and LANCZOS.

**Solution Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Solution Example:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 5-10 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import time
    
    # Test image
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    methods = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'LANCZOS': cv2.INTER_LANCZOS4
    }
    
    print("Processing time comparison for interpolation methods:")
    for name, method in methods.items():
        start = time.time()
        resized = cv2.resize(image, (500, 500), interpolation=method)
        elapsed = time.time() - start
        print(f"  {name:10s}: {elapsed*1000:.2f} ms")
    

**Exercise 3: Implementing a Custom Filter**

**Problem** : Implement convolution operation manually using the following kernel.
    
    
    Sharpening kernel:
     0  -1   0
    -1   5  -1
     0  -1   0
    

**Solution Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    import numpy as np
    import cv2
    
    def custom_convolution(image, kernel):
        """Custom convolution function"""
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        pad_h, pad_w = ker_h // 2, ker_w // 2
    
        # Padding
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
        # Output image
        output = np.zeros_like(image)
    
        # Convolution
        for i in range(img_h):
            for j in range(img_w):
                region = padded[i:i+ker_h, j:j+ker_w]
                output[i, j] = np.sum(region * kernel)
    
        return np.clip(output, 0, 255).astype(np.uint8)
    
    # Sharpening kernel
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    
    # Test
    test_image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    sharpened = custom_convolution(test_image, sharpen_kernel)
    
    print("Implemented custom convolution")
    print(f"Input: {test_image.shape}")
    print(f"Output: {sharpened.shape}")
    

**Exercise 4: Applying PyTorch Transforms**

**Problem** : Create a data augmentation pipeline that meets the following requirements:

  * 80% probability of horizontal flip
  * Randomly adjust brightness and contrast by ±20%
  * Random rotation of ±10 degrees
  * Finally resize to 224×224
  * Apply ImageNet standardization

**Solution Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - torchvision>=0.15.0
    
    """
    Example: Solution Example:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    from torchvision import transforms
    
    augmentation_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print("Created data augmentation pipeline")
    print(f"Number of transformations: {len(augmentation_pipeline.transforms)}")
    

**Exercise 5: Applying Edge Detection**

**Problem** : Use Canny edge detection to identify rectangular regions in an image and calculate their area.

**Solution Example** :
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Solution Example:
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: ~5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    
    # Create image with rectangles
    image = np.zeros((300, 400), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (200, 150), 255, -1)
    cv2.rectangle(image, (250, 100), (350, 250), 255, -1)
    
    # Canny edge detection
    edges = cv2.Canny(image, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Number of contours detected: {len(contours)}")
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        print(f"Contour{i+1}: Area={area:.0f}, Perimeter={perimeter:.2f}")
    

* * *

## References

  * [OpenCV Documentation](<https://docs.opencv.org/>) \- Official documentation
  * [Pillow (PIL) Documentation](<https://pillow.readthedocs.io/>) \- Python Imaging Library
  * [torchvision.transforms](<https://pytorch.org/vision/stable/transforms.html>) \- PyTorch image transformations
  * Szeliski, R. (2010). _Computer Vision: Algorithms and Applications_. Springer.
  * Gonzalez, R. C., & Woods, R. E. (2018). _Digital Image Processing_ (4th ed.). Pearson.
  * Bradski, G., & Kaehler, A. (2008). _Learning OpenCV_. O'Reilly Media.

* * *
