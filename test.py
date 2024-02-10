import cv2
import numpy as np

def clean_barcode_lines(image):
    # Apply adaptive thresholding to the image to isolate the barcode
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # Use a horizontal kernel to detect and preserve horizontal lines (barcode lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
    
    # Use bitwise 'and' to preserve only the regions of the barcode
    cleaned_image = cv2.bitwise_and(adaptive_thresh, detected_lines)
    
    return cleaned_image

# Load the image
image_path = 'BarcodeScreenshots/barcode-screenshot1.png'
image = cv2.imread(image_path, 0)  # Load the image in grayscale mode

# Clean the barcode lines
cleaned_image = clean_barcode_lines(image)

# Save the cleaned image
output_path = 'path_to_output_image.png'
cv2.imwrite(output_path, cleaned_image)

# # Specify the path to your image and the output path
# image_path = 'BarcodeScreenshots/barcode-screenshot1.png'
# output_path = 'path_to_output_image.png'

# process_image(image_path, output_path)
