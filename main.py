import cv2
import os
from pyzbar.pyzbar import decode
import numpy as np

source_folder = 'Gray-Barcodes'
output_folder = 'barcodeEdgeDetected'

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of image files from the source folder
files = [file for file in os.listdir(source_folder) if file.lower().startswith('gray-barcode-screenshot') and file.lower().endswith('.png')]
latest_file = sorted(files, key=str.lower).pop() if files else None

def remove_scratches(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use a binary threshold to separate the barcode from the background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Use morphological operations to identify potential scratches
    # Assuming scratches are white (lighter) lines on the barcode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # Adjust size to the expected scratch thickness
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Inpaint only where the morphological operation detected lines
    inpainted = cv2.inpaint(image, morph, 7, cv2.INPAINT_TELEA)
    
    return inpainted

if latest_file:
    image_path = os.path.join(source_folder, latest_file)
    output_path = os.path.join(output_folder, f"edge-{latest_file}")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}. Please check the file path and try again.")
        exit()

    # Remove scratches from the image
    cleaned_image = remove_scratches(image)

    # Continue processing on the cleaned image
    # Convert to grayscale
    gray_image = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)

    # Apply a blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect edges using a Canny detector
    edge_detected_image = cv2.Canny(blurred_image, 100, 200)

    # Dilate the edges to make the contours more visible
    dilated_image = cv2.dilate(edge_detected_image, None, iterations=1)

    # Attempt to decode any barcodes in the cleaned image
    barcodes = decode(cleaned_image)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(cleaned_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        text = f"{barcode_data} ({barcode_type})"
        cv2.putText(cleaned_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print(f"Found barcode: Type: {barcode_type}, Data: {barcode_data}")

    # Save the processed image
    cv2.imwrite(output_path, cleaned_image)
    print(f"Processed image saved to {output_path}")
else:
    print('No barcode screenshots found in the source folder.')
