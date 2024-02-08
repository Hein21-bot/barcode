import cv2
import numpy as np
from pyzbar.pyzbar import decode

def remove_horizontal_scratches(image, kernel_size=(1, 40), inpaint_radius=3):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect horizontal lines, which are likely to be scratches
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    detect_horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Create a mask for detected scratches
    _, scratch_mask = cv2.threshold(detect_horizontal, 0, 255, cv2.THRESH_BINARY_INV)

    # Inpaint the scratches on the original image
    inpainted_image = cv2.inpaint(image, scratch_mask, inpaint_radius, cv2.INPAINT_TELEA)
    
    return inpainted_image

# Load the image
image_path = 'Gray-Barcodes/gray-barcode-screenshot1.png'
image = cv2.imread(image_path)
if image is None:
    print("Failed to load the image. Please check the file path and try again.")
    exit()

# Remove horizontal scratches from the image
cleaned_image = remove_horizontal_scratches(image)

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

# Save the cleaned image
output_path = 'test.png'
cv2.imwrite(output_path, cleaned_image)
print(f"Cleaned image saved to {output_path}")
