import cv2
import os

source_folder = 'Gray-Barcodes'
output_folder = 'barcodeEdgeDetected'

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of image files from the source folder
files = [file for file in os.listdir(source_folder) if file.startswith('gray-barcode-screenshot') and file.endswith('.png')]
latest_file = sorted(files).pop()

if latest_file:
    image_path = os.path.join(source_folder, latest_file)
    output_path = os.path.join(output_folder, f"edge-{latest_file}")

    # Load the image
    image = cv2.imread(image_path)

    # Apply a blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Detect edges using a Canny detector
    edge_detected_image = cv2.Canny(blurred_image, 100, 200)

    # Dilate the edges to make the contours more visible
    dilated_image = cv2.dilate(edge_detected_image, None, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for rectangle-like contours and large enough area
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(contour) > 1000:
            color = (0, 255, 0)  # green
            cv2.drawContours(image, [approx], -1, color, 2)

    # Save the processed image
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to {output_path}")
else:
    print('No barcode screenshots found in the source folder.')