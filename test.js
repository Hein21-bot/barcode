const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const cv = require('opencv4nodejs');

const sourceFolder = 'Gray-Barcodes';
const outputFolder = 'barcodeEdgeDetected';

// Ensure the output directory exists
if (!fs.existsSync(outputFolder)) {
    fs.mkdirSync(outputFolder);
}

// Get the list of image files from the source folder
const files = fs.readdirSync(sourceFolder).filter(file => file.startsWith('gray-barcode-screenshot') && file.endsWith('.png'));
const latestFile = files.sort().pop();

if (latestFile) {
    const imagePath = path.join(sourceFolder, latestFile);
    const outputImagePath = path.join(outputFolder, `edge-${latestFile}`);

    // Load the image using sharp
    sharp(imagePath)
        .toBuffer()
        .then(data => {
            // Convert the image data to a buffer
            const buffer = Buffer.from(data);

            // Read the image using OpenCV
            const image = cv.imdecode(buffer);

            // Convert the image to grayscale
            const grayImage = image.cvtColor(cv.COLOR_BGR2GRAY);

            // Apply Gaussian blur to reduce noise
            const blurredImage = grayImage.gaussianBlur(new cv.Size(5, 5), 0);

            // Detect edges using Canny edge detector
            const edgeDetectedImage = blurredImage.canny(100, 200);

            // Create a canvas for drawing contours
            const contourCanvas = new cv.Mat(image.rows, image.cols, cv.CV_8UC3, new cv.Vec3(0, 0, 0));

            // Find and draw contours
            const contours = edgeDetectedImage.findContours(cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

            for (const contour of contours) {
                if (contour.area > 1000) {
                    const color = new cv.Vec3(0, 255, 0); // Green
                    contourCanvas.drawContours([contour], -1, color, 2);
                }
            }

            // Save the processed image
            cv.imwrite(outputImagePath, contourCanvas);
            console.log(`Processed image saved to ${outputImagePath}`);
        })
        .catch(err => {
            console.error('Error loading image:', err);
        });
} else {
    console.log('No barcode screenshots found in the source folder.');
}
