const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const cv = require('@u4/opencv4nodejs'); // Updated import statement

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
            const image = cv.imdecode(buffer); // Assuming imdecode still exists

            // Convert the image to grayscale
            const grayImage = image.cvtColor(cv.COLOR_BGR2GRAY); // Check COLOR_BGR2GRAY availability

            // Apply Gaussian blur to reduce noise
            const blurredImage = grayImage.gaussianBlur(new cv.Size(5, 5), 0); // Assuming gaussianBlur still exists

            // Detect edges using Canny edge detector
            const edgeDetectedImage = blurredImage.canny(100, 200); // Assuming canny still exists

            // Create a canvas for drawing contours
            const contourCanvas = new cv.Mat(image.rows, image.cols, cv.CV_8UC3, new cv.Vec3(0, 0, 0)); // Assuming Mat and Vec3 still exist

            // Find and draw contours
            const contours = edgeDetectedImage.findContours(cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE); // Check for findContours availability

            for (const contour of contours) {
                if (contour.area > 1000) {
                    const color = new cv.Vec3(0, 255, 0); // Green, assuming Vec3 still exists
                    contourCanvas.drawContours([contour], -1, color, 2); // Assuming drawContours still exists
                }
            }

            // Save the processed image
            cv.imwrite(outputImagePath, contourCanvas); // Assuming imwrite still exists
            console.log(`Processed image saved to ${outputImagePath}`);
        })
        .catch(err => {
            console.error('Error loading image:', err);
        });
} else {
    console.log('No barcode screenshots found in the source folder.');
}
