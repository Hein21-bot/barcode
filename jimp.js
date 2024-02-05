const Jimp = require('jimp');
const fs = require('fs');
const path = require('path');

const sourceFolder = 'BarcodeScreenshots';
const targetFolder = 'Gray-Barcodes';
const distanceThreshold = 40; // Euclidean distance threshold


//RGB Cube Corner Colors
const allowedColors = [
    [0, 0, 0], [255, 255, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
];

// Euclidean distance function
function euclideanDistance(rgb1, rgb2) {
    return Math.sqrt(
        Math.pow(rgb1[0] - rgb2[0], 2) +
        Math.pow(rgb1[1] - rgb2[1], 2) +
        Math.pow(rgb1[2] - rgb2[2], 2)
    );
}

// Ensure the target folder exists
if (!fs.existsSync(targetFolder)) {
    fs.mkdirSync(targetFolder);
}

// Find the latest file in the source folder
const files = fs.readdirSync(sourceFolder).filter(file => file.endsWith('.png'));
const latestFile = files.sort().pop();

if (latestFile) {
    const sourcePath = path.join(sourceFolder, latestFile);
    const targetPath = path.join(targetFolder, `gray-${latestFile}`);

    Jimp.read(sourcePath)
        .then(image => {
            image.scan(0, 0, image.bitmap.width, image.bitmap.height, function(x, y, idx) {
                const red = this.bitmap.data[idx + 0];
                const green = this.bitmap.data[idx + 1];
                const blue = this.bitmap.data[idx + 2];

                let isColorWithinThreshold = false;
                for (let color of allowedColors) {
                    if (euclideanDistance([red, green, blue], color) <= distanceThreshold) {
                        isColorWithinThreshold = true;
                        break;
                    }
                }

                if (!isColorWithinThreshold) {
                    this.bitmap.data[idx + 0] = 128; // Red
                    this.bitmap.data[idx + 1] = 128; // Green
                    this.bitmap.data[idx + 2] = 128; // Blue
                }
            });

            return image.writeAsync(targetPath);
        })
        .then(() => {
            console.log(`Processed image saved to ${targetPath}`);
        })
        .catch(err => {
            console.error('Error processing image:', err);
        });
} else {
    console.log('No barcode screenshots found in the source folder.');
}