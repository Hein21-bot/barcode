//Authors: Teg Louis
//Date: 12/10/2023
//Note: This code take a screenshot of a barcode-generator for analysis.

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  // Define the folder and file name
  let folderName = 'BarcodeScreenshots';
  let fileNamePattern = `barcode-screenshot`;

  // Go to the specified page
  await page.goto('http://romogi.com/barcode-generator-updated.html');

  // Ensure the folder exists, create if it doesn't
  if (!fs.existsSync(folderName)) {
    fs.mkdirSync(folderName);
  }

  // Read filenames in the folder
  let files = fs.readdirSync(folderName);
  let numbers = files.map(file => {
    let match = file.match(/barcode-screenshot(\d+)\.png/);
    return match ? parseInt(match[1], 10) : 0;
  }).sort((a, b) => a - b);

  // Renumbering files to fill gaps
  let currentNumber = 1;
  for (let number of numbers) {
    if (number !== currentNumber) {
      let oldName = `${fileNamePattern}${number}.png`;
      let newName = `${fileNamePattern}${currentNumber}.png`;
      fs.renameSync(path.join(folderName, oldName), path.join(folderName, newName));
    }
    currentNumber++;
  }

  // Determine the new file name
  let fileName = `${fileNamePattern}${currentNumber}.png`;

  // Take screenshot and save it in the folder
  await page.screenshot({ path: path.join(folderName, fileName) });

  // Close the browser
  await browser.close();
})();