const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
let model = null;

let isDrawing = false;
let lastPos = null;

const shapeNames = ["Circle", "Square", "Rectangle", "Triangle"];

ctx.fillStyle = "white";
ctx.fillRect(0, 0, 32, 32);

async function initializeModel() {
  try {
    model = await tf.loadLayersModel("./model/shape-model.json");

    console.log("Model loaded successfully");
    console.log("Model input shape:", model.inputs[0].shape);
  } catch (error) {
    console.error("Error loading model:", error);
    document.getElementById("predictBtn").disabled = true;
  }
}

function getPixelPosition(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;

  const x = Math.floor((clientX - rect.left) * scaleX);
  const y = Math.floor((clientY - rect.top) * scaleY);

  return { x, y };
}

function drawPixel(x, y) {
  if (x >= 0 && x < 32 && y >= 0 && y < 32) {
    ctx.fillStyle = "black";
    ctx.fillRect(x, y, 1, 1);
  }
}

function drawLine(x0, y0, x1, y1) {
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx - dy;

  let x = x0;
  let y = y0;

  while (true) {
    drawPixel(x, y);
    if (x === x1 && y === y1) break;

    const e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x += sx;
    }
    if (e2 < dx) {
      err += dx;
      y += sy;
    }
  }
}

function startDrawing(e) {
  e.preventDefault();
  isDrawing = true;
  const pos = getPixelPosition(e);
  lastPos = pos;
  drawPixel(pos.x, pos.y);
  hidePrediction();
}

function draw(e) {
  e.preventDefault();
  if (isDrawing) {
    const pos = getPixelPosition(e);
    if (lastPos) {
      drawLine(lastPos.x, lastPos.y, pos.x, pos.y);
    }
    lastPos = pos;
  }
}

function stopDrawing(e) {
  e.preventDefault();
  isDrawing = false;
  lastPos = null;
}

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

canvas.addEventListener("touchstart", startDrawing);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", stopDrawing);

function clearCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, 32, 32);
  hidePrediction();
}

function hidePrediction() {
  document.getElementById("predictionLabel").style.display = "none";
}

function getCanvasData() {
  const imageData = ctx.getImageData(0, 0, 32, 32);
  const pixels = imageData.data;

  const grayscalePixels = [];
  for (let i = 0; i < pixels.length; i += 4) {
    const gray = (255 - pixels[i]) / 255.0;
    grayscalePixels.push(gray);
  }

  return grayscalePixels;
}

async function predictShape() {
  const data = getCanvasData();

  if (data.every((pixel) => pixel === 0)) {
    alert("Please draw something first!");
    return;
  }

  if (!model) {
    alert("Model not loaded. Please ensure model files are available.");
    return;
  }

  const predictBtn = document.getElementById("predictBtn");
  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  try {
    const reshapedData = [];
    for (let i = 0; i < 32; i++) {
      reshapedData[i] = [];
      for (let j = 0; j < 32; j++) {
        reshapedData[i][j] = [data[i * 32 + j]];
      }
    }

    const tensor = tf.tensor4d([reshapedData]);
    const predictions = await model.predict(tensor);
    const predictionData = await predictions.data();

    displayPrediction(predictionData);

    tensor.dispose();
    predictions.dispose();
  } catch (error) {
    console.error("Prediction error:", error);
    alert("Error making prediction. Check console for details.");
  }

  predictBtn.disabled = false;
  predictBtn.textContent = "Predict Shape";
}

function displayPrediction(predictions) {
  const maxIndex = predictions.indexOf(Math.max(...predictions));
  const confidence = predictions[maxIndex] * 99.9;

  document.getElementById("predictionResult").textContent =
    shapeNames[maxIndex];
  document.getElementById("confidenceScore").textContent =
    `Confidence: ${confidence.toFixed(1)}%`;
  document.getElementById("predictionLabel").style.display = "block";
}

initializeModel();
