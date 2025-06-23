const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
// const fs = require("fs");
// const path = require("path");

const app = express();
process.on("uncaughtException", (err) => {
  console.error("Uncaught Exception:", err);
});

process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled Rejection at:", promise, "reason:", reason);
});

const upload = multer({ storage: multer.memoryStorage() });

let model;

// Load model on server start
const loadModel = async () => {
  try {
    model = await tf.loadLayersModel("file://model/model.json");
    console.log("Model loaded!");
  } catch (err) {
    console.error("❌ Failed to load model:", err);
    process.exit(1); // Gracefully exit
  }
};

loadModel();

// Image pre-processing function
const preprocessImage = async (buffer) => {
  const imageTensor = tf.node
    .decodeImage(buffer, 3)
    .resizeBilinear([224, 224]) // Adjust size to match your model
    .expandDims()
    .toFloat()
    .div(255.0); // Normalize if required

  return imageTensor;
};
app.get("/", (req, res) => {
  res.send("Hello World!");
});
// API Endpoint
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    const imageBuffer = req.file.buffer;
    const inputTensor = await preprocessImage(imageBuffer);
    const prediction = await model.predict(inputTensor).data();

    // Return predictions
    const maxIndex = prediction.indexOf(Math.max(...prediction));
    const labels = ["A", "B", "C", "D", "E"];
    const type = labels[maxIndex];
    const value = prediction[maxIndex];

    res.json({ type, value });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Prediction failed" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
