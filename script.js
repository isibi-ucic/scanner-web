// script.js (Versi Final)

// --- LANGKAH 1: Impor semua library yang dibutuhkan ---
// Mengimpor ini akan membuat objek 'tf' dan 'tflite' tersedia secara global
import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js";
import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/tf-tflite.min.js";

// Mengimpor ini secara spesifik mengambil kelas yang kita butuhkan
import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js";

// Dapatkan referensi ke elemen HTML
const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const outputDiv = document.getElementById("output");
const ctx = canvas.getContext("2d");

// --- Variabel & Konstanta ---
let handLandmarker;
let customTfliteModel;
const LABELS = [
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
];

// ... (variabel untuk merangkai kata tetap sama) ...
let currentSentence = "",
  lastPredictedLetter = "",
  lastAddedLetter = "",
  predictionCounter = 0;
const PREDICTION_THRESHOLD = 10;

// --- FUNGSI UTAMA ---
async function main() {
  // 1. Inisialisasi MediaPipe Hand Landmarker
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarke...float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
  console.log("Hand Landmarker siap.");

  // --- LANGKAH 2: PERBAIKAN TFLITE (INI KUNCINYA) ---
  // Beritahu TFLite di mana harus mencari file pendukungnya (Wasm backend)
  tflite.setWasmPath(
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/"
  );

  // Muat model TFLite kustom Anda
  customTfliteModel = await tflite.loadTFLiteModel(
    "./models/model_final_tf216.tflite"
  );
  console.log("Model TFLite kustom siap.");
  outputDiv.innerText = "Arahkan tangan";

  // 3. Setup Kamera & Mulai Loop
  await setupCamera();
  predictWebcam();
}

// ... (Fungsi setupCamera, predictWebcam, processPrediction, dan drawLandmarks tetap sama persis seperti sebelumnya) ...
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  video.addEventListener("loadeddata", () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  });
}

function predictWebcam() {
  const results = handLandmarker.detectForVideo(video, performance.now());
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (results.landmarks.length > 0) {
    drawLandmarks(results.landmarks);
    const landmarks = results.landmarks[0];
    const wrist = landmarks[0];
    const normalizedLandmarks = [];
    for (const lm of landmarks) {
      normalizedLandmarks.push(lm.x - wrist.x);
      normalizedLandmarks.push(lm.y - wrist.y);
    }
    const inputTensor = tf.tensor2d([normalizedLandmarks], [1, 42]);
    const prediction = customTfliteModel.predict(inputTensor);
    const predictionData = prediction.dataSync();
    const highestProbIndex = predictionData.indexOf(
      Math.max(...predictionData)
    );
    const predictedLetter = LABELS[highestProbIndex];
    processPrediction(predictedLetter);
    inputTensor.dispose();
    prediction.dispose();
  }
  window.requestAnimationFrame(predictWebcam);
}
function processPrediction(newPrediction) {
  if (!newPrediction) return;
  if (newPrediction === lastPredictedLetter) {
    predictionCounter++;
  } else {
    predictionCounter = 1;
    lastPredictedLetter = newPrediction;
  }
  if (
    predictionCounter >= PREDICTION_THRESHOLD &&
    newPrediction !== lastAddedLetter
  ) {
    currentSentence += newPrediction;
    lastAddedLetter = newPrediction;
  }
  outputDiv.innerText = currentSentence + (lastPredictedLetter || "");
}
function drawLandmarks(landmarks) {
  for (const landmark of landmarks) {
    for (const connection of HandLandmarker.HAND_CONNECTIONS) {
      const start = landmark[connection.start];
      const end = landmark[connection.end];
      ctx.beginPath();
      ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
      ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 3;
      ctx.stroke();
    }
    for (const point of landmark) {
      ctx.beginPath();
      ctx.arc(
        point.x * canvas.width,
        point.y * canvas.height,
        5,
        0,
        2 * Math.PI
      );
      ctx.fillStyle = "#00FF00";
      ctx.fill();
    }
  }
}
// Jalankan semuanya
main();
