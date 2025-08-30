// script.js (Dengan Debug di HTML & Kamera Belakang)

// --- Impor library ---
import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js";
import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/tf-tflite.min.js";
import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js";

// --- Referensi Elemen HTML ---
const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const outputDiv = document.getElementById("output");
const debugOutput = document.getElementById("debug-output"); // Referensi baru
const ctx = canvas.getContext("2d");

// --- Fungsi Baru untuk Logging ke HTML ---
function logToHTML(message) {
  debugOutput.innerHTML += `> ${message}<br>`;
  // Auto-scroll ke pesan terbaru
  debugOutput.scrollTop = debugOutput.scrollHeight;
}

// ... (Variabel & Konstanta lainnya tetap sama)
let handLandmarker, customTfliteModel;
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
let currentSentence = "",
  lastPredictedLetter = "",
  lastAddedLetter = "",
  predictionCounter = 0;
const PREDICTION_THRESHOLD = 10;

// --- FUNGSI UTAMA ---
async function main() {
  logToHTML("Memulai inisialisasi...");

  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: "./hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
  logToHTML("Hand Landmarker siap.");

  tflite.setWasmPath(
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/"
  );
  customTfliteModel = await tflite.loadTFLiteModel(
    "./model_final_tf216.tflite"
  );
  logToHTML("Model TFLite kustom siap.");

  outputDiv.innerText = "Arahkan tangan";

  await setupCamera();
  // predictWebcam();
}
// Di dalam script.js

async function setupCamera() {
  logToHTML("Setup kamera...");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
      },
    });
    video.srcObject = stream;

    // Tambahkan event listener ini
    video.addEventListener("loadeddata", () => {
      logToHTML("Kamera berhasil dimuat. Memulai deteksi...");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // MULAI LOOP PREDIKSI DI SINI
      predictWebcam();
    });
  } catch (err) {
    logToHTML(`Error kamera: ${err.message}`);
  }
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

// --- FUNGSI UNTUK MENGGAMBAR LANDMARK ---
function drawLandmarks(landmarks) {
  // Definisikan kuas
  const pointPaint = { color: "#00FF00", radius: 5 };
  const linePaint = { color: "#FFFFFF", lineWidth: 3 };

  // Loop melalui setiap tangan yang terdeteksi (meskipun kita set hanya 1)
  for (const landmark of landmarks) {
    // Gambar garis koneksi terlebih dahulu
    for (const connection of HandLandmarker.HAND_CONNECTIONS) {
      const start = landmark[connection.start];
      const end = landmark[connection.end];
      ctx.beginPath();
      ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
      ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
      ctx.strokeStyle = linePaint.color;
      ctx.lineWidth = linePaint.lineWidth;
      ctx.stroke();
    }
    // Gambar titik sendi di atas garis
    for (const point of landmark) {
      ctx.beginPath();
      ctx.arc(
        point.x * canvas.width,
        point.y * canvas.height,
        pointPaint.radius,
        0,
        2 * Math.PI
      );
      ctx.fillStyle = pointPaint.color;
      ctx.fill();
    }
  }
}

function resetSentence() {
  logToHTML("Kalimat direset.");
  currentSentence = "";
  lastAddedLetter = "";
  lastPredictedLetter = "";
  predictionCounter = 0;
  // Update tampilan output
  outputDiv.innerText = "Arahkan tangan";
}

// Tambahkan event listener untuk tombol
const resetButton = document.getElementById("reset-button");
resetButton.addEventListener("click", resetSentence);

// Jalankan semuanya
main();
