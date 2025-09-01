// script.js (Versi Final Lengkap)

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
const livePredictionDiv = document.getElementById("live-prediction-box");
const suggestionLine = document.getElementById("suggestion-line");
const debugOutput = document.getElementById("debug-output");
const resetButton = document.getElementById("reset-button");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("video-overlay");

// --- Variabel Global ---
let handLandmarker, customTfliteModel, bigramModel, trigramModel;
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

// --- Variabel State ---
let currentSentence = "";
let lastPredictedLetter = "";
let lastAddedLetter = "";
let predictionCounter = 0;
const PREDICTION_THRESHOLD = 50;

let inactivityTimer;
let spaceTimer;
let handPresent = false;
let isShowingPrompt = false; // <-- VARIABEL BARU

// --- Fungsi Logging ---
function logToHTML(message) {
  if (!debugOutput) return;
  debugOutput.innerHTML += `> ${message}<br>`;
  debugOutput.scrollTop = debugOutput.scrollHeight;
}

// --- FUNGSI INISIALISASI ---
async function initHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "./models/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
  logToHTML("Hand Landmarker siap.");
}

async function loadTFLiteModel() {
  tflite.setWasmPath(
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/"
  );
  customTfliteModel = await tflite.loadTFLiteModel(
    "./models/model_final_tf216.tflite"
  );
  logToHTML("Model TFLite kustom siap.");
}

async function loadNgramModels() {
  try {
    const bigramResponse = await fetch("./models/bigram_model.json");
    bigramModel = await bigramResponse.json();
    const trigramResponse = await fetch("./models/trigram_model.json");
    trigramModel = await trigramResponse.json();
    logToHTML("Model N-Gram (Bigram & Trigram) siap.");
  } catch (e) {
    logToHTML(`Error memuat model N-Gram: ${e}`);
  }
}

async function setupCamera() {
  logToHTML("Setup kamera...");
  try {
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "environment",
      },
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    logToHTML("Izin kamera didapatkan, stream diterima.");

    video.srcObject = stream;
    video.addEventListener("playing", () => {
      logToHTML("Video berhasil diputar!");
      if (overlay) overlay.style.display = "none";
    });

    video.addEventListener("loadeddata", () => {
      logToHTML("Data kamera dimuat, mencoba memulai video...");
      video.play().catch((e) => {
        logToHTML(`Error saat play() video: ${e.message}`);
      });
      logToHTML("Memulai deteksi...");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      predictWebcam();
    });
  } catch (err) {
    logToHTML(`ERROR SAAT GETUSERMEDIA: ${err.name} - ${err.message}`);
  }
}

// --- FUNGSI LOOP & PREDIKSI ---
function predictWebcam() {
  if (handLandmarker && customTfliteModel && video.readyState >= 4) {
    const results = handLandmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.landmarks.length > 0) {
      handPresent = true;
      clearTimeout(spaceTimer);
      resetInactivityTimer();
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
    } else {
      if (handPresent) {
        handPresent = false;
        startSpaceTimer();
      }
    }
  }
  window.requestAnimationFrame(predictWebcam);
}
function processPrediction(newPrediction) {
  if (!newPrediction) return;

  if (isShowingPrompt) {
    outputDiv.innerText = "";
    isShowingPrompt = false;

    // TAMBAHKAN BARIS INI untuk menampilkan kembali kotak prediksi
    livePredictionDiv.style.display = "block";
  }

  if (newPrediction === lastPredictedLetter) {
    predictionCounter++;
  } else {
    predictionCounter = 1;
    lastPredictedLetter = newPrediction;
  }

  livePredictionDiv.innerText = lastPredictedLetter || "";

  if (
    predictionCounter >= PREDICTION_THRESHOLD &&
    newPrediction !== lastAddedLetter
  ) {
    currentSentence += newPrediction;
    lastAddedLetter = newPrediction;
    outputDiv.innerText = currentSentence;
    updateWordSuggestions();
  }
}

function updateWordSuggestions() {
  if (!bigramModel || !trigramModel) return;

  const words = currentSentence.trim().toLowerCase().split(" ").filter(Boolean);
  let suggestions = [];

  if (words.length >= 2) {
    const lastTwo = words.slice(-2).join(" ");
    if (trigramModel[lastTwo]) {
      suggestions = Object.entries(trigramModel[lastTwo])
        .sort((a, b) => b[1] - a[1])
        .map((e) => e[0]);
    }
  }

  if (suggestions.length === 0 && words.length >= 1) {
    const lastOne = words[words.length - 1];
    if (bigramModel[lastOne]) {
      suggestions = Object.entries(bigramModel[lastOne])
        .sort((a, b) => b[1] - a[1])
        .map((e) => e[0]);
    }
  }

  const topSuggestions = suggestions.slice(0, 3);
  suggestionLine.innerHTML = "";
  if (topSuggestions.length > 0) {
    const suggestionText = topSuggestions
      .map((s) => s.toUpperCase())
      .join(" &nbsp; | &nbsp; ");
    suggestionLine.innerHTML = `<strong>Prediksi Selanjutnya :</strong> ${suggestionText}`;
  }
}

// --- FUNGSI TIMER & RESET ---
function resetInactivityTimer() {
  clearTimeout(inactivityTimer);
  inactivityTimer = setTimeout(() => {
    logToHTML("Tidak ada aktivitas selama 10 detik, mereset kalimat.");
    resetSentence();
  }, 10000);
}

function startSpaceTimer() {
  clearTimeout(spaceTimer);
  spaceTimer = setTimeout(() => {
    addSpace();
  }, 5000);
}

function addSpace() {
  if (currentSentence.length > 0 && !currentSentence.endsWith(" ")) {
    logToHTML("Tangan tidak ada, spasi ditambahkan.");
    currentSentence += " ";
    outputDiv.innerText = currentSentence;
    lastAddedLetter = "";
    lastPredictedLetter = "";
    updateWordSuggestions();
  }
}

function resetSentence() {
  logToHTML("Kalimat direset.");
  currentSentence = "";
  lastAddedLetter = "";
  lastPredictedLetter = "";
  predictionCounter = 0;
  outputDiv.innerText = "Arahkan tangan";
  livePredictionDiv.innerText = "";
  suggestionLine.innerHTML = "";
  isShowingPrompt = true; // <-- PERUBAHAN DI SINI

  // TAMBAHKAN BARIS INI untuk menyembunyikan kotak prediksi secara paksa
  livePredictionDiv.style.display = "none";

  clearTimeout(inactivityTimer);
  clearTimeout(spaceTimer);
}

// --- FUNGSI MENGGAMBAR ---
function drawLandmarks(landmarks) {
  const pointPaint = { color: "#3399FF", radius: 5 };
  const linePaint = { color: "#FFFFFF", lineWidth: 3 };
  for (const landmark of landmarks) {
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

// --- TITIK AWAL APLIKASI ---
async function main() {
  logToHTML("Memulai inisialisasi...");
  await Promise.all([
    initHandLandmarker(),
    loadTFLiteModel(),
    loadNgramModels(),
  ]);
  resetButton.addEventListener("click", resetSentence);
  await setupCamera();
  resetSentence();
}

// Jalankan semuanya
main();

// --- TAMBAHKAN BARIS INI UNTUK MEMBUAT FUNGSI BISA DIAKSES FLUTTER ---
window.resetSentence = resetSentence;
