// script.js (Versi Final Lengkap dan Bersih)

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
const statusText = document.getElementById("status-text");
const resetButton = document.getElementById("reset-button");
const loaderOverlay = document.getElementById("loader-overlay");
const ctx = canvas.getContext("2d");

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
let inactivityTimer, spaceTimer;
let handPresent = false;
let isShowingPrompt = false;

// --- Fungsi Utilitas ---
function updateStatusText(message) {
  if (statusText) statusText.innerText = message;
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
  updateStatusText("Detector tangan siap.");
}

async function loadTFLiteModel() {
  tflite.setWasmPath(
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/"
  );
  customTfliteModel = await tflite.loadTFLiteModel(
    "./models/model_final_tf216.tflite"
  );
  updateStatusText("Model penerjemah siap.");
}

async function loadNgramModels() {
  try {
    const [bigramResponse, trigramResponse] = await Promise.all([
      fetch("./models/bigram_model.json"),
      fetch("./models/trigram_model.json"),
    ]);
    bigramModel = await bigramResponse.json();
    trigramModel = await trigramResponse.json();
    updateStatusText("Model prediksi kata siap.");
  } catch (e) {
    updateStatusText(`Gagal memuat N-Gram: ${e.message}`);
  }
}

async function setupCamera() {
  updateStatusText("Membuka kamera...");
  try {
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "environment",
      },
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    updateStatusText("Izin kamera didapatkan.");
    video.srcObject = stream;

    video.addEventListener("loadeddata", () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      video
        .play()
        .catch((e) => updateStatusText(`Error play video: ${e.message}`));
      predictWebcam();
    });

    video.addEventListener("playing", () => {
      if (loaderOverlay) {
        loaderOverlay.style.opacity = "0";
        setTimeout(() => {
          loaderOverlay.style.display = "none";
        }, 500);
      }
      setTimeout(() => {
        updateStatusText("Kamera Siap! Arahkan Tangan Anda.");
      }, 1000);
    });
  } catch (err) {
    updateStatusText(`Kamera Error: ${err.name}`);
  }
}

// --- FUNGSI LOOP & PREDIKSI ---
function predictWebcam() {
  if (!handLandmarker || !customTfliteModel || video.readyState < 4) {
    window.requestAnimationFrame(predictWebcam);
    return;
  }

  const results = handLandmarker.detectForVideo(video, performance.now());
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (results.landmarks && results.landmarks.length > 0) {
    if (!handPresent) handPresent = true;

    resetInactivityTimer();
    clearTimeout(spaceTimer);
    drawLandmarks(results.landmarks);

    const landmarks = results.landmarks[0];
    const wrist = landmarks[0];
    const normalizedLandmarks = [];
    landmarks.forEach((lm) => {
      normalizedLandmarks.push(lm.x - wrist.x, lm.y - wrist.y);
    });

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
      if (livePredictionDiv) livePredictionDiv.innerText = "";
      startSpaceTimer();
    }
  }
  window.requestAnimationFrame(predictWebcam);
}

// --- FUNGSI LOGIKA APLIKASI ---
function processPrediction(newPrediction) {
  if (!newPrediction) return;

  if (isShowingPrompt) {
    outputDiv.innerText = "";
    isShowingPrompt = false;
  }

  if (newPrediction === lastPredictedLetter) {
    predictionCounter++;
  } else {
    predictionCounter = 1;
    lastPredictedLetter = newPrediction;
  }

  if (livePredictionDiv)
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
  if (!bigramModel || !trigramModel || !suggestionLine) return;

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
    const lastOne = words.slice(-1)[0];
    if (bigramModel[lastOne]) {
      suggestions = Object.entries(bigramModel[lastOne])
        .sort((a, b) => b[1] - a[1])
        .map((e) => e[0]);
    }
  }

  const topSuggestions = suggestions.slice(0, 3);
  if (topSuggestions.length > 0) {
    const suggestionText = topSuggestions
      .map((s) => s.toUpperCase())
      .join(" &nbsp; | &nbsp; ");
    suggestionLine.innerHTML = `<strong>Prediksi Selanjutnya :</strong> ${suggestionText}`;
  } else {
    suggestionLine.innerHTML = "";
  }
}

// --- FUNGSI TIMER & RESET ---
function resetInactivityTimer() {
  clearTimeout(inactivityTimer);
  inactivityTimer = setTimeout(() => {
    updateStatusText("Reset karena tidak ada aktivitas.");
    resetSentence();
  }, 8000);
}

function startSpaceTimer() {
  clearTimeout(spaceTimer);
  spaceTimer = setTimeout(addSpace, 3000);
}

function addSpace() {
  if (currentSentence.length > 0 && !currentSentence.endsWith(" ")) {
    updateStatusText("Spasi ditambahkan otomatis.");
    currentSentence += " ";
    outputDiv.innerText = currentSentence;
    lastAddedLetter = "";
    lastPredictedLetter = "";
    updateWordSuggestions();
    setTimeout(() => {
      updateStatusText("Kamera Siap! Arahkan Tangan Anda.");
    }, 2000);
  }
}

function resetSentence() {
  currentSentence = "";
  lastAddedLetter = "";
  lastPredictedLetter = "";
  predictionCounter = 0;

  if (outputDiv) outputDiv.innerText = "Arahkan tangan";
  if (livePredictionDiv) livePredictionDiv.innerText = "";
  if (suggestionLine) suggestionLine.innerHTML = "";

  isShowingPrompt = true;
  clearTimeout(inactivityTimer);
  clearTimeout(spaceTimer);
  updateStatusText("Kalimat direset. Arahkan tangan Anda.");
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
  updateStatusText("Memulai inisialisasi...");
  await Promise.all([
    initHandLandmarker(),
    loadTFLiteModel(),
    loadNgramModels(),
  ]);
  resetButton.addEventListener("click", resetSentence);
  await setupCamera();
  resetSentence();
}

main();

window.resetSentence = resetSentence;
