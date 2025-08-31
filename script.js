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
const debugOutput = document.getElementById("debug-output");
const suggestionsDiv = document.getElementById("suggestions");
const resetButton = document.getElementById("reset-button");
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
const PREDICTION_THRESHOLD = 50; // Huruf stabil setelah 10 frame

let inactivityTimer; // Timer untuk reset kalimat
let spaceTimer; // Timer untuk spasi
let handPresent = false; // Lacak keberadaan tangan

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
                facingMode: 'environment'
            }
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        logToHTML("Izin kamera didapatkan, stream diterima.");
        
        video.srcObject = stream;
        
        // Tambahkan event listener untuk melihat apakah video benar-benar bisa diputar
        video.addEventListener('playing', () => {
            logToHTML("Video berhasil diputar!");
        });

        // Event listener untuk saat data pertama dimuat
        video.addEventListener('loadeddata', () => {
            logToHTML("Data kamera dimuat, mencoba memulai video...");
            
            // --- INI BAGIAN KUNCINYA ---
            // Secara eksplisit panggil play() untuk memulai stream video
            video.play().catch(e => {
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
      clearTimeout(spaceTimer); // Batalkan timer spasi karena ada tangan
      resetInactivityTimer(); // Reset timer idle karena ada aktivitas

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

// --- FUNGSI LOGIKA APLIKASI ---
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
    updateWordSuggestions();
  }
  outputDiv.innerText =
    currentSentence.trim() + " " + (lastPredictedLetter || "");
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
  displaySuggestions(suggestions.slice(0, 3));
}

function displaySuggestions(suggestionList) {
  suggestionsDiv.innerHTML = "";
  for (const suggestion of suggestionList) {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerText = suggestion.toUpperCase();
    chip.onclick = () => {
      currentSentence =
        currentSentence.trim() + " " + suggestion.toUpperCase() + " ";
      lastAddedLetter = "";
      lastPredictedLetter = "";
      updateWordSuggestions();
    };
    suggestionsDiv.appendChild(chip);
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
  suggestionsDiv.innerHTML = "";
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
}

// Jalankan semuanya
main();
