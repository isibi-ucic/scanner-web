// script.js (Versi Final Lengkap dengan N-Gram)

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
const PREDICTION_THRESHOLD = 50;

// --- Variabel Baru untuk Timer ---
let inactivityTimer;
// --- Fungsi Logging ---
function logToHTML(message) {
  if (!debugOutput) return;
  debugOutput.innerHTML += `> ${message}<br>`;
  debugOutput.scrollTop = debugOutput.scrollHeight;
}

// --- FUNGSI BARU UNTUK TIMER ---
function resetInactivityTimer() {
  // Hapus timer lama jika ada
  clearTimeout(inactivityTimer);

  // Atur timer baru selama 5 detik (5000 milidetik)
  inactivityTimer = setTimeout(() => {
    logToHTML("Tidak ada aktivitas selama 5 detik, mereset kalimat.");
    resetSentence();
  }, 5000);
}

// --- FUNGSI INISIALISASI ---
async function initHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "./models/hand_landmarker.task", // Pastikan file ini ada di folder models
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
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
    });
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
      logToHTML("Kamera berhasil dimuat. Memulai deteksi...");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      predictWebcam();
    });
  } catch (err) {
    logToHTML(`Error kamera: ${err.message}`);
  }
}

// --- FUNGSI LOOP & PREDIKSI ---
function predictWebcam() {
  if (handLandmarker) {
    const results = handLandmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.landmarks.length > 0) {
      drawLandmarks(results.landmarks);

      // Jika ada tangan terdeteksi, reset timer. Ini menandakan aktivitas.
      resetInactivityTimer();

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
  }
  window.requestAnimationFrame(predictWebcam);
}
// Di dalam file script.js

function processPrediction(newPrediction) {
  if (!newPrediction) return;

  if (newPrediction === lastPredictedLetter) {
    predictionCounter++;
  } else {
    predictionCounter = 1;
    lastPredictedLetter = newPrediction;
  }

  // Hanya update saran kata JIKA huruf baru ditambahkan ke kalimat
  if (
    predictionCounter >= PREDICTION_THRESHOLD &&
    newPrediction !== lastAddedLetter
  ) {
    currentSentence += newPrediction;
    lastAddedLetter = newPrediction;

    // Panggil update saran setelah kalimat diubah
    updateWordSuggestions();
  }

  // Update tampilan output utama secara real-time
  outputDiv.innerText = currentSentence + (lastPredictedLetter || "");
}

function displaySuggestions(suggestionList) {
  // --- TAMBAHKAN DEBUG DI SINI ---
  logToHTML(
    `DEBUG: Menampilkan ${suggestionList.length} saran: [${suggestionList.join(
      ", "
    )}]`
  );

  suggestionsDiv.innerHTML = ""; // Kosongkan saran sebelumnya

  for (const suggestion of suggestionList) {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerText = suggestion.toUpperCase();

    chip.onclick = () => {
      // Logika untuk menambahkan kata yang disarankan
      currentSentence += suggestion.toUpperCase() + " ";
      lastAddedLetter = ""; // Reset agar huruf berikutnya bisa ditambahkan
      lastPredictedLetter = ""; // Reset prediksi huruf

      // Panggil lagi update saran untuk kata berikutnya
      updateWordSuggestions();
    };
    suggestionsDiv.appendChild(chip);
  }
}
function updateWordSuggestions() {
  if (!bigramModel || !trigramModel) {
    logToHTML("DEBUG: Model N-Gram belum siap.");
    return;
  }

  const cleanSentence = currentSentence.toLowerCase().replace(/[^a-z\s]/g, "");
  const words = cleanSentence.split(" ").filter(Boolean); // filter(Boolean) untuk hapus string kosong

  logToHTML(`DEBUG: Menganalisis kalimat: [${words.join(" ")}]`);

  let suggestions = [];

  // Coba prediksi dengan Trigram dulu (konteks 2 kata)
  if (words.length >= 2) {
    const lastTwo = words.slice(-2).join(" ");
    logToHTML(`DEBUG: Mencari Trigram dengan kunci: "${lastTwo}"`);
    if (trigramModel[lastTwo]) {
      suggestions = Object.entries(trigramModel[lastTwo])
        .sort((a, b) => b[1] - a[1])
        .map((e) => e[0]);
      logToHTML(`DEBUG: Trigram ditemukan! Saran: ${suggestions.join(", ")}`);
    } else {
      logToHTML("DEBUG: Trigram tidak ditemukan.");
    }
  }

  // Jika Trigram tidak menghasilkan apa-apa, coba Bigram (konteks 1 kata)
  if (suggestions.length === 0 && words.length >= 1) {
    const lastOne = words[words.length - 1];
    logToHTML(`DEBUG: Mencari Bigram dengan kunci: "${lastOne}"`);
    if (bigramModel[lastOne]) {
      suggestions = Object.entries(bigramModel[lastOne])
        .sort((a, b) => b[1] - a[1])
        .map((e) => e[0]);
      logToHTML(`DEBUG: Bigram ditemukan! Saran: ${suggestions.join(", ")}`);
    } else {
      logToHTML("DEBUG: Bigram tidak ditemukan.");
    }
  }

  // Tampilkan 3 saran teratas
  displaySuggestions(suggestions.slice(0, 3));
}

// --- FUNGSI MENGGAMBAR & RESET ---
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

function resetSentence() {
  logToHTML("Kalimat direset.");
  currentSentence = "";
  lastAddedLetter = "";
  lastPredictedLetter = "";
  predictionCounter = 0;
  outputDiv.innerText = "Arahkan tangan";
  suggestionsDiv.innerHTML = "";
  // Hentikan juga timer saat mereset manual
  clearTimeout(inactivityTimer);
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
