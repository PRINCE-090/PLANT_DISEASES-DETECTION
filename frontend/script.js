// ============================================================
// script.js — LeafScan Frontend Logic (Fixed)
// ============================================================

const API_URL = "http://127.0.0.1:5000/predict";

// ── DOM references ──────────────────────────────────────────
const fileInput = document.getElementById("fileInput");
const uploadZone = document.getElementById("uploadZone");
const idleState = document.getElementById("idleState");
const previewState = document.getElementById("previewState");
const previewImg = document.getElementById("previewImg");
const analyzeBtn = document.getElementById("analyzeBtn");
const browseBtn = document.getElementById("browseBtn");

const resultsPanel = document.getElementById("resultsPanel");
const statusBadge = document.getElementById("statusBadge");
const statusText = document.getElementById("statusText");
const predictionName = document.getElementById("predictionName");
const confidenceFill = document.getElementById("confidenceFill");
const confidencePct = document.getElementById("confidencePct");
const breakdownList = document.getElementById("breakdownList");

const errorPanel = document.getElementById("errorPanel");
const errorMsg = document.getElementById("errorMsg");

// selected image
let selectedFile = null;

// ── Prevent browser from navigating on drop ────────────────
["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
  window.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
});

// ── File select ─────────────────────────────────────────────
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

// ── Upload zone click ───────────────────────────────────────
uploadZone.addEventListener("click", () => {
  if (!selectedFile) fileInput.click();
});

browseBtn.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  fileInput.click();
});

// ── Drag & Drop ─────────────────────────────────────────────
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];

  if (file && file.type.startsWith("image/")) {
    handleFile(file);
  }
});

// ── Handle file ─────────────────────────────────────────────
function handleFile(file) {
  selectedFile = file;

  const reader = new FileReader();

  reader.onload = (e) => {
    previewImg.src = e.target.result;

    idleState.classList.add("hidden");
    previewState.classList.remove("hidden");

    analyzeBtn.disabled = false;
  };

  reader.readAsDataURL(file);

  hideResults();
  hideError();
}

// ── Analyze Button Click ───────────────────────────────────
analyzeBtn.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  analyze();
});

// ── Send image to Flask ─────────────────────────────────────
async function analyze() {
  if (!selectedFile) return;

  analyzeBtn.textContent = "Analyzing...";
  analyzeBtn.disabled = true;

  hideResults();
  hideError();

  try {
    const formData = new FormData();

    // Flask expects "image"
    formData.append("file", selectedFile);

    const response = await fetch(API_URL, {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || "Server error");
      return;
    }

    renderResults(data);

  } catch (error) {
    showError(
      "Cannot connect to Flask server.\n" +
      "Run backend using: python app.py"
    );
  }

  analyzeBtn.textContent = "Analyze Leaf";
  analyzeBtn.disabled = false;
}

// ── Render results ──────────────────────────────────────────
function renderResults(data) {

  const healthy = data.is_healthy;

  statusBadge.className =
    "status-badge " + (healthy ? "healthy" : "diseased");

  statusText.textContent =
    healthy ? "✓ Healthy Plant" : "⚠ Disease Detected";

  predictionName.textContent = data.prediction;

  const percent = Math.round(data.confidence * 100);

  confidenceFill.style.width = percent + "%";
  confidencePct.textContent = percent + "%";

  breakdownList.innerHTML = "";

  data.top5.forEach((item, index) => {

    const li = document.createElement("li");

    li.innerHTML = `
      <strong>${item.label}</strong>
      - ${Math.round(item.confidence * 100)}%
    `;

    breakdownList.appendChild(li);
  });

  resultsPanel.classList.remove("hidden");

  resultsPanel.scrollIntoView({
    behavior: "smooth"
  });
}

// ── Reset UI ────────────────────────────────────────────────
function resetAll() {

  selectedFile = null;

  fileInput.value = "";

  previewImg.src = "";

  idleState.classList.remove("hidden");
  previewState.classList.add("hidden");

  analyzeBtn.disabled = true;

  confidenceFill.style.width = "0%";

  hideResults();
  hideError();
}

// ── Utility functions ───────────────────────────────────────
function hideResults() {
  resultsPanel.classList.add("hidden");
}

function hideError() {
  errorPanel.classList.add("hidden");
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorPanel.classList.remove("hidden");
}