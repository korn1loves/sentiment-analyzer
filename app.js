// app.js (ES module version using transformers.js for local sentiment classification)

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

// Global variables
let reviews = [];
let apiToken = ""; // kept for UI compatibility, but not used with local inference
let sentimentPipeline = null; // transformers.js text-classification pipeline

// --- ДОБАВЛЕНО: Ссылка на твой Google Script ---
const GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbwB5hFpJEecvcA1IblnhIUrAtWP6IUi2lRWQ715nkF5Z2BkAe561D4vv-BqfPH1gRCKgA/exec";

// DOM elements
const analyzeBtn = document.getElementById("analyze-btn");
const reviewText = document.getElementById("review-text");
const sentimentResult = document.getElementById("sentiment-result");
const loadingElement = document.querySelector(".loading");
const errorElement = document.getElementById("error-message");
const apiTokenInput = document.getElementById("api-token");
const statusElement = document.getElementById("status"); // optional status label for model loading

// Initialize the app
document.addEventListener("DOMContentLoaded", function () {
  loadReviews();
  analyzeBtn.addEventListener("click", analyzeRandomReview);
  apiTokenInput.addEventListener("change", saveApiToken);

  const savedToken = localStorage.getItem("hfApiToken");
  if (savedToken) {
    apiTokenInput.value = savedToken;
    apiToken = savedToken;
  }

  initSentimentModel();
});

// Initialize transformers.js sentiment model
async function initSentimentModel() {
  try {
    if (statusElement) {
      statusElement.textContent = "Loading sentiment model...";
    }

    sentimentPipeline = await pipeline(
      "text-classification",
      "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
    );

    if (statusElement) {
      statusElement.textContent = "Sentiment model ready";
    }
  } catch (error) {
    console.error("Failed to load sentiment model:", error);
    showError("Failed to load sentiment model. Please check your network connection and try again.");
    if (statusElement) {
      statusElement.textContent = "Model load failed";
    }
  }
}

// Load and parse the TSV file using Papa Parse
function loadReviews() {
  fetch("reviews_test.tsv")
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to load TSV file");
      }
      return response.text();
    })
    .then((tsvData) => {
      Papa.parse(tsvData, {
        header: true,
        delimiter: "\t",
        complete: (results) => {
          reviews = results.data
            .map((row) => row.text)
            .filter((text) => typeof text === "string" && text.trim() !== "");
          console.log("Loaded", reviews.length, "reviews");
        },
        error: (error) => {
          console.error("TSV parse error:", error);
          showError("Failed to parse TSV file: " + error.message);
        },
      });
    })
    .catch((error) => {
      console.error("TSV load error:", error);
      showError("Failed to load TSV file: " + error.message);
    });
}

function saveApiToken() {
  apiToken = apiTokenInput.value.trim();
  if (apiToken) {
    localStorage.setItem("hfApiToken", apiToken);
  } else {
    localStorage.removeItem("hfApiToken");
  }
}

// Analyze a random review
function analyzeRandomReview() {
  hideError();

  if (!Array.isArray(reviews) || reviews.length === 0) {
    showError("No reviews available. Please try again later.");
    return;
  }

  if (!sentimentPipeline) {
    showError("Sentiment model is not ready yet. Please wait a moment.");
    return;
  }

  const selectedReview = reviews[Math.floor(Math.random() * reviews.length)];

  reviewText.textContent = selectedReview;
  loadingElement.style.display = "block";
  analyzeBtn.disabled = true;
  sentimentResult.innerHTML = ""; 
  sentimentResult.className = "sentiment-result"; 

  analyzeSentiment(selectedReview)
    .then((result) => {
      displaySentiment(result);
      
      // --- ИЗМЕНЕНО: Сбор данных и отправка в таблицу ---
      const sentimentData = result[0][0];
      const label = sentimentData.label.toUpperCase();
      const score = (sentimentData.score * 100).toFixed(1);
      const sentimentString = `${label} (${score}%)`;
      
      logToSheets(selectedReview, sentimentString);
    })
    .catch((error) => {
      console.error("Error:", error);
      showError(error.message || "Failed to analyze sentiment.");
    })
    .finally(() => {
      loadingElement.style.display = "none";
      analyzeBtn.disabled = false;
    });
}

// --- ДОБАВЛЕНО: Функция для отправки данных в Google Таблицу ---
async function logToSheets(review, sentiment) {
  const payload = {
    ts_iso: new Date().toISOString(),
    review: review,
    sentiment: sentiment,
    meta: JSON.stringify({
      userAgent: navigator.userAgent,
      language: navigator.language,
      platform: navigator.platform,
      screenResolution: `${window.screen.width}x${window.screen.height}`
    })
  };

  try {
    // Используем mode: "no-cors", так как Google Scripts не всегда корректно отвечают на CORS-запросы
    await fetch(GOOGLE_SCRIPT_URL, {
      method: "POST",
      mode: "no-cors",
      cache: "no-cache",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload)
    });
    console.log("Data logged to Google Sheets successfully");
  } catch (error) {
    console.error("Failed to log data to Google Sheets:", error);
  }
}

async function analyzeSentiment(text) {
  if (!sentimentPipeline) {
    throw new Error("Sentiment model is not initialized.");
  }
  const output = await sentimentPipeline(text);
  if (!Array.isArray(output) || output.length === 0) {
    throw new Error("Invalid sentiment output from local model.");
  }
  return [output];
}

function displaySentiment(result) {
  let sentiment = "neutral";
  let score = 0.5;
  let label = "NEUTRAL";

  if (
    Array.isArray(result) &&
    result.length > 0 &&
    Array.isArray(result[0]) &&
    result[0].length > 0
  ) {
    const sentimentData = result[0][0];
    if (sentimentData && typeof sentimentData === "object") {
      label = typeof sentimentData.label === "string" ? sentimentData.label.toUpperCase() : "NEUTRAL";
      score = typeof sentimentData.score === "number" ? sentimentData.score : 0.5;

      if (label === "POSITIVE" && score > 0.5) {
        sentiment = "positive";
      } else if (label === "NEGATIVE" && score > 0.5) {
        sentiment = "negative";
      } else {
        sentiment = "neutral";
      }
    }
  }

  sentimentResult.classList.add(sentiment);
  sentimentResult.innerHTML = `
        <i class="fas ${getSentimentIcon(sentiment)} icon"></i>
        <span>${label} (${(score * 100).toFixed(1)}% confidence)</span>
    `;
}

function getSentimentIcon(sentiment) {
  switch (sentiment) {
    case "positive": return "fa-thumbs-up";
    case "negative": return "fa-thumbs-down";
    default: return "fa-question-circle";
  }
}

function showError(message) {
  errorElement.textContent = message;
  errorElement.style.display = "block";
}

function hideError() {
  errorElement.style.display = "none";
}