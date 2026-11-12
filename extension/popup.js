// SentiPulse Chrome Extension Popup Controller
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const tabs = document.querySelectorAll(".nav-tab");
  const tabContents = document.querySelectorAll(".tab-content");
  const connectionStatus = document.getElementById("connection-status");

  // Tab 1: Single Analyze
  const feedbackInput = document.getElementById("feedback-input");
  const btnAnalyze = document.getElementById("btn-analyze");
  const resultBox = document.getElementById("result-box");
  const resSentiment = document.getElementById("res-sentiment");
  const resConfidence = document.getElementById("res-confidence");
  const escalationAlert = document.getElementById("escalation-alert");
  const barPos = document.getElementById("bar-pos");
  const barNeu = document.getElementById("bar-neu");
  const barNeg = document.getElementById("bar-neg");
  const lblPos = document.getElementById("lbl-pos");
  const lblNeu = document.getElementById("lbl-neu");
  const lblNeg = document.getElementById("lbl-neg");
  const resLatency = document.getElementById("res-latency");

  // Tab 2: Page Scraper
  const btnScrapePage = document.getElementById("btn-scrape-page");
  const pageResults = document.getElementById("page-results");
  const pageTotal = document.getElementById("page-total");
  const pagePosPct = document.getElementById("page-pos-pct");
  const pageNeuPct = document.getElementById("page-neu-pct");
  const pageNegPct = document.getElementById("page-neg-pct");
  const complaintsList = document.getElementById("page-complaints-list");

  // Tab 3: Settings
  const endpointUrlInput = document.getElementById("endpoint-url");
  const apiKeyInput = document.getElementById("api-key");
  const btnSaveSettings = document.getElementById("btn-save-settings");
  const settingsStatus = document.getElementById("settings-status");

  let apiUrl = "http://127.0.0.1:5000";
  let apiKey = "";

  function getHeaders() {
    const h = { "Content-Type": "application/json" };
    if (apiKey) h["X-API-Key"] = apiKey;
    return h;
  }

  // Load saved endpoint URL & API Key
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.sync) {
    chrome.storage.sync.get(["sentipulse_api_url", "sentipulse_api_key"], (res) => {
      if (res.sentipulse_api_url) {
        apiUrl = res.sentipulse_api_url;
        endpointUrlInput.value = apiUrl;
      }
      if (res.sentipulse_api_key) {
        apiKey = res.sentipulse_api_key;
        if (apiKeyInput) apiKeyInput.value = apiKey;
      }
      checkHealth();
    });
  } else {
    checkHealth();
  }

  // Tab Navigation
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tabContents.forEach((tc) => tc.classList.remove("active"));

      tab.classList.add("active");
      const target = document.getElementById(tab.getAttribute("data-tab"));
      if (target) target.classList.add("active");
    });
  });

  // Health check
  async function checkHealth() {
    try {
      const res = await fetch(`${apiUrl}/health`);
      if (res.ok) {
        connectionStatus.textContent = "Connected";
        connectionStatus.className = "status-indicator status-online";
      } else {
        throw new Error("Degraded");
      }
    } catch {
      connectionStatus.textContent = "Offline";
      connectionStatus.className = "status-indicator status-offline";
    }
  }

  // 1. Analyze single review
  btnAnalyze.addEventListener("click", async () => {
    const text = feedbackInput.value.trim().slice(0, 5000);
    if (!text) return;

    btnAnalyze.disabled = true;
    btnAnalyze.textContent = "Analyzing...";

    try {
      const t0 = performance.now();
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: getHeaders(),
        body: JSON.stringify({ text }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const latencyMs = Math.round(performance.now() - t0);

      const sentiment = data.sentiment || "Neutral";
      const confidence = data.confidence || 0.0;
      const probs = data.probabilities || { Positive: 0, Neutral: 0, Negative: 0 };

      // Update UI
      resSentiment.textContent = sentiment;
      resSentiment.className = `sentiment-tag tag-${sentiment.toLowerCase()}`;
      resConfidence.textContent = `${(confidence * 100).toFixed(1)}%`;

      const posW = Math.round(probs.Positive * 100);
      const neuW = Math.round(probs.Neutral * 100);
      const negW = Math.round(probs.Negative * 100);

      barPos.style.width = `${posW}%`;
      barNeu.style.width = `${neuW}%`;
      barNeg.style.width = `${negW}%`;

      lblPos.textContent = `${posW}%`;
      lblNeu.textContent = `${neuW}%`;
      lblNeg.textContent = `${negW}%`;

      resLatency.textContent = `Latency: ${latencyMs} ms • ${data.model_version || "ONNX INT8"}`;

      // Escalation alert for high-confidence negative
      if (sentiment === "Negative" && confidence >= 0.85) {
        escalationAlert.style.display = "block";
      } else {
        escalationAlert.style.display = "none";
      }

      resultBox.style.display = "block";
    } catch (err) {
      alert(`Inference failed: ${err.message}. Check that SentiPulse server is running at ${apiUrl}`);
    } finally {
      btnAnalyze.disabled = false;
      btnAnalyze.textContent = "Analyze Sentiment";
    }
  });

  // 2. Page Review Scraper
  btnScrapePage.addEventListener("click", async () => {
    btnScrapePage.disabled = true;
    btnScrapePage.textContent = "Scanning active tab...";

    if (typeof chrome === "undefined" || !chrome.tabs) {
      alert("Page scraper requires Chromium browser extension environment.");
      btnScrapePage.disabled = false;
      btnScrapePage.textContent = "Inspect Page Reviews";
      return;
    }

    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) throw new Error("No active tab found");

      chrome.tabs.sendMessage(tab.id, { action: "SCRAPE_REVIEWS" }, async (response) => {
        if (!response || !response.reviews || response.reviews.length === 0) {
          alert("No reviews found on current page. Open an Amazon product, Trustpilot, or Google Reviews page.");
          btnScrapePage.disabled = false;
          btnScrapePage.textContent = "Inspect Page Reviews";
          return;
        }

        const reviews = response.reviews.slice(0, 25); // Limit batch to 25 reviews for snappy responsiveness
        btnScrapePage.textContent = `Analyzing ${reviews.length} reviews...`;

        let posCount = 0;
        let neuCount = 0;
        let negCount = 0;
        const criticalComplaints = [];

        for (const reviewText of reviews) {
          try {
            const res = await fetch(`${apiUrl}/predict`, {
              method: "POST",
              headers: getHeaders(),
              body: JSON.stringify({ text: reviewText.slice(0, 5000) }),
            });
            if (res.ok) {
              const data = await res.json();
              if (data.sentiment === "Positive") posCount++;
              else if (data.sentiment === "Neutral") neuCount++;
              else if (data.sentiment === "Negative") {
                negCount++;
                if (data.confidence >= 0.85) {
                  criticalComplaints.push({ text: reviewText, conf: data.confidence });
                }
              }
            }
          } catch (e) {
            console.warn("Failed item analysis:", e);
          }
        }

        const total = posCount + neuCount + negCount;
        pageTotal.textContent = total;
        pagePosPct.textContent = total ? `${Math.round((posCount / total) * 100)}%` : "0%";
        pageNeuPct.textContent = total ? `${Math.round((neuCount / total) * 100)}%` : "0%";
        pageNegPct.textContent = total ? `${Math.round((negCount / total) * 100)}%` : "0%";

        complaintsList.innerHTML = "";
        if (criticalComplaints.length === 0) {
          complaintsList.innerHTML = `<div style="color:#94a3b8; font-size:11px; padding:6px;">No high-severity complaints detected.</div>`;
        } else {
          criticalComplaints.forEach((c) => {
            const div = document.createElement("div");
            div.className = "complaint-item";
            div.textContent = `[${Math.round(c.conf * 100)}% Neg] "${c.text.slice(0, 180)}..."`;
            complaintsList.appendChild(div);
          });
        }

        pageResults.style.display = "block";
        btnScrapePage.disabled = false;
        btnScrapePage.textContent = "Re-inspect Page Reviews";
      });
    } catch (err) {
      alert(`Scraping error: ${err.message}`);
      btnScrapePage.disabled = false;
      btnScrapePage.textContent = "Inspect Page Reviews";
    }
  });

  // 3. Save Settings
  btnSaveSettings.addEventListener("click", () => {
    const val = endpointUrlInput.value.trim().replace(/\/+$/, "");
    if (val) apiUrl = val;
    apiKey = apiKeyInput ? apiKeyInput.value.trim() : "";

    const payload = { sentipulse_api_url: apiUrl, sentipulse_api_key: apiKey };

    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.sync) {
      chrome.storage.sync.set(payload, () => {
        settingsStatus.style.display = "block";
        setTimeout(() => (settingsStatus.style.display = "none"), 2000);
        checkHealth();
      });
    } else {
      settingsStatus.style.display = "block";
      setTimeout(() => (settingsStatus.style.display = "none"), 2000);
      checkHealth();
    }
  });
});

