# SentiPulse Review Inspector (Chrome & Chromium Extension)

A lightweight browser extension powered by the **SentiPulse INT8 RoBERTa** inference engine for real-time customer sentiment intelligence, complaint triage, and review scraping.

---

## Features

* **Quick Feedback Analyzer:** Paste or select any text to inspect sentiment distribution, calibrated negative confidence, and customer churn risk.
* **1-Click Page Review Inspector:** Automatically scans the active browser tab on **Amazon, Trustpilot, or Google Reviews**, evaluates all visible customer reviews, and surfaces high-priority negative complaints.
* **Dual-Engine Connectivity:** Connects to your local SentiPulse server (`http://127.0.0.1:5000`) or production cloud endpoint (e.g., Render URL).

---

## Installation & Setup (Chrome / Edge / Brave)

1. Open your Chromium browser and navigate to `chrome://extensions/` (or `edge://extensions/`).
2. Toggle on **Developer mode** (top-right corner).
3. Click **Load unpacked** (top-left).
4. Select the `extension/` directory from this repository:
   ```
   d:/Projects/sentiment-analysis/extension
   ```
5. The **SentiPulse Review Inspector** icon will appear in your browser toolbar!

---

## Usage

1. **Verify Connection:** Ensure the local server is running (`sentipulse serve` or `python -m uvicorn src.api.app:app --port 5000`). The popup header will display green **Connected**.
2. **Scrape Reviews:** Navigate to any Amazon product page or Trustpilot review list, open the extension, switch to the **Page Scraper** tab, and click **Inspect Page Reviews**.
