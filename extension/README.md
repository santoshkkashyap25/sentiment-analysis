# SentiPulse Review Inspector (Chrome & Chromium Extension)

> **Real-time customer sentiment intelligence, complaint triage, and automated review scraping powered by SentiPulse INT8 RoBERTa.**

---

## 🌟 Overview & Key Features

The **SentiPulse Review Inspector** is a Manifest V3 browser extension built for e-commerce store operators, product managers, and support desk teams:

* ⚡ **1-Click Review Scraping**: Automatically parses and extracts visible customer reviews directly from the active tab on **Amazon, Trustpilot, Google Reviews / Maps**, and generic feedback pages.
* 🛡️ **Calibrated Complaint Triage**: Surfaces high-priority negative complaints ($\ge 85\%$ confidence) with calibrated negative recall ($>90\%$), highlighting urgent churn risks and refund disputes.
* 🔬 **Quick Feedback Analyzer**: Manually paste or type any customer feedback string (up to 5,000 chars) for instant sentiment breakdown, probability bars, and latency metrics.
* 🌐 **Dual-Engine Connectivity**:
  * **Local Engine**: Connect to your self-hosted instance at `http://127.0.0.1:5000`.
  * **Render Cloud**: Connect directly to the production cloud service at `https://sentipulse.onrender.com`.
* 🔑 **Multi-Tenant API Keys**: Supports `sk_live_...` workspace API keys to route and isolate telemetry, query history, and drift tracking to your dedicated team workspace.

---

## 📦 How to Install & Load (Developer Mode)

Works with any Chromium browser (**Google Chrome, Microsoft Edge, Brave, Opera, Vivaldi**):

1. Open your browser and navigate to the extensions manager:
   * **Chrome**: `chrome://extensions/`
   * **Edge**: `edge://extensions/`
   * **Brave**: `brave://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner).
3. Click the **Load unpacked** button (top-left corner).
4. Select the `extension/` folder in this repository:
   ```text
   d:/Projects/sentiment-analysis/extension
   ```
5. Pin the **SentiPulse** icon to your browser toolbar for quick 1-click access!

---

## 🎯 How to Use

### 1. Configure the API Server
1. Click the SentiPulse icon in your toolbar.
2. Switch to the **Settings** tab.
3. Choose your backend:
   * Click **Localhost (5000)** if you are running `sentipulse serve` locally.
   * Click **Render Cloud** to connect to `https://sentipulse.onrender.com`.
4. *(Optional)* Paste your workspace API key (`sk_live_...`).
5. Click **Save Settings**. The status indicator in the top header will turn green (**Connected**).

### 2. Inspect Live Reviews on Any Website
1. Navigate to any Amazon product page (e.g. customer reviews section) or Trustpilot review page.
2. Open the extension and switch to the **Page Scraper** tab.
3. Click **Inspect Page Reviews**.
4. The extension extracts up to 50 visible reviews, runs them through the SentiPulse classification pipeline, and presents:
   * Overall positive / neutral / negative sentiment breakdown.
   * Total reviews evaluated.
   * Priority complaints list highlighting high-confidence negative feedback.

---

## 🚀 Packaging & Chrome Web Store Distribution

### Packaging as a Standalone ZIP (for GitHub Releases or Sideloading)
To distribute the extension to users without cloning the repository, generate a release `.zip`:

```bash
# In PowerShell / terminal from project root:
Compress-Archive -Path extension/* -DestinationPath dist/sentipulse-chrome-extension-v1.0.0.zip -Force
```

Users can download the `.zip`, extract it anywhere, and load unpacked in 10 seconds.

### Publishing to the Official Chrome Web Store
If you wish to publish to the public [Chrome Web Store](https://chrome.google.com/webstore):

1. **Create Developer Account**: Sign in to the [Chrome Developer Dashboard](https://chrome.google.com/webstore/devconsole/) (requires a one-time $5 Google registration fee).
2. **Upload Package**: Click **New Item** and upload `sentipulse-chrome-extension-v1.0.0.zip`.
3. **Store Listing Assets**:
   * **Icon**: The included `icons/icon128.png` (128x128).
   * **Screenshots**: At least 1 screenshot (1280x800 or 640x400) showing the popup and review scraper in action.
   * **Description**: Detailed description of the sentiment analysis capabilities.
   * **Privacy Policy**: A simple markdown or hosted page stating that review texts are only sent to the user-configured SentiPulse API for inference and are not sold or tracked.
4. **Submit for Review**: Google typically approves extensions within 24–48 hours.

---

## 🔒 Permissions & Privacy
* `activeTab` & `scripting`: Required strictly to read visible review DOM text when you explicitly click "Inspect Page Reviews".
* `storage`: Required to save your chosen API endpoint URL and API key locally in browser sync storage.
* **Zero Telemetry Leaks**: SentiPulse does not track browsing history, sell data, or monitor inactive tabs.
