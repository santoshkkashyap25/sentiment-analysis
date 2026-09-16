/**
 * Dashboard Logic - SentiPulse Core
 * Handles real-time sentiment inference, dual-mode tabs, Chart.js telemetry,
 * and Kolmogorov-Smirnov statistical drift simulation.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Navigation Tabs Elements
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  // Input & Inference Form Elements
  const form = document.getElementById('sentimentForm');
  const textarea = document.getElementById('feedbackInput');
  const charCount = document.getElementById('charCount');
  const btnClear = document.getElementById('btnClear');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnText = document.getElementById('btnText');
  const sampleChips = document.querySelectorAll('.sample-chip');
  const btnTestDriftSample = document.getElementById('btnTestDriftSample');

  // Prediction Result Elements
  const resultBox = document.getElementById('predictionResult');
  const sentimentBadge = document.getElementById('sentimentBadge');
  const confidenceVal = document.getElementById('confidenceVal');
  const latencyVal = document.getElementById('latencyVal');
  const barPos = document.getElementById('barPos');
  const barNeu = document.getElementById('barNeu');
  const barNeg = document.getElementById('barNeg');
  const pctPos = document.getElementById('pctPos');
  const pctNeu = document.getElementById('pctNeu');
  const pctNeg = document.getElementById('pctNeg');
  const nlpTokens = document.getElementById('nlpTokens');

  // KPI Summary Elements
  const kpiTotalRequests = document.getElementById('kpiTotalRequests');
  const kpiAvgLatency = document.getElementById('kpiAvgLatency');
  const kpiP50Latency = document.getElementById('kpiP50Latency');
  const kpiP95Latency = document.getElementById('kpiP95Latency');
  const kpiEngineTier = document.getElementById('kpiEngineTier');
  const kpiActiveBenchmark = document.getElementById('kpiActiveBenchmark');
  const modelNameDisplay = document.getElementById('modelNameDisplay');
  const tableCountBadge = document.getElementById('tableCountBadge');
  const tableBody = document.getElementById('inferencesTableBody');

  // Drift simulation elements
  const btnSimulateDrift = document.getElementById('btnSimulateDrift');
  const driftAlertFeedback = document.getElementById('driftAlertFeedback');
  const badgeDriftIndicator = document.getElementById('badgeDriftIndicator');

  // Initialize Charts
  let latencyChart = null;
  let sentimentChart = null;

  // 1. Dual-Mode Tab Switching
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTabId = btn.dataset.tab;

      tabButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });

      tabPanes.forEach(pane => {
        pane.classList.remove('active');
        pane.style.display = 'none';
      });

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');

      const targetPane = document.getElementById(targetTabId);
      if (targetPane) {
        targetPane.classList.add('active');
        targetPane.style.display = 'block';
      }

      // Re-trigger chart rendering if switching to MLOps tab
      if (targetTabId === 'tab-mlops') {
        setTimeout(() => {
          if (latencyChart) latencyChart.resize();
          if (sentimentChart) sentimentChart.resize();
        }, 50);
      }
    });
  });

  // 1b. Header Brand Click navigates to Sentiment Analyzer tab
  const brandHeaderBtn = document.getElementById('brandHeaderBtn') || document.querySelector('.header-brand');
  if (brandHeaderBtn) {
    brandHeaderBtn.addEventListener('click', () => {
      if (typeof window.goToSentimentAnalyzer === 'function') {
        window.goToSentimentAnalyzer();
      } else {
        const analyzerBtn = document.getElementById('tabBtnAnalyzer');
        if (analyzerBtn) analyzerBtn.click();
      }
    });
  }

  // 2. Setup Chart.js Telemetry
  function initCharts() {
    const canvasLatency = document.getElementById('latencyChart');
    const canvasSentiment = document.getElementById('sentimentChart');

    if (canvasLatency) {
      const ctxLatency = canvasLatency.getContext('2d');
      latencyChart = new Chart(ctxLatency, {
        type: 'line',
        data: {
          labels: ['#1', '#2', '#3', '#4', '#5', '#6', '#7', '#8', '#9', '#10'],
          datasets: [{
            label: 'Latency (ms)',
            data: [18, 14, 12, 15, 11, 14, 16, 12, 13, 11],
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.15)',
            fill: true,
            tension: 0.35,
            borderWidth: 2,
            pointRadius: 3,
            pointBackgroundColor: '#818cf8',
            pointHoverRadius: 5
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#1e293b',
              titleColor: '#cbd5e1',
              bodyColor: '#f8fafc',
              borderColor: 'rgba(255, 255, 255, 0.1)',
              borderWidth: 1,
              callbacks: {
                label: (context) => ` ${context.parsed.y} ms`
              }
            }
          },
          scales: {
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#64748b', font: { size: 10 } }
            },
            y: {
              grid: { color: 'rgba(255, 255, 255, 0.04)' },
              ticks: { color: '#64748b', font: { size: 10 } },
              beginAtZero: true
            }
          }
        }
      });
    }

    if (canvasSentiment) {
      const ctxSentiment = canvasSentiment.getContext('2d');
      sentimentChart = new Chart(ctxSentiment, {
        type: 'doughnut',
        data: {
          labels: ['Positive', 'Neutral', 'Negative'],
          datasets: [{
            data: [1, 1, 1],
            backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
            borderColor: '#0b0f19',
            borderWidth: 3
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '72%',
          plugins: {
            legend: {
              position: 'bottom',
              labels: {
                boxWidth: 10,
                color: '#94a3b8',
                font: { size: 11, family: "'Plus Jakarta Sans', sans-serif" },
                padding: 12
              }
            },
            tooltip: {
              backgroundColor: '#1e293b',
              titleColor: '#cbd5e1',
              bodyColor: '#f8fafc',
              borderColor: 'rgba(255, 255, 255, 0.1)',
              borderWidth: 1
            }
          }
        }
      });
    }
  }

  // 3. Character Counter
  if (textarea && charCount) {
    textarea.addEventListener('input', () => {
      charCount.textContent = textarea.value.length;
    });
  }

  // 4. Clear Button
  if (btnClear && textarea) {
    btnClear.addEventListener('click', () => {
      textarea.value = '';
      if (charCount) charCount.textContent = '0';
      if (resultBox) resultBox.style.display = 'none';
    });
  }

  // 5. Preset Sample Review Chips
  sampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      if (textarea && chip.dataset.text) {
        textarea.value = chip.dataset.text;
        if (charCount) charCount.textContent = textarea.value.length;
        triggerPrediction(textarea.value);
      }
    });
  });



  // 6. Form Submission
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const text = textarea ? textarea.value.trim() : '';
      if (text) {
        triggerPrediction(text);
      }
    });
  }

  // 7. Prediction Trigger
  async function triggerPrediction(text) {
    if (btnSpinner) btnSpinner.style.display = 'inline-block';
    if (btnAnalyze) btnAnalyze.disabled = true;

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        throw new Error(`API error ${response.status}`);
      }

      const data = await response.json();
      displayPredictionResult(data);
      fetchTelemetry();
    } catch (err) {
      console.error('Inference error:', err);
      alert('Prediction failed: ' + err.message);
    } finally {
      if (btnSpinner) btnSpinner.style.display = 'none';
      if (btnAnalyze) btnAnalyze.disabled = false;
    }
  }

  // 8. Render Prediction Output
  function displayPredictionResult(data) {
    if (!resultBox) return;
    resultBox.style.display = 'block';

    const sentiment = data.sentiment || 'Neutral';
    if (sentimentBadge) {
      sentimentBadge.textContent = sentiment;
      sentimentBadge.className = 'sentiment-badge-large';

      if (sentiment === 'Positive') {
        sentimentBadge.classList.add('badge-positive');
      } else if (sentiment === 'Negative') {
        sentimentBadge.classList.add('badge-negative');
      } else {
        sentimentBadge.classList.add('badge-neutral');
      }
    }

    const confPct = Math.round((data.confidence || 0) * 1000) / 10;
    if (confidenceVal) confidenceVal.textContent = `${confPct}%`;
    if (latencyVal) latencyVal.textContent = `${data.response_time_ms || 0} ms`;

    // Probability breakdown
    const probs = data.probabilities || { Positive: 0.33, Neutral: 0.34, Negative: 0.33 };
    const pPos = Math.round((probs.Positive || 0) * 100);
    const pNeu = Math.round((probs.Neutral || 0) * 100);
    const pNeg = Math.round((probs.Negative || 0) * 100);

    if (barPos) barPos.style.width = `${pPos}%`;
    if (barNeu) barNeu.style.width = `${pNeu}%`;
    if (barNeg) barNeg.style.width = `${pNeg}%`;

    if (pctPos) pctPos.textContent = `${pPos}%`;
    if (pctNeu) pctNeu.textContent = `${pNeu}%`;
    if (pctNeg) pctNeg.textContent = `${pNeg}%`;

    if (nlpTokens) {
      if (data.processed_text && data.processed_text.trim().length > 0) {
        nlpTokens.textContent = data.processed_text;
        nlpTokens.style.color = '';
        nlpTokens.style.fontStyle = 'normal';
      } else {
        nlpTokens.textContent = 'No sentiment keywords detected (input contained only stop words or symbols — neutral baseline applied)';
        nlpTokens.style.color = '#94a3b8';
        nlpTokens.style.fontStyle = 'italic';
      }
    }
  }

  // 9. Fetch Live Telemetry from SQLite
  async function fetchTelemetry() {
    try {
      const res = await fetch('/api/telemetry');
      if (!res.ok) return;
      const data = await res.json();

      // Update KPI cards
      if (kpiTotalRequests) kpiTotalRequests.textContent = data.total_predictions || 0;
      if (kpiAvgLatency) kpiAvgLatency.innerHTML = `${data.avg_latency_ms || 0} <span class="kpi-unit">ms</span>`;
      if (kpiP50Latency) kpiP50Latency.textContent = data.p50_latency_ms || 0;
      if (kpiP95Latency) kpiP95Latency.textContent = data.p95_latency_ms || 0;

      // Drift status
      const alerts = data.drift_alerts || [];
      const hasRecentDrift = alerts.some(a => a.type === 'data_drift');

      if (badgeDriftIndicator) {
        if (hasRecentDrift) {
          badgeDriftIndicator.textContent = 'Shift Alert';
          badgeDriftIndicator.className = 'badge-status-alert';
        } else {
          badgeDriftIndicator.textContent = 'Normal';
          badgeDriftIndicator.className = 'badge-status-ok';
        }
      }

      // Update Sentiment Chart
      if (sentimentChart && data.sentiment_counts) {
        const counts = data.sentiment_counts;
        const total = (counts.Positive || 0) + (counts.Neutral || 0) + (counts.Negative || 0);
        if (total > 0) {
          sentimentChart.data.datasets[0].data = [
            counts.Positive || 0,
            counts.Neutral || 0,
            counts.Negative || 0
          ];
          sentimentChart.update();
        }
      }

      // Update Latency Trend Chart
      if (latencyChart && data.recent_predictions && data.recent_predictions.length > 0) {
        const recent = data.recent_predictions.slice(0, 10).reverse();
        latencyChart.data.labels = recent.map((_, i) => `#${i + 1}`);
        latencyChart.data.datasets[0].data = recent.map(r => r.latency_ms);
        latencyChart.update();
      }

      // Render Inferences Table
      renderTable(data.recent_predictions || []);
    } catch (err) {
      console.warn('Telemetry sync error:', err);
    }
  }

  // 10. Fetch Server & Model Health Info
  async function fetchServerHealth() {
    try {
      const res = await fetch('/health');
      if (!res.ok) return;
      const data = await res.json();

      if (data.model_name && modelNameDisplay) {
        modelNameDisplay.textContent = data.model_name.includes('RoBERTa') ? 'RoBERTa INT8' : data.model_name;
      }
      if (data.benchmark && kpiActiveBenchmark) {
        const pct = Math.round((data.benchmark.f1_score || 0.846) * 1000) / 10;
        kpiActiveBenchmark.innerHTML = `${pct}% <span class="kpi-unit">F1</span>`;
      }
    } catch (err) {
      console.warn('Health check fetch error:', err);
    }
  }

  function renderTable(rows) {
    if (tableCountBadge) tableCountBadge.textContent = `${rows.length} logged`;
    if (!tableBody) return;

    if (!rows || rows.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="5" class="table-empty">No inference requests logged yet. Submit a review above!</td></tr>`;
      return;
    }

    tableBody.innerHTML = rows.map(r => {
      let pillClass = 'pill-neu';
      if (r.sentiment === 'Positive') pillClass = 'pill-pos';
      if (r.sentiment === 'Negative') pillClass = 'pill-neg';

      const timeStr = r.timestamp ? r.timestamp.split('T')[1]?.split('.')[0] || r.timestamp : '';
      const confPct = Math.round((r.confidence || 0) * 100);

      return `
        <tr>
          <td class="table-time">${timeStr}</td>
          <td class="table-text" title="${escapeHtml(r.text)}">${escapeHtml(r.text)}</td>
          <td><span class="table-sentiment-pill ${pillClass}">${r.sentiment}</span></td>
          <td style="font-family: 'JetBrains Mono', monospace;">${confPct}%</td>
          <td style="font-family: 'JetBrains Mono', monospace;">${r.latency_ms} ms</td>
        </tr>
      `;
    }).join('');
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // 11. Real Drift Check on Live Queries
  if (btnSimulateDrift) {
    btnSimulateDrift.addEventListener('click', async () => {
      btnSimulateDrift.disabled = true;
      btnSimulateDrift.innerHTML = `Evaluating Live Queries...`;

      try {
        const res = await fetch('/api/check-drift', { method: 'POST' });
        const data = await res.json();

        if (driftAlertFeedback) {
          driftAlertFeedback.style.display = 'block';
          if (data.status === 'insufficient_data') {
            driftAlertFeedback.className = 'drift-feedback drift-feedback-ok';
            driftAlertFeedback.innerHTML = `
              <strong>Collecting Sample Window:</strong><br>
              ${data.message}
            `;
          } else if (data.drift_detected) {
            driftAlertFeedback.className = 'drift-feedback drift-feedback-alert';
            driftAlertFeedback.innerHTML = `
              <strong>Statistical Drift Alert Triggered!</strong><br>
              ${data.drift_percentage}% of vocabulary features shifted across ${data.queries_evaluated} live reviews (KS p &lt; 0.05).<br>
              <em>Automated retraining alert logged to SQLite monitoring database.</em>
            `;
            if (badgeDriftIndicator) {
              badgeDriftIndicator.textContent = 'Shift Alert';
              badgeDriftIndicator.className = 'badge-status-alert';
            }
          } else {
            driftAlertFeedback.className = 'drift-feedback drift-feedback-ok';
            driftAlertFeedback.innerHTML = `
              <strong>Statistical Check Complete:</strong><br>
              ${data.message}
            `;
            if (badgeDriftIndicator) {
              badgeDriftIndicator.textContent = 'Normal';
              badgeDriftIndicator.className = 'badge-status-ok';
            }
          }
        }

        fetchTelemetry();
      } catch (err) {
        console.error('Drift test error:', err);
        alert('Drift test failed: ' + err.message);
      } finally {
        btnSimulateDrift.disabled = false;
        btnSimulateDrift.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 6px; vertical-align: text-bottom;">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
            <line x1="12" y1="9" x2="12" y2="13"></line>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
          </svg>
          Run Drift Check on Live Queries
        `;
      }
    });
  }

  // Init
  initCharts();
  fetchServerHealth();
  fetchTelemetry();

  // Continuous auto-sync every 5 seconds
  setInterval(fetchTelemetry, 5000);
});
