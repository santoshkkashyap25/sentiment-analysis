/**
 * Dashboard Logic - FeedbackML Core
 * Handles real-time sentiment inference, Chart.js telemetry, and live drift simulation.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const form = document.getElementById('sentimentForm');
  const textarea = document.getElementById('feedbackInput');
  const charCount = document.getElementById('charCount');
  const btnClear = document.getElementById('btnClear');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnText = document.getElementById('btnText');
  const sampleChips = document.querySelectorAll('.sample-chip');

  // Prediction elements
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

  // KPI elements
  const kpiTotalRequests = document.getElementById('kpiTotalRequests');
  const kpiAvgLatency = document.getElementById('kpiAvgLatency');
  const kpiP50Latency = document.getElementById('kpiP50Latency');
  const kpiP95Latency = document.getElementById('kpiP95Latency');
  const kpiDriftStatus = document.getElementById('kpiDriftStatus');
  const badgeDriftIndicator = document.getElementById('badgeDriftIndicator');
  const tableCountBadge = document.getElementById('tableCountBadge');
  const tableBody = document.getElementById('inferencesTableBody');
  const modelNameDisplay = document.getElementById('modelNameDisplay');

  // Drift simulation elements
  const btnSimulateDrift = document.getElementById('btnSimulateDrift');
  const driftAlertFeedback = document.getElementById('driftAlertFeedback');
  const btnRefreshTelemetry = document.getElementById('btnRefreshTelemetry');

  // Initialize Charts
  let latencyChart = null;
  let sentimentChart = null;

  function initCharts() {
    const ctxLatency = document.getElementById('latencyChart').getContext('2d');
    const ctxSentiment = document.getElementById('sentimentChart').getContext('2d');

    // Latency Line Chart
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

    // Sentiment Doughnut Chart
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

  // Character Counter
  textarea.addEventListener('input', () => {
    charCount.textContent = textarea.value.length;
  });

  // Clear Button
  btnClear.addEventListener('click', () => {
    textarea.value = '';
    charCount.textContent = '0';
    resultBox.style.display = 'none';
  });

  // Sample Chips
  sampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      textarea.value = chip.dataset.text;
      charCount.textContent = textarea.value.length;
      triggerPrediction(textarea.value);
    });
  });

  // Form Submit
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = textarea.value.trim();
    if (text) {
      triggerPrediction(text);
    }
  });

  // Prediction Trigger Function
  async function triggerPrediction(text) {
    btnSpinner.style.display = 'inline-block';
    btnAnalyze.disabled = true;

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
      btnSpinner.style.display = 'none';
      btnAnalyze.disabled = false;
    }
  }

  // Render Prediction Result
  function displayPredictionResult(data) {
    resultBox.style.display = 'block';

    // Sentiment styling
    const sentiment = data.sentiment || 'Neutral';
    sentimentBadge.textContent = sentiment;
    sentimentBadge.className = 'sentiment-badge-large';

    if (sentiment === 'Positive') {
      sentimentBadge.classList.add('badge-positive');
    } else if (sentiment === 'Negative') {
      sentimentBadge.classList.add('badge-negative');
    } else {
      sentimentBadge.classList.add('badge-neutral');
    }

    const confPct = Math.round((data.confidence || 0) * 1000) / 10;
    confidenceVal.textContent = `${confPct}%`;
    latencyVal.textContent = `${data.response_time_ms || 0} ms`;

    // Probabilities
    const probs = data.probabilities || { Positive: 0.33, Neutral: 0.34, Negative: 0.33 };
    const pPos = Math.round((probs.Positive || 0) * 100);
    const pNeu = Math.round((probs.Neutral || 0) * 100);
    const pNeg = Math.round((probs.Negative || 0) * 100);

    barPos.style.width = `${pPos}%`;
    barNeu.style.width = `${pNeu}%`;
    barNeg.style.width = `${pNeg}%`;

    pctPos.textContent = `${pPos}%`;
    pctNeu.textContent = `${pNeu}%`;
    pctNeg.textContent = `${pNeg}%`;

    nlpTokens.textContent = data.processed_text || '(clean tokens empty)';
  }

  // Fetch Telemetry & Database Audit Trail
  async function fetchTelemetry() {
    try {
      const res = await fetch('/api/telemetry');
      if (!res.ok) return;
      const data = await res.json();

      // Update KPIs
      kpiTotalRequests.textContent = data.total_predictions || 0;
      kpiAvgLatency.innerHTML = `${data.avg_latency_ms || 0} <span class="kpi-unit">ms</span>`;
      kpiP50Latency.textContent = data.p50_latency_ms || 0;
      kpiP95Latency.textContent = data.p95_latency_ms || 0;

      // Drift alert status in KPI
      const alerts = data.drift_alerts || [];
      const hasRecentDrift = alerts.some(a => a.type === 'data_drift');

      if (hasRecentDrift) {
        kpiDriftStatus.textContent = 'Drift Alert';
        kpiDriftStatus.className = 'kpi-value kpi-status-alert';
        badgeDriftIndicator.textContent = 'Shift Alert';
        badgeDriftIndicator.className = 'badge-status-alert';
      } else {
        kpiDriftStatus.textContent = 'Healthy';
        kpiDriftStatus.className = 'kpi-value kpi-status-ok';
        badgeDriftIndicator.textContent = 'Normal';
        badgeDriftIndicator.className = 'badge-status-ok';
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

      // Render Recent Inferences Table
      renderTable(data.recent_predictions || []);
    } catch (err) {
      console.warn('Telemetry sync error:', err);
    }
  }

  function renderTable(rows) {
    tableCountBadge.textContent = `${rows.length} logged`;

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

  // Drift Simulation Button Action
  btnSimulateDrift.addEventListener('click', async () => {
    btnSimulateDrift.disabled = true;
    btnSimulateDrift.innerHTML = `Running KS Statistical Test...`;

    try {
      const res = await fetch('/api/simulate-drift', { method: 'POST' });
      const data = await res.json();

      driftAlertFeedback.style.display = 'block';
      if (data.drift_detected) {
        driftAlertFeedback.className = 'drift-feedback drift-feedback-alert';
        driftAlertFeedback.innerHTML = `
          <strong>Drift Alert Triggered!</strong><br>
          ${data.drift_percentage}% of features shifted (KS p-value &lt; 0.05).<br>
          <em>Retraining threshold exceeded: logged alert to monitoring database.</em>
        `;
        kpiDriftStatus.textContent = 'Drift Alert';
        kpiDriftStatus.className = 'kpi-value kpi-status-alert';
        badgeDriftIndicator.textContent = 'Shift Alert';
        badgeDriftIndicator.className = 'badge-status-alert';
      } else {
        driftAlertFeedback.className = 'drift-feedback drift-feedback-ok';
        driftAlertFeedback.innerHTML = `Statistical check completed: No drift detected (${data.drift_percentage}% shifted).`;
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
        Simulate Production Data Drift
      `;
    }
  });

  // Manual Telemetry Refresh
  btnRefreshTelemetry.addEventListener('click', () => {
    fetchTelemetry();
  });

  // Init
  initCharts();
  fetchTelemetry();

  // Poll telemetry every 5 seconds
  setInterval(fetchTelemetry, 5000);
});
