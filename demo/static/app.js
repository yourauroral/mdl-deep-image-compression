// MDL Deep Image Compression — 前端可视化

const API = "";

// ── 工具函数 ──
async function fetchJSON(url) {
  const res = await fetch(API + url);
  if (!res.ok) return null;
  return res.json();
}

// Chart.js 全局配色
Chart.defaults.color = "#8b8fa3";
Chart.defaults.borderColor = "#2a2d3a";

// ── Panel 1: 图片上传 + 预测 ──
(function initUpload() {
  const area = document.getElementById("upload-area");
  const input = document.getElementById("file-input");
  const preview = document.getElementById("preview-img");
  const placeholder = document.getElementById("upload-placeholder");

  area.addEventListener("click", () => input.click());
  area.addEventListener("dragover", e => { e.preventDefault(); area.classList.add("dragover"); });
  area.addEventListener("dragleave", () => area.classList.remove("dragover"));
  area.addEventListener("drop", e => {
    e.preventDefault();
    area.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files.length) handleFile(input.files[0]); });

  async function handleFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      preview.src = reader.result;
      preview.hidden = false;
      placeholder.hidden = true;
    };
    reader.readAsDataURL(file);

    const form = new FormData();
    form.append("file", file);

    document.getElementById("result-bpp").textContent = "...";
    document.getElementById("result-ce").textContent = "...";
    document.getElementById("result-model").textContent = "...";

    try {
      const res = await fetch(API + "/api/predict", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        document.getElementById("result-bpp").textContent = "N/A";
        document.getElementById("result-model").textContent = err.detail || "错误";
        return;
      }
      const data = await res.json();
      document.getElementById("result-bpp").textContent = data.bpp;
      document.getElementById("result-ce").textContent = data.ce_loss;
      document.getElementById("result-model").textContent = data.model_type.toUpperCase();

      const hmImg = document.getElementById("heatmap-img");
      const hmPlaceholder = document.getElementById("heatmap-placeholder");
      if (data.heatmap) {
        hmImg.src = "data:image/png;base64," + data.heatmap;
        hmImg.hidden = false;
        hmPlaceholder.hidden = true;
      } else {
        hmImg.hidden = true;
        hmPlaceholder.hidden = false;
        hmPlaceholder.textContent = data.model_type === "mspa"
          ? "MSPA 模型暂不支持热力图" : "热力图不可用";
      }
    } catch (e) {
      document.getElementById("result-bpp").textContent = "离线";
      document.getElementById("result-model").textContent = "无法连接后端";
    }
  }
})();

// ── Panel 2: BPP 对比 ──
(async function initMetrics() {
  const data = await fetchJSON("/api/metrics");
  if (!data) return;

  const methods = data.methods.filter(m => m.bpp !== null);
  const labels = methods.map(m => m.name);
  const values = methods.map(m => m.bpp);
  const colors = methods.map(m => m.name.includes("Ours") ? "#6c8cff" : "#4a4d5e");

  new Chart(document.getElementById("chart-metrics"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "BPP (bits/dim)",
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
        barThickness: 36,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "BPP (bits/dim) ↓" }, min: 0 },
        y: { ticks: { font: { size: 12 } } }
      }
    }
  });

  const tbody = document.querySelector("#table-metrics tbody");
  data.methods.forEach(m => {
    const tr = document.createElement("tr");
    const isOurs = m.name.includes("Ours");
    tr.innerHTML = `
      <td class="${isOurs ? "highlight" : ""}">${m.name}</td>
      <td class="${isOurs ? "highlight" : ""}">${m.bpp !== null ? m.bpp.toFixed(2) : "TBD"}</td>
      <td>${m.note}</td>`;
    tbody.appendChild(tr);
  });
})();

// ── Panel 3: Linear Probe ──
(async function initProbe() {
  const data = await fetchJSON("/api/probe");
  if (!data) return;

  new Chart(document.getElementById("chart-probe"), {
    type: "line",
    data: {
      labels: data.layers.map(l => "L" + l),
      datasets: [{
        label: "Top-1 Accuracy (%)",
        data: data.accuracy,
        borderColor: "#6c8cff",
        backgroundColor: "rgba(108,140,255,0.1)",
        fill: true,
        tension: 0.3,
        pointRadius: 3,
        pointBackgroundColor: "#6c8cff",
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: { display: true, text: `${data.model} — ${data.dataset} (${data.num_classes} classes)`, color: "#e1e4ed" }
      },
      scales: {
        x: { title: { display: true, text: "Transformer Layer" } },
        y: { title: { display: true, text: "Accuracy (%)" }, min: 0, max: 100 }
      }
    }
  });
})();

// ── Panel 4: Kernel 性能 ──
(async function initKernels() {
  const data = await fetchJSON("/api/kernels");
  if (!data) return;

  const labels = data.kernels.map(k => k.name);
  new Chart(document.getElementById("chart-kernels"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Triton (ms)",
          data: data.kernels.map(k => k.triton_ms),
          backgroundColor: "#6c8cff",
          borderRadius: 3,
        },
        {
          label: "PyTorch (ms)",
          data: data.kernels.map(k => k.pytorch_ms),
          backgroundColor: "#4a4d5e",
          borderRadius: 3,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const idx = items[0].dataIndex;
              return `Speedup: ${data.kernels[idx].speedup}x`;
            }
          }
        }
      },
      scales: {
        x: { ticks: { maxRotation: 45, font: { size: 10 } } },
        y: { title: { display: true, text: "Latency (ms) ↓" }, min: 0 }
      }
    }
  });
})();

// ── Panel 5: MSPA 多尺度 ──
(async function initScales() {
  const data = await fetchJSON("/api/scales");
  if (!data) return;

  const labels = data.scales.map(s => s.scale);
  const tokens = data.scales.map(s => s.tokens);
  const total = data.total_tokens;

  new Chart(document.getElementById("chart-scales"), {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: tokens,
        backgroundColor: ["#ef5350", "#ff9800", "#ffeb3b", "#4caf50", "#2196f3", "#6c8cff"],
        borderColor: "#1a1d27",
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "right" },
        title: { display: true, text: `总计 ${total} tokens`, color: "#e1e4ed" }
      }
    }
  });

  const tbody = document.querySelector("#table-scales tbody");
  data.scales.forEach(s => {
    const pct = (s.tokens / total * 100).toFixed(1);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${s.scale}</td><td>${s.resolution}</td><td>${s.tokens}</td><td>${pct}%</td>`;
    tbody.appendChild(tr);
  });
})();
