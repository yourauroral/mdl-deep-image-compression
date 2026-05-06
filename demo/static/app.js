// MDL Deep Image Compression — 前端可视化

const API = "";

async function fetchJSON(url) {
  const res = await fetch(API + url);
  return res.ok ? res.json() : null;
}

const $ = (id) => document.getElementById(id);

// Chart.js 全局配色
Chart.defaults.color = "#8b8fa3";
Chart.defaults.borderColor = "#2a2d3a";

// ── Panel 1: 图片上传 + 预测 ──
(function initUpload() {
  const area = $("upload-area");
  const input = $("file-input");
  const preview = $("preview-img");
  const placeholder = $("upload-placeholder");
  const bppEl = $("result-bpp");
  const ceEl = $("result-ce");
  const modelEl = $("result-model");
  const hmImg = $("heatmap-img");
  const hmPlaceholder = $("heatmap-placeholder");

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

    bppEl.textContent = ceEl.textContent = modelEl.textContent = "...";

    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(API + "/api/predict", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        bppEl.textContent = "N/A";
        modelEl.textContent = err.detail || "错误";
        return;
      }
      const data = await res.json();
      bppEl.textContent = data.bpp;
      ceEl.textContent = data.ce_loss;
      modelEl.textContent = data.model_type.toUpperCase();

      if (data.heatmap) {
        hmImg.src = "data:image/png;base64," + data.heatmap;
        hmImg.hidden = false;
        hmPlaceholder.hidden = true;
      } else {
        hmImg.hidden = true;
        hmPlaceholder.hidden = false;
        hmPlaceholder.textContent = data.model_type === "ccigpt"
          ? "CC-iGPT 模型暂不支持热力图" : "热力图不可用";
      }
    } catch (e) {
      bppEl.textContent = "离线";
      modelEl.textContent = "无法连接后端";
    }
  }
})();

// ── Panel 2: BPP 对比 ──
// 设计：传统方法 (PNG ~5.87, WebP ~5.02) 与神经 AR (~2.81-2.97) 量级差异过大，
// 同图柱状图会把神经方法间的差异挤成视觉噪声。改为聚焦神经 AR 的 lollipop 图，
// 横轴聚焦 2.70-3.05，数值标签贴在点末端；传统方法放表格 / 副标题作为上下文。
(async function initMetrics() {
  const data = await fetchJSON("/api/metrics");
  if (!data) return;

  const isTraditional = (n) => n.includes("PNG") || n.includes("WebP");
  const traditional = data.methods.filter(m => isTraditional(m.name) && m.bpp !== null);
  const neural = data.methods.filter(m => !isTraditional(m.name) && m.bpp !== null)
                             .sort((a, b) => a.bpp - b.bpp);

  const labels = neural.map(m => m.name);
  const values = neural.map(m => m.bpp);

  const colorFor = (m) => {
    if (m.name === "CC-iGPT (Ours)") return "#6c8cff";
    if (m.name === "iGPT-S (Ours)")  return "#9bb0ff";
    return "#5a5d72";
  };
  const colors = neural.map(colorFor);

  const ourBest = neural.find(m => m.name === "CC-iGPT (Ours)");
  const ourBaseline = neural.find(m => m.name === "iGPT-S (Ours)");

  // 副标题：传统方法上下文 + 主结果
  const desc = document.createElement("p");
  desc.className = "panel-desc";
  desc.innerHTML =
    `聚焦神经自回归方法 (BPP ∈ [2.7, 3.0])。传统无损基线作为参照: ` +
    traditional.map(m => `<b>${m.name.replace(" (lossless)", "")}</b> ${m.bpp.toFixed(2)}`).join(" · ") +
    (ourBest && ourBaseline
      ? ` &nbsp;|&nbsp; <span style="color:#6c8cff">CC-iGPT 较 iGPT-S baseline 改善 <b>−${(ourBaseline.bpp - ourBest.bpp).toFixed(2)}</b> bits/dim</span>`
      : "");
  const panel = document.getElementById("panel-metrics");
  const chartCt = panel.querySelector(".chart-container");
  if (!panel.querySelector(".panel-desc")) panel.insertBefore(desc, chartCt);

  // Lollipop: 用一条从 xMin 起的细线 + 末端粗点表示
  // Chart.js 没有原生 lollipop，用 bar(很细) + scatter 叠加
  const X_MIN = 2.70;
  const X_MAX = 3.05;

  // 自定义 plugin: 在每个点末端绘制数值标签 + 在 baseline 处画虚线
  const overlayPlugin = {
    id: "overlay",
    afterDatasetsDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      // baseline 虚线 (iGPT-S)
      if (ourBaseline) {
        const x = scales.x.getPixelForValue(ourBaseline.bpp);
        ctx.save();
        ctx.strokeStyle = "rgba(155,176,255,0.35)";
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, chartArea.top);
        ctx.lineTo(x, chartArea.bottom);
        ctx.stroke();
        ctx.fillStyle = "#9bb0ff";
        ctx.font = "11px -apple-system, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("iGPT-S baseline", x + 4, chartArea.top + 12);
        ctx.restore();
      }
      // 数值标签
      const meta = chart.getDatasetMeta(0);
      ctx.save();
      ctx.font = "600 12px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      meta.data.forEach((bar, i) => {
        ctx.fillStyle = neural[i].name.includes("Ours") ? "#6c8cff" : "#cfd3e0";
        ctx.fillText(values[i].toFixed(2), bar.x + 10, bar.y);
      });
      ctx.restore();
    }
  };

  new Chart(document.getElementById("chart-metrics"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "BPP",
        data: values,
        backgroundColor: colors,
        borderRadius: 0,
        barThickness: 3,             // 细线 (lollipop 的杆)
        categoryPercentage: 1.0,
        // 通过 pointStyle 在末端画大圆点
        pointStyle: false,
      }, {
        // 第二个 dataset: 散点画末端粗点
        type: "scatter",
        label: "_dot",
        data: values.map((v, i) => ({ x: v, y: i })),
        backgroundColor: colors,
        borderColor: colors.map(c => c === "#6c8cff" ? "#ffffff" : c),
        borderWidth: neural.map(m => m.name === "CC-iGPT (Ours)" ? 2 : 0),
        pointRadius: neural.map(m => m.name === "CC-iGPT (Ours)" ? 9 : 6),
        pointHoverRadius: neural.map(m => m.name === "CC-iGPT (Ours)" ? 11 : 8),
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { left: 8, right: 56, top: 8, bottom: 4 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: (item) => item.datasetIndex === 1,  // 只在散点上显示 tooltip
          callbacks: {
            title: (items) => neural[items[0].dataIndex].name,
            label: (item) => {
              const m = neural[item.dataIndex];
              const std = m.std ? ` ± ${m.std.toFixed(2)}` : "";
              return [`BPP: ${m.bpp.toFixed(2)}${std}`, m.note];
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: "BPP (bits/dim) — 越低越好 ↓" },
          min: X_MIN,
          max: X_MAX,
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { stepSize: 0.05 }
        },
        y: {
          type: "category",
          labels,
          ticks: {
            font: (ctx) => labels[ctx.index]?.includes("Ours")
              ? { size: 12, weight: "700" } : { size: 12 },
            color: (ctx) => labels[ctx.index]?.includes("Ours") ? "#6c8cff" : "#8b8fa3",
            autoSkip: false,
            padding: 8,
          },
          grid: { display: false },
          afterFit: (axis) => { axis.width = 175; }
        }
      }
    },
    plugins: [overlayPlugin]
  });

  // 表格保留全部方法（含 PNG/WebP）作为完整数据展示
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

// ── Panel 5: CC-iGPT 双尺度 ──
(async function initScales() {
  const data = await fetchJSON("/api/scales");
  if (!data) return;

  const labels = data.scales.map(s => `${s.scale} (${s.resolution})`);
  const tokens = data.scales.map(s => s.tokens);
  const total = data.total_tokens;

  new Chart(document.getElementById("chart-scales"), {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: tokens,
        backgroundColor: ["#ff9800", "#6c8cff"],
        borderColor: "#1a1d27",
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "right" },
        title: { display: true, text: `总计 ${total} tokens (coarse + fine)`, color: "#e1e4ed" }
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
