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
  const bpdEl = $("result-bpd");
  const ceEl = $("result-ce");
  const modelEl = $("result-model");
  const extrasEl = $("result-extras");
  const ceCoarseEl = $("result-ce-coarse");
  const ceFineEl = $("result-ce-fine");
  const alphaEl = $("result-alpha");
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

    bpdEl.textContent = ceEl.textContent = modelEl.textContent = "...";
    extrasEl.hidden = true;

    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(API + "/api/predict", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        bpdEl.textContent = "N/A";
        modelEl.textContent = err.detail || "错误";
        return;
      }
      const data = await res.json();
      bpdEl.textContent = data.bpd;
      ceEl.textContent = data.ce_loss;
      modelEl.textContent = data.model_type.toUpperCase();

      // CC-iGPT: 显示双尺度 CE 分解 + α
      if (data.ce_coarse !== undefined) {
        ceCoarseEl.textContent = data.ce_coarse;
        ceFineEl.textContent = data.ce_fine;
        alphaEl.textContent = data.ctx_alpha !== undefined ? data.ctx_alpha : "—";
        extrasEl.hidden = false;
      }

      if (data.heatmap) {
        hmImg.src = "data:image/png;base64," + data.heatmap;
        hmImg.hidden = false;
        hmPlaceholder.hidden = true;
      } else {
        hmImg.hidden = true;
        hmPlaceholder.hidden = false;
        hmPlaceholder.textContent = "热力图不可用";
      }
    } catch (e) {
      bpdEl.textContent = "离线";
      modelEl.textContent = "无法连接后端";
    }
  }
})();

// ── Panel 2: bits/dim 对比 ──
// 设计：聚焦神经 AR 方法之间的 bits/dim 差异（~2.81-2.97）。
// 横轴聚焦 2.70-3.05，数值标签贴在点末端。
(async function initMetrics() {
  const data = await fetchJSON("/api/metrics");
  if (!data) return;

  const isOurs = (n) => n.includes("(Ours)");
  const isOursMain = (n) => n.includes("v2 (Ours)");   // 主表数字（突出 v2，v1 作历史对照）
  // 过滤掉 bpd=null 的占位行，lollipop 图只画已落地结果
  const neural = data.methods.filter(m => m.bpd !== null)
                             .sort((a, b) => a.bpd - b.bpd);

  const labels = neural.map(m => m.name);
  const values = neural.map(m => m.bpd);

  const colorFor = (m) => {
    if (isOursMain(m.name)) return "#6c8cff";          // v2 主表：亮蓝
    if (isOurs(m.name)) return "#8a9bd0";              // v1 历史 Ours：淡蓝
    return "#5a5d72";                                   // baseline：灰
  };
  const colors = neural.map(colorFor);

  const ourBest = neural.find(m => isOursMain(m.name)) || neural.find(m => isOurs(m.name));

  // 副标题：仅展示主结果（取 Ours 中 bpd 最低的一行）
  const desc = document.createElement("p");
  desc.className = "panel-desc";
  desc.innerHTML =
    `聚焦神经自回归方法 (bits/dim ∈ [2.7, 3.0])。` +
    (ourBest
      ? `<span style="color:#6c8cff">${ourBest.name} <b>${ourBest.bpd.toFixed(4)}</b> bits/dim</span>`
      : "");
  const panel = document.getElementById("panel-metrics");
  const chartCt = panel.querySelector(".chart-container");
  if (!panel.querySelector(".panel-desc")) panel.insertBefore(desc, chartCt);

  // Lollipop: 用一条从 xMin 起的细线 + 末端粗点表示
  // Chart.js 没有原生 lollipop，用 bar(很细) + scatter 叠加
  const X_MIN = 2.70;
  const X_MAX = 3.05;

  // 自定义 plugin: 在每个点末端绘制数值标签
  const overlayPlugin = {
    id: "overlay",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      // 数值标签
      const meta = chart.getDatasetMeta(0);
      ctx.save();
      ctx.font = "600 12px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      meta.data.forEach((bar, i) => {
        const name = neural[i].name;
        ctx.fillStyle = isOursMain(name) ? "#6c8cff" : (isOurs(name) ? "#8a9bd0" : "#cfd3e0");
        const truncated = (Math.trunc(values[i] * 100) / 100).toFixed(2);
        ctx.fillText(truncated, bar.x + 10, bar.y);
      });
      ctx.restore();
    }
  };

  new Chart(document.getElementById("chart-metrics"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "bits/dim",
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
        borderWidth: neural.map(m => isOursMain(m.name) ? 2 : 0),
        pointRadius: neural.map(m => isOursMain(m.name) ? 9 : 6),
        pointHoverRadius: neural.map(m => isOursMain(m.name) ? 11 : 8),
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
              const truncated = (Math.trunc(m.bpd * 100) / 100).toFixed(2);
              return [`bits/dim: ${truncated}${std}`, m.note];
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: "bits/dim — 越低越好 ↓" },
          min: X_MIN,
          max: X_MAX,
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { stepSize: 0.05 }
        },
        y: {
          type: "category",
          labels,
          ticks: {
            font: (ctx) => isOursMain(labels[ctx.index] || "")
              ? { size: 12, weight: "700" } : { size: 12 },
            color: (ctx) => {
              const n = labels[ctx.index] || "";
              return isOursMain(n) ? "#6c8cff" : (isOurs(n) ? "#8a9bd0" : "#8b8fa3");
            },
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

  // 表格保留 TBD 占位行作为完整数据展示
  const tbody = document.querySelector("#table-metrics tbody");
  data.methods.forEach(m => {
    const tr = document.createElement("tr");
    const main = isOursMain(m.name);
    const bpdCell = m.bpd !== null ? m.bpd.toFixed(4) : "<i>TBD</i>";
    tr.innerHTML = `
      <td class="${main ? "highlight" : ""}">${m.name}</td>
      <td class="${main ? "highlight" : ""}">${bpdCell}</td>
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
// kernels.json schema：嵌套 `{forward_only: {kernels:[...]}, forward_backward: {kernels:[...]}}`。
// Panel 4 默认渲染 forward_only；avg/max speedup 来自同一段。
(async function initKernels() {
  const data = await fetchJSON("/api/kernels");
  if (!data) return;

  const section = data.forward_only || data.forward_backward;
  if (!section || !Array.isArray(section.kernels)) return;
  const kernels = section.kernels;

  const labels = kernels.map(k => k.name);
  new Chart(document.getElementById("chart-kernels"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Triton (ms)",
          data: kernels.map(k => k.triton_ms),
          backgroundColor: "#6c8cff",
          borderRadius: 3,
        },
        {
          label: "PyTorch (ms)",
          data: kernels.map(k => k.pytorch_ms),
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
        title: { display: true,
                 text: `Forward-only — avg ${section.avg_speedup}x / max ${section.max_speedup}x (${section.max_speedup_kernel})`,
                 color: "#e1e4ed" },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const idx = items[0].dataIndex;
              return `Speedup: ${kernels[idx].speedup}x`;
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
        title: { display: true, text: `总计 ${total} tokens (coarse + fine)`, color: "#e1e4ed" },
        tooltip: {
          callbacks: {
            label: (item) => {
              const s = data.scales[item.dataIndex];
              const tokenPct = (s.tokens / total * 100).toFixed(1);
              const bpdPct = s.share_pct !== undefined ? `bpd 占比 ${s.share_pct}%` : null;
              return [`tokens: ${s.tokens} (${tokenPct}%)`, bpdPct].filter(Boolean);
            }
          }
        }
      }
    }
  });

  const tbody = document.querySelector("#table-scales tbody");
  data.scales.forEach(s => {
    const tokenPct = (s.tokens / total * 100).toFixed(1);
    const bpdPct = s.share_pct !== undefined ? `${s.share_pct}%` : "—";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${s.scale}</td><td>${s.resolution}</td><td>${s.tokens}</td><td>${tokenPct}%</td><td>${bpdPct}</td>`;
    tbody.appendChild(tr);
  });
})();
