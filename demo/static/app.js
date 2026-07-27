// MDL Deep Image Compression — 前端可视化

const API = "";

async function apiFetch(url, options = {}) {
  const request = { ...options };
  const headers = new Headers(options.headers || {});
  const storedKey = sessionStorage.getItem("mdlic_api_key");
  if (storedKey) headers.set("X-API-Key", storedKey);
  request.headers = headers;

  let res = await fetch(API + url, request);
  if (res.status === 401 && res.headers.get("X-MDLIC-API-Key-Required") === "1") {
    const suppliedKey = window.prompt("API key");
    if (suppliedKey) {
      sessionStorage.setItem("mdlic_api_key", suppliedKey);
      headers.set("X-API-Key", suppliedKey);
      res = await fetch(API + url, request);
    }
  }
  return res;
}

async function fetchJSON(url) {
  const res = await apiFetch(url);
  return res.ok ? res.json() : null;
}

const $ = (id) => document.getElementById(id);

function appendCell(row, value, className = "") {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  if (value instanceof Node) cell.appendChild(value);
  else cell.textContent = String(value);
  row.appendChild(cell);
  return cell;
}

// Chart.js 全局配色
Chart.defaults.color = "#8b8fa3";
Chart.defaults.borderColor = "#2a2d3a";

// ── 全局数据集状态 ──
// 顶栏 toggle 在 CIFAR-10 / ImageNet64 间切换。数据面板 (3 bpd / 4 probe / 6 scales)
// 重新拉取并重渲染；交互面板 (predict/encode/decode/complete) 在各自 FormData 带上
// currentDataset，由后端按 dataset lazy-load 对应 ckpt（IN64 显存约 2×，64×64 解码慢约 4×）。
let currentDataset = "cifar10";

function renderDataPanels(dataset) {
  renderMetrics(dataset);
  renderProbe(dataset);
  renderScales(dataset);
}

// IN64 交互面板很慢（64×64 解码 ~12500 步/图），切到 IN64 时显示警示条
function updateDatasetWarnings(dataset) {
  document.querySelectorAll(".dataset-warning").forEach(el => {
    el.hidden = (dataset !== "imagenet64");
  });
  const size = dataset === "imagenet64" ? 64 : 32;
  $("predict-size-hint").textContent = `支持 PNG / JPG；分析前转为 ${size}x${size} RGB`;
  $("complete-size-hint").textContent = `转为 ${size}x${size} RGB 后 AR 补全`;
  $("encode-size-hint").textContent = `须恰好为 ${size}x${size}；不做静默缩放`;
  $("cmp-orig-label").textContent = `原图 ${size}×${size}`;
  const coarseSize = size / 4;
  $("scales-desc").textContent =
    `coarse R-only ${coarseSize}×${coarseSize}×1 独立编码，量化后作为条件注入 ` +
    `fine ${size}×${size}×3。`;
}

function onDatasetChange(dataset) {
  currentDataset = dataset;
  renderDataPanels(dataset);
  updateDatasetWarnings(dataset);
}

document.addEventListener("DOMContentLoaded", () => {
  // 顶栏 toggle 绑定
  document.querySelectorAll('input[name="dataset-toggle"]').forEach(radio => {
    radio.addEventListener("change", (e) => {
      if (e.target.checked) onDatasetChange(e.target.value);
    });
  });
  // 首屏渲染默认 CIFAR
  renderDataPanels(currentDataset);
  updateDatasetWarnings(currentDataset);
});

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
  const preprocessingEl = $("result-preprocessing");
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
    preprocessingEl.hidden = true;

    const form = new FormData();
    form.append("file", file);
    form.append("dataset", currentDataset);
    try {
      const res = await apiFetch("/api/predict", { method: "POST", body: form });
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

      if (data.preprocessing) {
        const p = data.preprocessing;
        const operation = p.resized ? `bilinear resize -> ${p.model_width}x${p.model_height}`
                                    : "尺寸保持不变";
        const color = p.converted_to_rgb ? `${p.source_mode} -> RGB` : "RGB";
        preprocessingEl.textContent = `预处理：${p.source_width}x${p.source_height} ${color}；${operation}`;
        preprocessingEl.hidden = false;
      }

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

// ── Panel 3: bits/dim 对比 ──
// 设计：聚焦神经 AR 方法之间的 bits/dim 差异。横轴自适应数据范围，数值标签贴在点末端。
// dataset 感知：CIFAR (~2.7-3.0) / ImageNet64 (~3.4-3.6) 横轴范围不同，按数据动态算。
let _metricsChart = null;
async function renderMetrics(dataset) {
  const data = await fetchJSON("/api/metrics?dataset=" + dataset);
  if (!data) return;

  const isOurs = (n) => n.includes("(Ours)");
  // 过滤掉 bpd=null 的占位行，lollipop 图只画已落地结果
  const neural = data.methods.filter(m => m.bpd !== null)
                             .sort((a, b) => a.bpd - b.bpd);
  // 只有显式 primary=true 且符合单 checkpoint/no-TTA 协议的结果才高亮为主结果。
  const ourBest = neural.filter(m => isOurs(m.name) && m.primary === true)
                        .sort((a, b) => a.bpd - b.bpd)[0] || null;
  const isOursMain = (n) => ourBest && n === ourBest.name;

  const labels = neural.map(m => m.name);
  const values = neural.map(m => m.bpd);

  const colorFor = (m) => {
    if (isOursMain(m.name)) return "#6c8cff";          // 主表：亮蓝
    if (isOurs(m.name)) return "#8a9bd0";              // 其它 Ours：淡蓝
    return "#5a5d72";                                   // baseline：灰
  };
  const colors = neural.map(colorFor);

  // 副标题明确区分正式主结果与仍待新 evaluator 重跑的历史诊断数字。
  const panel = document.getElementById("panel-metrics");
  const chartCt = panel.querySelector(".chart-container");
  let desc = panel.querySelector(".panel-desc");
  if (!desc) {
    desc = document.createElement("p");
    desc.className = "panel-desc";
    panel.insertBefore(desc, chartCt);
  }
  desc.replaceChildren(document.createTextNode(`${data.dataset} - 聚焦神经自回归方法。`));
  if (ourBest) {
    const primary = document.createElement("span");
    primary.style.color = "#6c8cff";
    primary.append(document.createTextNode(` ${ourBest.name} `));
    const value = document.createElement("b");
    value.textContent = ourBest.bpd.toFixed(4);
    primary.append(value, document.createTextNode(" bits/dim"));
    desc.appendChild(primary);
  } else {
    desc.append(document.createTextNode(" 当前 Ours 数字均为历史 ensemble+TTA 诊断结果，正式单模型结果待重评。"));
  }

  // Lollipop: 细线 + 末端粗点。横轴范围按数据自适应（留 ±0.1 余量并对齐 0.05）
  const dataMin = Math.min(...values), dataMax = Math.max(...values);
  const X_MIN = Math.floor((dataMin - 0.10) / 0.05) * 0.05;
  const X_MAX = Math.ceil((dataMax + 0.12) / 0.05) * 0.05;

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
        ctx.fillText(values[i].toFixed(2), bar.x + 10, bar.y);
      });
      ctx.restore();
    }
  };

  if (_metricsChart) _metricsChart.destroy();
  _metricsChart = new Chart(document.getElementById("chart-metrics"), {
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
              return [`bits/dim: ${m.bpd.toFixed(2)}${std}`, m.note];
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

  // 表格保留 TBD 占位行作为完整数据展示（重渲染前清空避免累积）
  const tbody = document.querySelector("#table-metrics tbody");
  tbody.replaceChildren();
  data.methods.forEach(m => {
    const tr = document.createElement("tr");
    const main = isOursMain(m.name);
    appendCell(tr, m.name, main ? "highlight" : "");
    if (m.bpd !== null) {
      appendCell(tr, m.bpd.toFixed(4), main ? "highlight" : "");
    } else {
      const pending = document.createElement("i");
      pending.textContent = "TBD";
      appendCell(tr, pending, main ? "highlight" : "");
    }
    appendCell(tr, m.note);
    tbody.appendChild(tr);
  });
}

// ── Panel 4: Linear Probe ──
let _probeChart = null;
async function renderProbe(dataset) {
  const data = await fetchJSON("/api/probe?dataset=" + dataset);
  if (!data) return;

  const historical = data.protocol_status !== "formal_validation_selected";
  const status = $("probe-status");
  status.textContent = data.note || `Protocol: ${data.protocol_status || "unknown"}`;
  status.className = `data-status ${historical ? "data-status-warning" : "data-status-valid"}`;
  status.hidden = false;
  $("probe-desc").textContent =
    `${data.model}；层曲线仅按其记录的选择协议解释，不从末层下降推断因果机制。`;

  if (_probeChart) _probeChart.destroy();
  _probeChart = new Chart(document.getElementById("chart-probe"), {
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
        // 稀疏锚点（IN64 transfer 仅 3 点）用虚线区分完整曲线
        borderDash: data.sparse ? [6, 4] : [],
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: { display: true,
                 text: `${data.model} — ${data.dataset} (${data.num_classes} classes)`
                       + (historical ? " [历史 test-selected]" : "")
                       + (data.sparse ? " [稀疏锚点]" : ""),
                 color: "#e1e4ed" }
      },
      scales: {
        x: { title: { display: true, text: "Transformer Layer" } },
        y: { title: { display: true, text: "Accuracy (%)" }, min: 0, max: 100 }
      }
    }
  });
}

// ── Panel 5: Kernel 性能 ──
// kernels.json schema：嵌套 `{forward_only: {kernels:[...]}, forward_backward: {kernels:[...]}}`。
// Kernel 面板只在 valid=true 时渲染 forward_only；无效历史数据仅显示警告。
(async function initKernels() {
  const data = await fetchJSON("/api/kernels");
  if (!data) return;

  const desc = $("kernels-desc");
  const chartWrap = $("kernels-chart-wrap");
  if (data.valid === false) {
    desc.textContent = data.note ||
      "历史 benchmark 已失效；修正后的 harness 尚未在 GPU 上重跑。";
    desc.className = "data-status data-status-warning";
    chartWrap.hidden = true;
    return;
  } else if (data.note) {
    desc.textContent = data.note;
  }
  desc.className = "panel-desc";
  chartWrap.hidden = false;

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

// ── Panel 6: CC-iGPT 双尺度 ──
let _scalesChart = null;
async function renderScales(dataset) {
  const data = await fetchJSON("/api/scales?dataset=" + dataset);
  if (!data) return;

  const labels = data.scales.map(s => `${s.scale} (${s.resolution})`);
  const tokens = data.scales.map(s => s.tokens);
  const total = data.total_tokens;
  const status = $("scales-status");
  if (data.derived_from_rounded_ce) {
    status.textContent = data.note;
    status.className = "data-status data-status-warning";
    status.hidden = false;
  } else {
    status.hidden = true;
  }

  if (_scalesChart) _scalesChart.destroy();
  _scalesChart = new Chart(document.getElementById("chart-scales"), {
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
        title: {
          display: true,
          text: `${data.derived_from_rounded_ce ? "历史近似 " : ""}${data.bpd_total} bpd · ${total} tokens`,
          color: "#e1e4ed"
        },
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
  tbody.replaceChildren();
  data.scales.forEach(s => {
    const tokenPct = (s.tokens / total * 100).toFixed(1);
    const bpdPct = s.share_pct !== undefined ? `${s.share_pct}%` : "—";
    const tr = document.createElement("tr");
    [s.scale, s.resolution, s.tokens, `${tokenPct}%`, bpdPct]
      .forEach(value => appendCell(tr, value));
    tbody.appendChild(tr);
  });
}

// ── Panel 7: 图像补全 (AR inpainting) ──
// 实时 POST /api/complete（无 KV-cache，~20–40s）。上传后 enable 按钮，点击才跑（避免误触长采样）。
(function initComplete() {
  const area = $("cmp-upload-area");
  const input = $("cmp-file-input");
  const preview = $("cmp-preview-img");
  const placeholder = $("cmp-upload-placeholder");
  const keepSlider = $("cmp-keep"), keepVal = $("cmp-keep-val");
  const tempSlider = $("cmp-temp"), tempVal = $("cmp-temp-val");
  const runBtn = $("cmp-run");
  const origImg = $("cmp-orig"), origPh = $("cmp-orig-ph");
  const maskedImg = $("cmp-masked"), maskedPh = $("cmp-masked-ph");
  const compImg = $("cmp-completed"), compPh = $("cmp-completed-ph");
  const resultPh = $("cmp-placeholder");

  let currentFile = null;

  keepSlider.addEventListener("input", () => { keepVal.textContent = keepSlider.value + "%"; });
  tempSlider.addEventListener("input", () => { tempVal.textContent = (tempSlider.value / 10).toFixed(1); });

  area.addEventListener("click", () => input.click());
  area.addEventListener("dragover", e => { e.preventDefault(); area.classList.add("dragover"); });
  area.addEventListener("dragleave", () => area.classList.remove("dragover"));
  area.addEventListener("drop", e => {
    e.preventDefault();
    area.classList.remove("dragover");
    if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files.length) pickFile(input.files[0]); });

  function pickFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = () => { preview.src = reader.result; preview.hidden = false; placeholder.hidden = true; };
    reader.readAsDataURL(file);
    runBtn.disabled = false;
    runBtn.textContent = "运行补全";
  }

  runBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    runBtn.disabled = true;
    runBtn.textContent = "采样中…（约 20–40s）";
    resultPh.hidden = false;
    resultPh.textContent = "AR 逐 token 采样中…（无 KV-cache，每 token 一次完整 forward，请稍候）";
    origImg.hidden = maskedImg.hidden = compImg.hidden = true;
    origPh.hidden = maskedPh.hidden = compPh.hidden = false;

    const form = new FormData();
    form.append("file", currentFile);
    form.append("keep_frac", (keepSlider.value / 100).toFixed(2));
    form.append("temperature", (tempSlider.value / 10).toFixed(1));
    form.append("top_k", "100");
    form.append("dataset", currentDataset);
    try {
      const res = await apiFetch("/api/complete", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        resultPh.textContent = err.detail || "错误";
        runBtn.disabled = false; runBtn.textContent = "重试";
        return;
      }
      const d = await res.json();
      origImg.src = "data:image/png;base64," + d.orig_png;
      maskedImg.src = "data:image/png;base64," + d.masked_png;
      compImg.src = "data:image/png;base64," + d.completed_png;
      origImg.hidden = maskedImg.hidden = compImg.hidden = false;
      origPh.hidden = maskedPh.hidden = compPh.hidden = true;
      resultPh.hidden = false;
      resultPh.textContent = `补全完成 - 保留上半 ${d.keep_pct}%，温度 ${d.temperature}，top-k ${d.top_k}。`;
      runBtn.disabled = false; runBtn.textContent = "重新采样";
    } catch (e) {
      resultPh.textContent = "无法连接后端";
      runBtn.disabled = false; runBtn.textContent = "重试";
    }
  });
})();

// ── Panel 10: 无损 codec — 图像 ⇄ .bin ──
// 编码：POST /api/encode → base64 .bin + 统计 → Blob 下载，暂存 fingerprint。
// 解析：POST /api/inspect → 即时填结构表（无 GPU）。
// 解码：POST /api/decode → StreamingResponse NDJSON，逐行读 getReader() 更新进度，
//       末行 done 显示还原图；若与刚编码的 .bin fingerprint 一致 → 同会话交叉校验通过。
(function initCodec() {
  // —— 编码块 ——
  const encArea = $("co-enc-upload-area");
  const encInput = $("co-enc-file-input");
  const encPreview = $("co-enc-preview");
  const encPh = $("co-enc-placeholder");
  const encRun = $("co-enc-run");
  const encResult = $("co-enc-result");
  const encStats = $("co-enc-stats");
  const encDownload = $("co-enc-download");
  const encNote = $("co-enc-note");
  const encProgWrap = $("co-enc-progress-wrap");
  const encProgFill = $("co-enc-progress-fill");
  const encProgLabel = $("co-enc-progress-label");

  let encFile = null;
  let lastDownloadUrl = null;   // 上一个 Blob object URL，换图时 revoke 防泄漏
  let lastEncFingerprint = null;

  function bindUpload(area, input, onPick) {
    area.addEventListener("click", () => input.click());
    area.addEventListener("dragover", e => { e.preventDefault(); area.classList.add("dragover"); });
    area.addEventListener("dragleave", () => area.classList.remove("dragover"));
    area.addEventListener("drop", e => {
      e.preventDefault(); area.classList.remove("dragover");
      if (e.dataTransfer.files.length) onPick(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", () => { if (input.files.length) onPick(input.files[0]); });
  }

  bindUpload(encArea, encInput, (file) => {
    encFile = file;
    const reader = new FileReader();
    reader.onload = () => { encPreview.src = reader.result; encPreview.hidden = false; encPh.hidden = true; };
    reader.readAsDataURL(file);
    encRun.disabled = false; encRun.textContent = "编码为 .bin";
  });

  encRun.addEventListener("click", async () => {
    if (!encFile) return;
    encRun.disabled = true; encRun.textContent = "编码中…";
    encResult.hidden = true;
    encNote.textContent = "逐 token gold 算术编码中…（与解码端同源，bit-exact 可解）";
    encProgWrap.hidden = false;
    encProgFill.style.width = "0%";
    encProgLabel.textContent = "启动编码…";

    const form = new FormData();
    form.append("file", encFile);
    form.append("dataset", currentDataset);
    try {
      const res = await apiFetch("/api/encode", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        encNote.textContent = err.detail || "编码失败";
        encProgWrap.hidden = true;
        encRun.disabled = false; encRun.textContent = "重试";
        return;
      }
      // 流式逐行读 NDJSON（同 /api/decode）
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let done = false;
      let streamErr = false;   // 收到 error 行 or 流提前断 → 用 '重试' 标签，note 不留 'bit-exact 可解'
      while (!done) {
        const { value, done: rdDone } = await reader.read();
        if (rdDone) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          let msg;
          try { msg = JSON.parse(line); } catch { continue; }
          if (msg.type === "start") {
            encProgLabel.textContent = `模型预测中… 0 / ${msg.total}`;
          } else if (msg.type === "progress") {
            const pct = (msg.done / msg.total * 100).toFixed(1);
            encProgFill.style.width = pct + "%";
            encProgLabel.textContent = `[${msg.stage}] ${msg.done} / ${msg.total}（${pct}%）`;
          } else if (msg.type === "done") {
            encProgFill.style.width = "100%";
            encProgLabel.textContent = `编码完成 - payload ${msg.payload_bpd} bpd；文件 ${msg.file_bpd} bpd`;
            // base64 .bin → Blob → object URL 供下载
            const bytes = Uint8Array.from(atob(msg.bin_b64), c => c.charCodeAt(0));
            const blob = new Blob([bytes], { type: "application/octet-stream" });
            if (lastDownloadUrl) URL.revokeObjectURL(lastDownloadUrl);
            lastDownloadUrl = URL.createObjectURL(blob);
            encDownload.href = lastDownloadUrl;
            encDownload.download = msg.filename || "image.mdlc.bin";
            lastEncFingerprint = msg.fingerprint;

            const partsStr = msg.dual
              ? `coarse ${msg.coarse_bits} + fine ${msg.fine_bits} bit`
              : `${msg.neural_bits} bit`;
            encStats.replaceChildren();
            const addStat = (label, value, wide = false, code = false) => {
              const stat = document.createElement("div");
              stat.className = "codec-stat" + (wide ? " codec-stat-wide" : "");
              const name = document.createElement("span");
              name.textContent = label;
              const strong = document.createElement("b");
              if (code) {
                const codeEl = document.createElement("code");
                codeEl.textContent = String(value);
                strong.appendChild(codeEl);
              } else {
                strong.textContent = String(value);
              }
              stat.append(name, strong);
              encStats.appendChild(stat);
            };
            addStat("文件大小", `${msg.bin_bytes} B`);
            addStat("容器", `MDLC v${msg.container_version}`);
            addStat("算术码长", `${msg.neural_bits} bit`);
            addStat("payload bpd", msg.payload_bpd);
            addStat("完整文件 bpd", msg.file_bpd);
            addStat("分段", partsStr, true);
            addStat("指纹", msg.fingerprint, true, true);
            encResult.hidden = false;
            encNote.textContent = `可在右侧解码；指纹 ${msg.fingerprint} 用于同会话 RGB token 校验。`;
            done = true;
          } else if (msg.type === "error") {
            encProgLabel.textContent = "编码出错：" + msg.detail;
            encNote.textContent = "编码失败，未生成 .bin";
            streamErr = true;
            done = true;
          }
        }
      }
      // 流提前断（rdDone 但没收到 done/error）：别留下卡住的进度条静默假成功
      if (!done) {
        encProgLabel.textContent = "连接中断，编码未完成";
        encNote.textContent = "编码失败，未生成 .bin";
        streamErr = true;
      }
      encRun.disabled = false; encRun.textContent = streamErr ? "重试" : "重新编码";
    } catch (e) {
      encNote.textContent = "无法连接后端";
      encProgWrap.hidden = true;
      encRun.disabled = false; encRun.textContent = "重试";
    }
  });

  // —— 解码块 ——
  const decArea = $("co-dec-upload-area");
  const decInput = $("co-dec-file-input");
  const decPh = $("co-dec-placeholder");
  const decFileInfo = $("co-dec-fileinfo");
  const inspectRun = $("co-inspect-run");
  const decRun = $("co-dec-run");
  const structTable = $("co-struct-table");
  const progWrap = $("co-dec-progress-wrap");
  const progFill = $("co-dec-progress-fill");
  const progLabel = $("co-dec-progress-label");
  const decResult = $("co-dec-result");
  const decImg = $("co-dec-img");
  const decBadge = $("co-dec-badge");

  let decFile = null;

  bindUpload(decArea, decInput, (file) => {
    decFile = file;
    decFileInfo.hidden = false;
    const filename = document.createElement("code");
    filename.textContent = file.name;
    decFileInfo.replaceChildren(filename, document.createTextNode(` - ${file.size} B`));
    decPh.hidden = true;
    inspectRun.disabled = false; decRun.disabled = false;
    structTable.hidden = true; decResult.hidden = true; progWrap.hidden = true;
  });

  inspectRun.addEventListener("click", async () => {
    if (!decFile) return;
    inspectRun.disabled = true; inspectRun.textContent = "解析中…";
    const form = new FormData();
    form.append("file", decFile);
    try {
      const res = await apiFetch("/api/inspect", { method: "POST", body: form });
      const d = await res.json().catch(() => ({}));
      const tbody = structTable.querySelector("tbody");
      tbody.replaceChildren();
      const row = (key, value, className = "") => {
        const tr = document.createElement("tr");
        appendCell(tr, key);
        appendCell(tr, value, className);
        tbody.appendChild(tr);
      };
      if (!res.ok) {
        structTable.hidden = false;
        row("错误", d.detail || "解析失败", "co-fail");
        return;
      }
      const scaleStr = d.dual ? "双尺度 (coarse+fine)" : "单尺度 (igpt)";
      row("magic / 版本", `${d.magic} / v${d.version}`);
      const header = document.createElement("code");
      header.className = "co-hex";
      header.textContent = d.header_hex;
      row(`fixed header (${d.fixed_header_size}B)`, header);
      row("尺度", scaleStr);
      row("图像", `${d.H}x${d.W}x${d.C}（${d.n_subpix} 子像素）`);
      if (d.dual) row("coarse / fine", `${d.coarse_nbits} bit / ${d.fine_nbits} bit`);
      row("文件大小", `${d.total_bytes} B（payload ${d.payload_bytes} + 非 payload ${d.header_size}）`);
      row("算术码长", `${d.total_bits} bit`);
      row("payload bpd", d.payload_bpd);
      row("packed payload bpd", d.packed_payload_bpd);
      row("完整文件 bpd", d.file_bpd);
      row("完整性", d.integrity_verified ? "SHA-256 已验证" : "仅结构校验（legacy v1）",
          d.integrity_verified ? "co-ok" : "");
      row("模型绑定", d.model_bound ? "是" : "否（legacy v1）",
          d.model_bound ? "co-ok" : "");
      if (d.codec_identity) {
        row("模型类型", d.codec_identity.model_type);
        row("checkpoint SHA-256", d.codec_identity.checkpoint_sha256, "co-hex");
        row("RGB SHA-256", d.source_rgb_sha256, "co-hex");
      }
      row("自洽校验", d.self_consistent ? "payload 与 header 一致" : "不一致（文件可能损坏）",
          d.self_consistent ? "co-ok" : "co-fail");
      structTable.hidden = false;
    } catch (e) {
      structTable.hidden = false;
      const tbody = structTable.querySelector("tbody");
      tbody.replaceChildren();
      const tr = document.createElement("tr");
      appendCell(tr, "错误");
      appendCell(tr, "无法连接后端", "co-fail");
      tbody.appendChild(tr);
    } finally {
      inspectRun.disabled = false; inspectRun.textContent = "解析结构（即时）";
    }
  });

  decRun.addEventListener("click", async () => {
    if (!decFile) return;
    decRun.disabled = true; inspectRun.disabled = true;
    decRun.textContent = "解码中…";
    decResult.hidden = true; decImg.hidden = true;
    progWrap.hidden = false;
    progFill.style.width = "0%";
    progLabel.textContent = "启动解码…";

    const form = new FormData();
    form.append("file", decFile);
    form.append("dataset", currentDataset);
    try {
      const res = await apiFetch("/api/decode", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        progLabel.textContent = err.detail || "解码失败";
        decRun.disabled = false; inspectRun.disabled = false; decRun.textContent = "重试";
        return;
      }
      // 流式逐行读 NDJSON
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let done = false;
      let streamErr = false;
      while (!done) {
        const { value, done: rdDone } = await reader.read();
        if (rdDone) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          let msg;
          try { msg = JSON.parse(line); } catch { continue; }
          if (msg.type === "start") {
            progLabel.textContent = `模型预测中… 0 / ${msg.total}`;
          } else if (msg.type === "progress") {
            const pct = (msg.done / msg.total * 100).toFixed(1);
            progFill.style.width = pct + "%";
            progLabel.textContent = `[${msg.stage}] ${msg.done} / ${msg.total}（${pct}%）`;
          } else if (msg.type === "done") {
            progFill.style.width = "100%";
            progLabel.textContent = `解码完成 - payload ${msg.payload_bpd} bpd；文件 ${msg.file_bpd} bpd`;
            decImg.src = "data:image/png;base64," + msg.recon_png;
            decImg.hidden = false;
            decResult.hidden = false;
            // 真实交叉校验：只有指纹与本会话刚编码的 .bin 逐 token 一致才算成功。
            // 无 lastEncFingerprint（换会话/刷新后传入的 .bin）→ 无法校验，不下成功结论。
            if (msg.rgb_checksum_verified && lastEncFingerprint && msg.fingerprint === lastEncFingerprint) {
              decBadge.className = "ll-badge ll-badge-ok";
              decBadge.textContent = `盲解码还原 - RGB SHA-256 已验证，指纹 ${msg.fingerprint} 与本次编码一致`;
            } else if (lastEncFingerprint) {
              decBadge.className = "ll-badge ll-badge-fail";
              decBadge.textContent = `解码失步 - 指纹 ${msg.fingerprint} 与编码端 ${lastEncFingerprint} 不一致；请检查模型和权重`;
            } else {
              decBadge.className = "ll-badge";
              decBadge.textContent = `盲解码完成 - RGB SHA-256 已验证；指纹 ${msg.fingerprint}，payload ${msg.payload_bpd} bpd`;
            }
            done = true;
          } else if (msg.type === "error") {
            progLabel.textContent = "解码出错：" + msg.detail;
            streamErr = true;
            done = true;
          }
        }
      }
      // 流提前断（rdDone 但没收到 done/error）：明确报错，别留卡住的进度条
      if (!done) { progLabel.textContent = "连接中断，解码未完成"; streamErr = true; }
      decRun.disabled = false; inspectRun.disabled = false;
      decRun.textContent = streamErr ? "重试" : "重新解码";
    } catch (e) {
      progLabel.textContent = "无法连接后端（或连接中断）";
      decRun.disabled = false; inspectRun.disabled = false; decRun.textContent = "重试";
    }
  });
})();
