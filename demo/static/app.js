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

// ── Panel 2: 真实可解性 — 无损 roundtrip ──
(function initLossless() {
  const area = $("ll-upload-area");
  const input = $("ll-file-input");
  const preview = $("ll-preview-img");
  const placeholder = $("ll-upload-placeholder");
  const origImg = $("ll-orig"), origPh = $("ll-orig-ph");
  const reconImg = $("ll-recon"), reconPh = $("ll-recon-ph");
  const origDl = $("ll-orig-download"), reconDl = $("ll-recon-download");
  const badge = $("ll-badge");
  const compare = $("ll-compare");
  const resultPh = $("ll-placeholder");
  const codecSub = $("ll-codec-sub");

  area.addEventListener("click", () => input.click());
  area.addEventListener("dragover", e => { e.preventDefault(); area.classList.add("dragover"); });
  area.addEventListener("dragleave", () => area.classList.remove("dragover"));
  area.addEventListener("drop", e => {
    e.preventDefault();
    area.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files.length) handleFile(input.files[0]); });

  function setBars(neural, png, webp) {
    // 以三者最大字节数为满刻度，画相对长度
    const vals = [neural, png, webp].filter(v => v != null);
    const max = Math.max(...vals);
    const pct = (v) => v == null ? 0 : (v / max * 100).toFixed(1);
    $("ll-bar-neural").style.width = pct(neural) + "%";
    $("ll-bar-png").style.width = pct(png) + "%";
    $("ll-bar-webp").style.width = pct(webp) + "%";
  }

  async function handleFile(file) {
    const reader = new FileReader();
    reader.onload = () => { preview.src = reader.result; preview.hidden = false; placeholder.hidden = true; };
    reader.readAsDataURL(file);

    resultPh.hidden = false;
    resultPh.textContent = "执行 roundtrip 中...（一次 forward + 真实算术编解码）";
    badge.hidden = true; compare.hidden = true;
    origImg.hidden = reconImg.hidden = true;
    origPh.hidden = reconPh.hidden = false;
    origDl.hidden = reconDl.hidden = true;
    codecSub.textContent = "neural codec";

    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(API + "/api/lossless", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        resultPh.textContent = err.detail || "错误";
        return;
      }
      const d = await res.json();

      origImg.src = "data:image/png;base64," + d.orig_png;
      reconImg.src = "data:image/png;base64," + d.recon_png;
      origImg.hidden = reconImg.hidden = false;
      origPh.hidden = reconPh.hidden = true;
      // 下载用真实 32×32 原始像素 PNG（非展示用的 4× 放大图）
      origDl.href = "data:image/png;base64," + (d.orig_png_full || d.orig_png);
      reconDl.href = "data:image/png;base64," + (d.recon_png_full || d.recon_png);
      origDl.hidden = reconDl.hidden = false;
      codecSub.textContent = d.neural_bytes + " B";

      badge.hidden = false;
      if (d.pixel_exact) {
        badge.className = "ll-badge ll-badge-ok";
        badge.innerHTML = `✅ bit-identical — 逐像素 diff = 0，bitstream 真实可逆`
          + `　|　achieved <b>${d.achieved_bpd}</b> bpd`;
      } else {
        badge.className = "ll-badge ll-badge-fail";
        badge.innerHTML = `❌ 重建不一致（不应发生 — 请检查 ckpt / 精度）`;
      }

      // 字节数对比
      $("ll-val-neural").innerHTML = `${d.neural_bytes} B　<small>${d.achieved_bpd} bpd</small>`;
      $("ll-val-png").innerHTML = `${d.png_bytes} B　<small>${d.png_bpd} bpd</small>`;
      $("ll-val-webp").innerHTML = d.webp_bytes != null
        ? `${d.webp_bytes} B　<small>${d.webp_bpd} bpd</small>` : "不可用";
      setBars(d.neural_bytes, d.png_bytes, d.webp_bytes);
      compare.hidden = false;
      resultPh.hidden = true;
    } catch (e) {
      resultPh.textContent = "无法连接后端";
    }
  }
})();

// ── Panel 3: bits/dim 对比 ──
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

// ── Panel 4: Linear Probe ──
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

// ── Panel 5: Kernel 性能 ──
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

// ── Panel 6: CC-iGPT 双尺度 ──
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

// ── Panel 7: OOD 检测 (typicality) ──
// ood.json schema: {generated, scorers:[{key,label,desc}], ood:[{name,label,raw_bpd,typ_total,typ_dualscale}]}
// generated=null 或全行无数值 → 显示"待 AutoDL 跑"占位，不画图。
(async function initOOD() {
  const data = await fetchJSON("/api/ood");
  if (!data) return;

  const pending = $("ood-pending");
  const hasData = data.generated &&
    (data.ood || []).some(o => o.raw_bpd != null || o.typ_total != null || o.typ_dualscale != null);

  if (!hasData) {
    pending.hidden = false;
    pending.innerHTML = data.note
      ? data.note
      : "待 AutoDL 跑 <code>bash downstream/run_downstream.sh ood</code> 回填。";
    return;
  }

  $("ood-chart-wrap").hidden = false;
  $("table-ood").hidden = false;

  const oodList = data.ood;
  const labels = oodList.map(o => o.label || o.name);
  // 三个 scorer 各一组柱；颜色：raw 灰、typ_total 淡蓝、dual-scale 主蓝（本工作）
  const series = [
    { key: "raw_bpd",       name: "raw bpd",            color: "#5a5d72" },
    { key: "typ_total",     name: "typicality |z|",     color: "#8a9bd0" },
    { key: "typ_dualscale", name: "dual-scale (Ours)",  color: "#6c8cff" },
  ];

  // 0.5 随机基线参考线
  const baselinePlugin = {
    id: "oodBaseline",
    afterDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      const y = scales.y.getPixelForValue(0.5);
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(chartArea.left, y);
      ctx.lineTo(chartArea.right, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.font = "11px -apple-system, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("0.5 随机基线", chartArea.left + 4, y - 4);
      ctx.restore();
    }
  };

  new Chart(document.getElementById("chart-ood"), {
    type: "bar",
    data: {
      labels,
      datasets: series.map(s => ({
        label: s.name,
        data: oodList.map(o => o[s.key]),
        backgroundColor: s.color,
        borderRadius: 3,
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
        title: { display: true, text: `AUROC — ID=${data.id_dataset || "CIFAR-10 test"}（越高越能识别 OOD）`, color: "#e1e4ed" },
        tooltip: {
          callbacks: {
            afterLabel: (item) => {
              const sc = series[item.datasetIndex];
              const meta = (data.scorers || []).find(s => s.key === sc.key);
              return meta ? meta.desc : "";
            }
          }
        }
      },
      scales: {
        x: { grid: { display: false } },
        y: { title: { display: true, text: "AUROC ↑" }, min: 0, max: 1, ticks: { stepSize: 0.1 } }
      }
    },
    plugins: [baselinePlugin]
  });

  const tbody = document.querySelector("#table-ood tbody");
  const fmt = (v) => v == null ? "—" : v.toFixed(4);
  const cls = (v) => v == null ? "" : (v < 0.5 ? "ood-low" : "");
  oodList.forEach(o => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${o.label || o.name}</td>` +
      `<td class="${cls(o.raw_bpd)}">${fmt(o.raw_bpd)}</td>` +
      `<td>${fmt(o.typ_total)}</td>` +
      `<td class="highlight">${fmt(o.typ_dualscale)}</td>`;
    tbody.appendChild(tr);
  });
})();

// ── Panel 8: 跨数据集 bpd 泛化 ──
// transfer.json schema: {generated, datasets:[{name,label,in_domain,bpd,std,note}]}
(async function initTransfer() {
  const data = await fetchJSON("/api/transfer");
  if (!data) return;

  const pending = $("transfer-pending");
  const rows = (data.datasets || []).filter(d => d.bpd != null);
  const hasData = data.generated && rows.length > 0;

  if (!hasData) {
    pending.hidden = false;
    pending.innerHTML = data.note
      ? data.note
      : "待 AutoDL 跑 <code>bash downstream/run_downstream.sh cross</code> 回填。";
    return;
  }

  $("transfer-chart-wrap").hidden = false;
  $("table-transfer").hidden = false;

  const inDom = rows.find(d => d.in_domain);
  const labels = rows.map(d => d.label || d.name);
  const values = rows.map(d => d.bpd);
  const colors = rows.map(d => d.in_domain ? "#6c8cff" : "#5a5d72");

  new Chart(document.getElementById("chart-transfer"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "bits/dim",
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: { display: true, text: `${data.model || "CC-iGPT v2"} — 同一模型跨数据集 bpd`, color: "#e1e4ed" },
        tooltip: {
          callbacks: {
            label: (item) => {
              const d = rows[item.dataIndex];
              const std = d.std != null ? ` ± ${d.std.toFixed(2)}` : "";
              return `bits/dim: ${d.bpd.toFixed(4)}${std}`;
            },
            afterLabel: (item) => rows[item.dataIndex].note || ""
          }
        }
      },
      scales: {
        x: { title: { display: true, text: "bits/dim — 越低越好 ↓" }, min: 0,
             grid: { color: "rgba(255,255,255,0.04)" } },
        y: { grid: { display: false } }
      }
    }
  });

  const tbody = document.querySelector("#table-transfer tbody");
  rows.forEach(d => {
    const tr = document.createElement("tr");
    const delta = (inDom && !d.in_domain) ? (d.bpd - inDom.bpd) : null;
    const deltaStr = d.in_domain ? "<i>参照</i>"
      : (delta != null ? `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}` : "—");
    const stdStr = d.std != null ? ` ± ${d.std.toFixed(2)}` : "";
    tr.innerHTML =
      `<td class="${d.in_domain ? "highlight" : ""}">${d.label || d.name}</td>` +
      `<td class="${d.in_domain ? "highlight" : ""}">${d.bpd.toFixed(4)}${stdStr}</td>` +
      `<td>${deltaStr}</td>` +
      `<td class="hint">${d.note || ""}</td>`;
    tbody.appendChild(tr);
  });
})();

// ── Panel 9: 图像补全 (AR inpainting) ──
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
    try {
      const res = await fetch(API + "/api/complete", { method: "POST", body: form });
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
      resultPh.innerHTML = `补全完成 — 保留上半 <b>${d.keep_pct}%</b>，温度 <b>${d.temperature}</b>，top-k <b>${d.top_k}</b>。`;
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
    try {
      const res = await fetch(API + "/api/encode", { method: "POST", body: form });
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
            encProgLabel.textContent = `逐 token 编码中… 0 / ${msg.total}`;
          } else if (msg.type === "progress") {
            const pct = (msg.done / msg.total * 100).toFixed(1);
            encProgFill.style.width = pct + "%";
            encProgLabel.textContent = `[${msg.stage}] ${msg.done} / ${msg.total}（${pct}%）`;
          } else if (msg.type === "done") {
            encProgFill.style.width = "100%";
            encProgLabel.textContent = `编码完成 — achieved ${msg.achieved_bpd} bpd`;
            // base64 .bin → Blob → object URL 供下载
            const bytes = Uint8Array.from(atob(msg.bin_b64), c => c.charCodeAt(0));
            const blob = new Blob([bytes], { type: "application/octet-stream" });
            if (lastDownloadUrl) URL.revokeObjectURL(lastDownloadUrl);
            lastDownloadUrl = URL.createObjectURL(blob);
            encDownload.href = lastDownloadUrl;
            encDownload.download = msg.filename || "image.mdlc.bin";
            lastEncFingerprint = msg.fingerprint;

            const exactStr = msg.pixel_exact
              ? `<span class="co-ok">✅ bit-identical</span>`
              : `<span class="co-fail">⚠ 自检不一致</span>`;
            const partsStr = msg.dual
              ? `coarse ${msg.coarse_bits} + fine ${msg.fine_bits} bit`
              : `${msg.neural_bits} bit`;
            encStats.innerHTML =
              `<div class="codec-stat"><span>文件大小</span><b>${msg.bin_bytes} B</b></div>` +
              `<div class="codec-stat"><span>码长</span><b>${msg.neural_bits} bit</b></div>` +
              `<div class="codec-stat"><span>achieved bpd</span><b>${msg.achieved_bpd}</b></div>` +
              `<div class="codec-stat"><span>编码自检</span><b>${exactStr}</b></div>` +
              `<div class="codec-stat codec-stat-wide"><span>分段</span><b>${partsStr}</b></div>` +
              `<div class="codec-stat codec-stat-wide"><span>指纹</span><b><code>${msg.fingerprint}</code></b></div>`;
            encResult.hidden = false;
            encNote.innerHTML = `下载后可直接拖到右侧 ② 解码块还原。<b>指纹 ${msg.fingerprint}</b> 用于同会话交叉校验。`;
            done = true;
          } else if (msg.type === "error") {
            encProgLabel.textContent = "编码出错：" + msg.detail;
            done = true;
          }
        }
      }
      encRun.disabled = false; encRun.textContent = "重新编码";
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
    decFileInfo.innerHTML = `<code>${file.name}</code> · ${file.size} B`;
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
      const res = await fetch(API + "/api/inspect", { method: "POST", body: form });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) {
        structTable.hidden = false;
        structTable.querySelector("tbody").innerHTML =
          `<tr><td>错误</td><td class="co-fail">${d.detail || "解析失败"}</td></tr>`;
        return;
      }
      const row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
      const okStr = d.self_consistent
        ? `<span class="co-ok">✅ payload 与 header 一致</span>`
        : `<span class="co-fail">❌ 不一致（文件可能损坏）</span>`;
      const scaleStr = d.dual ? "双尺度 (coarse+fine)" : "单尺度 (igpt)";
      const partsRow = d.dual
        ? row("coarse / fine", `${d.coarse_nbits} bit / ${d.fine_nbits} bit`)
        : "";
      structTable.querySelector("tbody").innerHTML =
        row("magic / 版本", `${d.magic} / v${d.version}`) +
        row("header (16B)", `<code class="co-hex">${d.header_hex}</code>`) +
        row("尺度", scaleStr) +
        row("图像", `${d.H}×${d.H}×${d.C}（${d.n_subpix} 子像素）`) +
        partsRow +
        row("文件大小", `${d.total_bytes} B（payload ${d.payload_bytes} + header ${d.header_size}）`) +
        row("码长", `${d.total_bits} bit`) +
        row("achieved bpd", `<b>${d.bpd}</b>`) +
        row("自洽校验", okStr);
      structTable.hidden = false;
    } catch (e) {
      structTable.hidden = false;
      structTable.querySelector("tbody").innerHTML =
        `<tr><td>错误</td><td class="co-fail">无法连接后端</td></tr>`;
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
    try {
      const res = await fetch(API + "/api/decode", { method: "POST", body: form });
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
            progLabel.textContent = `逐 token 解码中… 0 / ${msg.total}`;
          } else if (msg.type === "progress") {
            const pct = (msg.done / msg.total * 100).toFixed(1);
            progFill.style.width = pct + "%";
            progLabel.textContent = `[${msg.stage}] ${msg.done} / ${msg.total}（${pct}%）`;
          } else if (msg.type === "done") {
            progFill.style.width = "100%";
            progLabel.textContent = `解码完成 — achieved ${msg.achieved_bpd} bpd`;
            decImg.src = "data:image/png;base64," + msg.recon_png;
            decImg.hidden = false;
            decResult.hidden = false;
            // 真实交叉校验：只有指纹与本会话刚编码的 .bin 逐 token 一致才算成功。
            // 无 lastEncFingerprint（换会话/刷新后传入的 .bin）→ 无法校验，不下成功结论。
            if (lastEncFingerprint && msg.fingerprint === lastEncFingerprint) {
              decBadge.className = "ll-badge ll-badge-ok";
              decBadge.innerHTML = `✅ 盲解码还原 — 指纹 <code>${msg.fingerprint}</code> 与刚编码的 .bin 逐 token 一致`;
            } else if (lastEncFingerprint) {
              decBadge.className = "ll-badge ll-badge-fail";
              decBadge.innerHTML = `❌ 解码失步 — 指纹 <code>${msg.fingerprint}</code> 与编码端 <code>${lastEncFingerprint}</code> 不一致`
                + `<br><small>解码模型/权重与编码时不是同一个（server 重启或 ckpt 切换？），bitstream 无法正确还原</small>`;
            } else {
              decBadge.className = "ll-badge";
              decBadge.innerHTML = `ℹ️ 盲解码完成 — 指纹 <code>${msg.fingerprint}</code>，achieved <b>${msg.achieved_bpd}</b> bpd`
                + `<br><small>本会话未编码该 .bin，无法做逐 token 交叉校验；若图像异常请在同一会话内「编码→下载→解码」复核</small>`;
            }
            done = true;
          } else if (msg.type === "error") {
            progLabel.textContent = "解码出错：" + msg.detail;
            done = true;
          }
        }
      }
      decRun.disabled = false; inspectRun.disabled = false; decRun.textContent = "重新解码";
    } catch (e) {
      progLabel.textContent = "无法连接后端（或连接中断）";
      decRun.disabled = false; inspectRun.disabled = false; decRun.textContent = "重试";
    }
  });
})();

