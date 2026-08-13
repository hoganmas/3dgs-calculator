import { SplatViewer } from "/web/splats3d.js";

const $ = (id) => document.getElementById(id);

const statusEl = $("status");
const runBtn = $("run");
const metaEl = $("meta");
const centersEl = $("centers");
const exprInput = $("expr");
const canvas = $("plot");
const ctx = canvas.getContext("2d");
const splatViewer = new SplatViewer($("viewport3d"));

const intensityInput = $("intensity");
const intensityValue = $("intensityValue");

function syncIntensity() {
  const value = Number(intensityInput.value);
  intensityValue.textContent = value.toFixed(2);
  splatViewer.setIntensity(value);
}

intensityInput.addEventListener("input", syncIntensity);
syncIntensity();

const worker = new Worker("/web/pyodide-worker.js");
let ready = false;
let parseOk = true;
let requestId = 0;
let parseTimer = null;
let latestParseId = 0;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function updateRunEnabled() {
  runBtn.disabled = !(ready && parseOk);
}

function setReady(isReady) {
  ready = isReady;
  updateRunEnabled();
}

worker.onmessage = (event) => {
  const { id, type, message, result } = event.data;

  if (type === "status") {
    setStatus(message);
    return;
  }
  if (type === "ready") {
    setStatus(message);
    setReady(true);
    scheduleParse();
    return;
  }
  if (type === "error") {
    setStatus(message, true);
    setReady(true);
    return;
  }
  if (type === "parse_result") {
    if (id !== latestParseId) return;
    if (result.ok) {
      parseOk = true;
      setStatus(`Parsed: ${result.parsed}`);
    } else {
      parseOk = false;
      setStatus(result.error || "Could not parse expression.", true);
    }
    updateRunEnabled();
    return;
  }
  if (type === "result") {
    renderResult(result);
    setStatus("Ready.");
    setReady(true);
  }
};

function currentPayload() {
  return {
    latex: exprInput.value.trim(),
    N: Number($("degree").value),
    epsilon: Number($("epsilon").value),
    x0: Number($("x0").value),
  };
}

function scheduleParse() {
  if (!ready) return;
  clearTimeout(parseTimer);
  parseTimer = setTimeout(() => {
    const latex = exprInput.value.trim();
    if (!latex) {
      parseOk = false;
      setStatus("Expression is empty.", true);
      updateRunEnabled();
      return;
    }
    const id = ++latestParseId;
    worker.postMessage({
      id,
      type: "parse",
      payload: { latex },
    });
  }, 280);
}

function renderResult(result) {
  const errLabel =
    result.max_rel_error == null
      ? "n/a (multi-d)"
      : Number(result.max_rel_error).toExponential(3);

  metaEl.innerHTML = `
    <dt>parsed</dt><dd>${escapeHtml(result.parsed)}</dd>
    <dt>taylor</dt><dd>${escapeHtml(result.taylor)}</dd>
    <dt>vars / dims</dt><dd>${(result.variables || []).join(", ")} · ${result.dims}D</dd>
    <dt>N / ε</dt><dd>${result.N} / ${result.epsilon}</dd>
    <dt>splats</dt><dd>${result.splat_count ?? result.centers.length}</dd>
    <dt>max rel err</dt><dd>${errLabel}</dd>
  `;

  const isMulti = (result.dims || 1) > 1;
  centersEl.innerHTML = result.centers
    .slice(0, 500)
    .map((row) =>
      isMulti
        ? `<tr>
            <td>${fmt(row.x)}</td><td>${fmt(row.y)}</td><td>${fmt(row.z)}</td>
            <td>${fmt(row.weight)}</td>
          </tr>`
        : `<tr>
            <td>${fmt(row.center ?? row.x)}</td>
            <td>${fmt(row.weight)}</td>
          </tr>`
    )
    .join("");

  const head = document.querySelector("#centers").closest("table").querySelector("thead tr");
  head.innerHTML = isMulti
    ? "<th>x</th><th>y</th><th>z</th><th>W</th>"
    : "<th>c</th><th>W</th>";

  drawPreview(result);
  splatViewer.setResult(result);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function fmt(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (Math.abs(n) < 1e-12) return "0";
  return n.toPrecision(6);
}

function evalPoly(coeffs, x) {
  let y = 0;
  let xp = 1;
  for (const c of coeffs) {
    y += c * xp;
    xp *= x;
  }
  return y;
}

function drawPreview(result) {
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 900;
  const cssHeight = 420;
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const width = cssWidth;
  const height = cssHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fffdf8";
  ctx.fillRect(0, 0, width, height);

  if ((result.dims || 1) > 1) {
    ctx.fillStyle = "#6a6358";
    ctx.font = "14px IBM Plex Mono, monospace";
    ctx.fillText("Multi-variable result — see the 3D splat view above.", 24, height / 2);
    ctx.font = "12px IBM Plex Mono, monospace";
    ctx.fillText(`${result.splat_count ?? result.centers.length} gaussians in R^3`, 24, height / 2 + 24);
    return;
  }

  const pad = { l: 44, r: 18, t: 18, b: 36 };
  const xMin = -3;
  const xMax = 3;
  const n = 500;
  const xs = Array.from({ length: n }, (_, i) => xMin + ((xMax - xMin) * i) / (n - 1));

  const components = result.centers.map((c) => {
    const center = c.center ?? c.x;
    const weight = c.weight;
    return xs.map((x) => weight * Math.exp(-0.5 * (x - center) ** 2));
  });
  const sum = xs.map((_, i) => components.reduce((acc, row) => acc + row[i], 0));
  const target = xs.map((x) => evalPoly(result.coeffs, x) * Math.exp(-0.5 * x * x));

  const allY = [...sum, ...target, ...components.flat()];
  let yMin = Math.min(...allY);
  let yMax = Math.max(...allY);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const yPad = 0.08 * (yMax - yMin);
  yMin -= yPad;
  yMax += yPad;

  const xToPx = (x) => pad.l + ((x - xMin) / (xMax - xMin)) * (width - pad.l - pad.r);
  const yToPx = (y) => pad.t + ((yMax - y) / (yMax - yMin)) * (height - pad.t - pad.b);

  ctx.strokeStyle = "rgba(28,25,21,0.08)";
  ctx.lineWidth = 1;
  for (let x = -3; x <= 3; x += 1) {
    ctx.beginPath();
    ctx.moveTo(xToPx(x), pad.t);
    ctx.lineTo(xToPx(x), height - pad.b);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(28,25,21,0.25)";
  ctx.beginPath();
  ctx.moveTo(pad.l, yToPx(0));
  ctx.lineTo(width - pad.r, yToPx(0));
  ctx.stroke();

  const maxAbsW = Math.max(...result.centers.map((c) => Math.abs(c.weight)), 1e-9);
  result.centers.forEach((c, idx) => {
    const center = c.center ?? c.x;
    const weight = c.weight;
    const ys = components[idx];
    const t = Math.abs(weight) / maxAbsW;
    const pos = weight >= 0;
    ctx.beginPath();
    ctx.moveTo(xToPx(xs[0]), yToPx(0));
    ys.forEach((y, i) => ctx.lineTo(xToPx(xs[i]), yToPx(y)));
    ctx.lineTo(xToPx(xs[n - 1]), yToPx(0));
    ctx.closePath();
    ctx.fillStyle = pos
      ? `rgba(15, 107, 92, ${0.12 + 0.25 * t})`
      : `rgba(155, 61, 46, ${0.12 + 0.25 * t})`;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(xToPx(center), yToPx(0), 4 + 6 * t, 0, Math.PI * 2);
    ctx.fillStyle = pos ? "#0f6b5c" : "#9b3d2e";
    ctx.fill();
  });

  const strokeSeries = (ys, color, dashed = false) => {
    ctx.beginPath();
    ys.forEach((y, i) => {
      const px = xToPx(xs[i]);
      const py = yToPx(y);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash(dashed ? [6, 5] : []);
    ctx.stroke();
    ctx.setLineDash([]);
  };

  strokeSeries(target, "#0f6b5c", true);
  strokeSeries(sum, "#1c1915", false);

  ctx.fillStyle = "#6a6358";
  ctx.font = "12px IBM Plex Mono, monospace";
  ctx.fillText("x", width - 24, height - 14);
}

runBtn.addEventListener("click", () => {
  if (!ready || !parseOk) return;
  setReady(false);
  const id = ++requestId;
  worker.postMessage({
    id,
    type: "decompose",
    payload: currentPayload(),
  });
});

exprInput.addEventListener("input", scheduleParse);
exprInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") runBtn.click();
});
