/* ============================================================
   PADC Dashboard — app.js
   Parallel Matrix Multiplication Visualizer
   Custom Rectangular Dimensions (e.g. A: 2x3, B: 3x4 -> C: 2x4)
   Manual Value Entry + OpenMP Multi-Thread Simulation
   ============================================================ */

'use strict';

// ── State ─────────────────────────────────────────────────────
const state = {
  aRows: 2, // M
  aCols: 3, // K
  bRows: 3, // K (must equal aCols for multiplication)
  bCols: 4, // N
  threads: 4,
  linkDims: true,
  running: false,
  runCount: 0,
  history: [],
  logStart: Date.now(),
};

// Operation count for rectangular multiplication: 2 * M * K * N FLOPs
function calcFlops(M, K, N) {
  return 2 * M * K * N;
}

// Realistic timing model (Amdahl's law + thread overhead)
function seqTimeRect(M, K, N) {
  const flops = calcFlops(M, K, N);
  return flops / 3e9; // baseline 3 GFLOPS
}

function parTimeRect(M, K, N, p) {
  const ts = seqTimeRect(M, K, N);
  const serial = 0.03; // ~3% serial fraction
  const overhead = 0.00008 * p * Math.log2(p + 1);
  const t = ts * (serial + (1 - serial) / p) + overhead;
  return t * (1 + (Math.random() - 0.5) * 0.04);
}

// ── DOM refs ──────────────────────────────────────────────────
const dimARows       = document.getElementById('dim-a-rows');
const dimACols       = document.getElementById('dim-a-cols');
const dimBRows       = document.getElementById('dim-b-rows');
const dimBCols       = document.getElementById('dim-b-cols');
const tagDimA        = document.getElementById('tag-dim-a');
const tagDimB        = document.getElementById('tag-dim-b');
const linkDimsCb     = document.getElementById('link-dims');
const mismatchAlert  = document.getElementById('dim-mismatch-alert');
const alertACols     = document.getElementById('alert-a-cols');
const alertBRows     = document.getElementById('alert-b-rows');
const btnFixDims     = document.getElementById('btn-fix-dims');

const sliderThreads  = document.getElementById('slider-threads');
const valThreads     = document.getElementById('val-threads');
const btnRun         = document.getElementById('btn-run');
const btnReset       = document.getElementById('btn-reset');
const btnClearLog    = document.getElementById('btn-clear-log');
const matrixWrapper  = document.getElementById('matrix-wrapper');
const matrixOverflow = document.getElementById('matrix-overflow');
const matrixStatus   = document.getElementById('matrix-status');
const threadList     = document.getElementById('thread-list');
const threadsActive  = document.getElementById('threads-active');
const execLog        = document.getElementById('exec-log');
const toast          = document.getElementById('toast');
const toastIcon      = document.getElementById('toast-icon');
const toastMsg       = document.getElementById('toast-msg');
const rowsPerThread  = document.getElementById('rows-per-thread');
const totalOps       = document.getElementById('total-ops');
const speedupNLabel  = document.getElementById('speedup-n-label');
const speedupTable   = document.getElementById('speedup-table');
const matrixInputSec = document.getElementById('matrix-input-section');

const statTime       = document.getElementById('stat-time');
const statSpeedup    = document.getElementById('stat-speedup');
const statEff        = document.getElementById('stat-efficiency');
const barTime        = document.getElementById('bar-time');
const barSpeedup     = document.getElementById('bar-speedup');
const barEff         = document.getElementById('bar-efficiency');

// ── Validation & Dimension Sync ───────────────────────────────
function sanitizeDim(val, defaultVal = 2) {
  let n = parseInt(val);
  if (isNaN(n) || n < 1) n = 1;
  if (n > 500) n = 500;
  return n;
}

function updateDimensions() {
  state.aRows = sanitizeDim(dimARows.value, 2);
  state.aCols = sanitizeDim(dimACols.value, 3);
  state.bRows = sanitizeDim(dimBRows.value, 3);
  state.bCols = sanitizeDim(dimBCols.value, 4);
  state.threads = parseInt(sliderThreads.value);
  state.linkDims = linkDimsCb.checked;

  dimARows.value = state.aRows;
  dimACols.value = state.aCols;
  dimBRows.value = state.bRows;
  dimBCols.value = state.bCols;

  tagDimA.textContent = `${state.aRows} × ${state.aCols}`;
  tagDimB.textContent = `${state.bRows} × ${state.bCols}`;
  valThreads.textContent = state.threads;
  sliderThreads.setAttribute('aria-valuenow', state.threads);

  // Check dimension compatibility: Cols(A) must equal Rows(B)
  const isValid = state.aCols === state.bRows;
  if (!isValid) {
    mismatchAlert.style.display = 'flex';
    alertACols.textContent = state.aCols;
    alertBRows.textContent = state.bRows;
    btnRun.disabled = true;
    btnRun.title = 'Cannot multiply: Cols of A must equal Rows of B';
  } else {
    mismatchAlert.style.display = 'none';
    btnRun.disabled = false;
    btnRun.title = '';
  }

  // Update Algorithm info
  rowsPerThread.textContent = Math.ceil(state.aRows / state.threads);
  const ops = calcFlops(state.aRows, state.aCols, state.bCols);
  totalOps.textContent = formatSI(ops);
  speedupNLabel.textContent = `${state.aRows}×${state.aCols} · ${state.bRows}×${state.bCols}`;

  buildMatrixInputGrids();
  buildSpeedupTable();
  buildMatrixVisualizer(null, null);
  buildThreadList();
  drawChart();
}

// Event Listeners for Dimensions
dimARows.addEventListener('change', updateDimensions);
dimACols.addEventListener('change', () => {
  if (linkDimsCb.checked) {
    dimBRows.value = dimACols.value;
  }
  updateDimensions();
});
dimBRows.addEventListener('change', () => {
  if (linkDimsCb.checked) {
    dimACols.value = dimBRows.value;
  }
  updateDimensions();
});
dimBCols.addEventListener('change', updateDimensions);

linkDimsCb.addEventListener('change', () => {
  if (linkDimsCb.checked) {
    dimBRows.value = dimACols.value;
    updateDimensions();
  }
});

btnFixDims.addEventListener('click', () => {
  dimBRows.value = dimACols.value;
  linkDimsCb.checked = true;
  updateDimensions();
  showToast('🔗', `Aligned Matrix B rows to ${dimACols.value} (Cols of A)`);
});

sliderThreads.addEventListener('input', () => {
  state.threads = parseInt(sliderThreads.value);
  valThreads.textContent = state.threads;
  sliderThreads.setAttribute('aria-valuenow', state.threads);
  rowsPerThread.textContent = Math.ceil(state.aRows / state.threads);
  buildThreadList();
  drawChart();
});

// ── Matrix Input Grids ────────────────────────────────────────
function buildMatrixInputGrids() {
  if (!matrixInputSec) return;

  const cellW_A = state.aCols <= 4 ? 64 : state.aCols <= 8 ? 50 : 38;
  const cellH_A = state.aRows <= 4 ? 54 : state.aRows <= 8 ? 44 : 34;

  const cellW_B = state.bCols <= 4 ? 64 : state.bCols <= 8 ? 50 : 38;
  const cellH_B = state.bRows <= 4 ? 54 : state.bRows <= 8 ? 44 : 34;

  matrixInputSec.innerHTML = `
    <div class="row-2col">
      ${buildInputCard('A', state.aRows, state.aCols, cellW_A, cellH_A, 'icon-blue', 'var(--blue-primary)')}
      ${buildInputCard('B', state.bRows, state.bCols, cellW_B, cellH_B, 'icon-purple', 'var(--purple)')}
    </div>
  `;
}

// In-memory values for large matrices (>10x10)
const matrixMemory = {
  a: null,
  b: null,
};

function buildInputCard(label, rows, cols, cellW, cellH, iconClass, color) {
  const isLarge = rows > 10 || cols > 10;
  const dispRows = Math.min(rows, 6);
  const dispCols = Math.min(cols, 6);
  const previewW = dispCols <= 4 ? 64 : 48;
  const previewH = dispRows <= 4 ? 54 : 40;

  let cells = '';
  for (let i = 0; i < dispRows; i++) {
    for (let j = 0; j < dispCols; j++) {
      const defaultVal = 1;
      cells += `<input
        type="number"
        class="matrix-input-cell"
        id="inp-${label.toLowerCase()}-${i}-${j}"
        data-matrix="${label.toLowerCase()}"
        data-row="${i}"
        data-col="${j}"
        value="${defaultVal}"
        style="width:${previewW}px; height:${previewH}px; font-size:14px;"
        aria-label="Matrix ${label} [${i}][${j}]"
        onkeydown="handleCellKeyNav(event, '${label.toLowerCase()}', ${i}, ${j}, ${dispRows}, ${dispCols})"
      />`;
    }
  }

  const largeNotice = isLarge ? `
    <div style="font-size:11px; color:var(--text-muted); font-family:'JetBrains Mono',monospace; margin-bottom:10px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:6px;">
      <span>⚡ Large Matrix Mode: Showing top-left ${dispRows}×${dispCols} preview (Full: ${rows}×${cols} = ${rows*cols} elements)</span>
    </div>
  ` : '';

  return `
    <section class="card">
      <div class="card-header">
        <div class="card-title">
          <div class="card-title-icon ${iconClass}">✏️</div>
          <span style="color:${color}; font-weight:700;">Matrix ${label}</span>
          <span style="color:var(--text-muted); font-size:11px; margin-left:4px;">(${rows} × ${cols})</span>
        </div>
        <div class="matrix-input-toolbar">
          <button class="tool-btn" onclick="fillMatrixPrompt('${label.toLowerCase()}', ${rows}, ${cols})" title="Set all cells to a number">Fill all</button>
          <button class="tool-btn" onclick="randomizeMatrix('${label.toLowerCase()}', ${rows}, ${cols})" title="Random numbers 1-9">Random</button>
          <button class="tool-btn" onclick="identityMatrix('${label.toLowerCase()}', ${rows}, ${cols})" title="Identity / Diagonal">Identity</button>
          <button class="tool-btn" onclick="clearMatrix('${label.toLowerCase()}', ${rows}, ${cols})" title="Clear to zeros">Zeros</button>
        </div>
      </div>
      ${largeNotice}
      <div
        class="matrix-input-grid"
        id="input-grid-${label.toLowerCase()}"
        style="grid-template-columns: repeat(${dispCols}, ${previewW}px);"
        role="grid"
        aria-label="Matrix ${label} manual input"
      >${cells}</div>
    </section>
  `;
}

// Arrow key navigation between cells
window.handleCellKeyNav = function(e, label, r, c, totalR, totalC) {
  let targetR = r;
  let targetC = c;
  if (e.key === 'ArrowRight') { targetC = (c + 1) % totalC; if (targetC === 0) targetR = (r + 1) % totalR; }
  else if (e.key === 'ArrowLeft') { targetC = (c - 1 + totalC) % totalC; if (targetC === totalC - 1) targetR = (r - 1 + totalR) % totalR; }
  else if (e.key === 'ArrowDown') { targetR = (r + 1) % totalR; }
  else if (e.key === 'ArrowUp') { targetR = (r - 1 + totalR) % totalR; }
  else if (e.key === 'Enter') { targetC = (c + 1) % totalC; if (targetC === 0) targetR = (r + 1) % totalR; }
  else return;

  const nextEl = document.getElementById(`inp-${label}-${targetR}-${targetC}`);
  if (nextEl) {
    e.preventDefault();
    nextEl.focus();
    nextEl.select();
  }
};

// Matrix fill helpers
window.fillMatrixPrompt = function(label, rows, cols) {
  const val = prompt(`Fill all cells of Matrix ${label.toUpperCase()} (${rows}×${cols}) with value:`, '1');
  if (val === null) return;
  const num = parseFloat(val);
  if (isNaN(num)) { showToast('❌', 'Please enter a valid numeric value'); return; }

  const dispRows = Math.min(rows, 6);
  const dispCols = Math.min(cols, 6);
  for (let i = 0; i < dispRows; i++) {
    for (let j = 0; j < dispCols; j++) {
      const el = document.getElementById(`inp-${label}-${i}-${j}`);
      if (el) el.value = num;
    }
  }

  matrixMemory[label] = { type: 'constant', value: num, rows, cols };
  showToast('✅', `Matrix ${label.toUpperCase()} (${rows}×${cols}) filled with ${num}`);
};

window.randomizeMatrix = function(label, rows, cols) {
  const dispRows = Math.min(rows, 6);
  const dispCols = Math.min(cols, 6);
  for (let i = 0; i < dispRows; i++) {
    for (let j = 0; j < dispCols; j++) {
      const el = document.getElementById(`inp-${label}-${i}-${j}`);
      if (el) el.value = Math.floor(Math.random() * 9) + 1;
    }
  }
  matrixMemory[label] = { type: 'random', rows, cols };
  showToast('🎲', `Matrix ${label.toUpperCase()} (${rows}×${cols}) randomized with values (1-9)`);
};

window.identityMatrix = function(label, rows, cols) {
  const dispRows = Math.min(rows, 6);
  const dispCols = Math.min(cols, 6);
  for (let i = 0; i < dispRows; i++) {
    for (let j = 0; j < dispCols; j++) {
      const el = document.getElementById(`inp-${label}-${i}-${j}`);
      if (el) el.value = (i === j) ? 1 : 0;
    }
  }
  matrixMemory[label] = { type: 'identity', rows, cols };
  showToast('⚡', `Matrix ${label.toUpperCase()} set to identity pattern`);
};

window.clearMatrix = function(label, rows, cols) {
  const dispRows = Math.min(rows, 6);
  const dispCols = Math.min(cols, 6);
  for (let i = 0; i < dispRows; i++) {
    for (let j = 0; j < dispCols; j++) {
      const el = document.getElementById(`inp-${label}-${i}-${j}`);
      if (el) el.value = 0;
    }
  }
  matrixMemory[label] = { type: 'constant', value: 0, rows, cols };
  showToast('🧹', `Matrix ${label.toUpperCase()} cleared to zeros`);
};

// Read matrix values from inputs or memory
function getMatrixValues(label, rows, cols) {
  const mem = matrixMemory[label];
  const mat = [];
  const dispRows = Math.min(rows, 6);
  const dispCols = Math.min(cols, 6);

  for (let i = 0; i < rows; i++) {
    mat.push(new Array(cols));
    for (let j = 0; j < cols; j++) {
      if (i < dispRows && j < dispCols) {
        const el = document.getElementById(`inp-${label}-${i}-${j}`);
        const val = el ? parseFloat(el.value) : 1;
        mat[i][j] = isNaN(val) ? 1 : val;
      } else {
        if (mem && mem.type === 'constant') mat[i][j] = mem.value;
        else if (mem && mem.type === 'identity') mat[i][j] = (i === j) ? 1 : 0;
        else if (mem && mem.type === 'random') mat[i][j] = Math.floor(Math.random() * 9) + 1;
        else mat[i][j] = 1;
      }
    }
  }
  return mat;
}

// Matrix multiplication: A (M x K) * B (K x N) -> C (M x N)
function multiplyMatrices(A, B, M, K, N) {
  const C = Array.from({ length: M }, () => Array(N).fill(0));
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      for (let j = 0; j < N; j++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

// ── Matrix Visualizer ─────────────────────────────────────────
function buildMatrixVisualizer(matA, matB) {
  const M = state.aRows;
  const K = state.aCols;
  const B_rows = state.bRows;
  const N = state.bCols;

  matrixWrapper.innerHTML = '';

  const isLarge = M > 8 || K > 8 || B_rows > 8 || N > 8;
  const dispM = Math.min(M, 8);
  const dispK = Math.min(K, 8);
  const dispB_rows = Math.min(B_rows, 8);
  const dispN = Math.min(N, 8);

  const maxDisp = Math.max(dispM, dispK, dispB_rows, dispN);
  const cellSize = maxDisp <= 3 ? 42 : maxDisp <= 6 ? 32 : 24;

  // A Matrix block (M x K)
  matrixWrapper.appendChild(buildVisualizerBlock('A', 'cell-a', M, K, dispM, dispK, cellSize, matA));

  // Multiplier sign
  const mulSign = document.createElement('div');
  mulSign.className = 'matrix-op-sign';
  mulSign.textContent = '×';
  matrixWrapper.appendChild(mulSign);

  // B Matrix block (B_rows x N)
  matrixWrapper.appendChild(buildVisualizerBlock('B', 'cell-b', B_rows, N, dispB_rows, dispN, cellSize, matB));

  // Equals sign
  const eqSign = document.createElement('div');
  eqSign.className = 'matrix-op-sign';
  eqSign.textContent = '=';
  matrixWrapper.appendChild(eqSign);

  // C Matrix block (M x N)
  if (K === B_rows) {
    matrixWrapper.appendChild(buildVisualizerBlock('C', 'cell-c', M, N, dispM, dispN, cellSize, null));
    if (isLarge) {
      matrixOverflow.style.display = 'block';
      matrixOverflow.textContent = `⚡ Displaying top-left 8×8 preview — Full multiplication is ${M}×${K} · ${B_rows}×${N} → ${M}×${N} (${M * N} elements)`;
    } else {
      matrixOverflow.style.display = 'none';
    }
  } else {
    const errWrap = document.createElement('div');
    errWrap.style.cssText = 'text-align:center; padding:12px; color:var(--red); font-family:"JetBrains Mono",monospace; font-size:12px;';
    errWrap.innerHTML = `❌ Undefined<br/><span style="font-size:10px; color:var(--text-muted);">Cols(A) ≠ Rows(B)</span>`;
    matrixWrapper.appendChild(errWrap);
    matrixOverflow.style.display = 'none';
  }
}

function buildVisualizerBlock(label, cellClass, fullRows, fullCols, dispRows, dispCols, cellSize, values) {
  const wrap = document.createElement('div');
  wrap.style.cssText = 'text-align:center; flex:1; min-width:0;';

  const lbl = document.createElement('div');
  lbl.className = 'matrix-label';
  const colors = { A: 'var(--blue-primary)', B: 'var(--purple)', C: 'var(--green)' };
  lbl.style.color = colors[label];
  lbl.textContent = `Matrix ${label} (${fullRows}×${fullCols})`;
  wrap.appendChild(lbl);

  const grid = document.createElement('div');
  grid.className = 'matrix-grid';
  grid.id = `matrix-${label.toLowerCase()}`;
  grid.style.gridTemplateColumns = `repeat(${dispCols}, ${cellSize}px)`;

  for (let i = 0; i < dispRows; i++) {
    for (let j = 0; j < dispCols; j++) {
      const cell = document.createElement('div');
      cell.className = `matrix-cell ${cellClass}`;
      cell.id = `cell-${label.toLowerCase()}-${i}-${j}`;
      cell.style.width = cell.style.height = `${cellSize}px`;
      cell.style.fontSize = cellSize <= 24 ? '9px' : cellSize <= 32 ? '11px' : '13px';

      if (label === 'C') {
        cell.textContent = '·';
      } else if (values && values[i] !== undefined && values[i][j] !== undefined) {
        cell.textContent = fmtVal(values[i][j]);
      } else {
        cell.textContent = '1';
      }
      grid.appendChild(cell);
    }
  }

  wrap.appendChild(grid);
  return wrap;
}

function fmtVal(v) {
  if (v === undefined || v === null) return '?';
  if (Number.isInteger(v)) return String(v);
  return String(parseFloat(v.toFixed(2)));
}

// ── Animate Matrix Computation ────────────────────────────────
async function animateMatrix(matC, M, N) {
  const CGrid = document.getElementById('matrix-c');
  if (!CGrid) return;

  matrixStatus.textContent = 'Computing…';
  matrixStatus.style.color = 'var(--orange)';

  const cells = CGrid.querySelectorAll('.matrix-cell');
  const dispRows = Math.min(M, 8);
  const dispCols = Math.min(N, 8);
  const p = state.threads;
  const rowsPerTh = Math.ceil(dispRows / p);

  for (let i = 0; i < dispRows; i++) {
    const threadIdx = Math.min(p - 1, Math.floor(i / rowsPerTh));
    setThreadStatus(threadIdx, 'active');

    for (let j = 0; j < dispCols; j++) {
      const cellIdx = i * dispCols + j;
      const cell = cells[cellIdx];
      if (!cell) continue;

      cell.classList.add('computing');
      await sleep(15 + Math.random() * 10);
      cell.classList.remove('computing');
      cell.classList.add('computed');

      cell.textContent = matC && matC[i] ? fmtVal(matC[i][j]) : '?';
    }

    setThreadStatus(threadIdx, 'done');
    await sleep(20);
  }

  matrixStatus.textContent = 'Done ✓';
  matrixStatus.style.color = 'var(--green)';
}

function setThreadStatus(idx, status) {
  const el = document.getElementById(`thread-status-${idx}`);
  if (el) el.className = `thread-status ${status}`;
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// ── Thread List ───────────────────────────────────────────────
const THREAD_COLORS = [
  'linear-gradient(90deg, #00d4ff, #0099bb)',
  'linear-gradient(90deg, #7c3aed, #a78bfa)',
  'linear-gradient(90deg, #00ff88, #00cc66)',
  'linear-gradient(90deg, #ff8c42, #ffd166)',
  'linear-gradient(90deg, #ff4d6d, #ff8c42)',
  'linear-gradient(90deg, #00d4ff, #7c3aed)',
  'linear-gradient(90deg, #00ff88, #00d4ff)',
  'linear-gradient(90deg, #a78bfa, #ff4d6d)',
  'linear-gradient(90deg, #ffd166, #00ff88)',
  'linear-gradient(90deg, #0099bb, #7c3aed)',
  'linear-gradient(90deg, #ff4d6d, #7c3aed)',
  'linear-gradient(90deg, #00cc66, #0099bb)',
  'linear-gradient(90deg, #ff8c42, #a78bfa)',
  'linear-gradient(90deg, #00d4ff, #ffd166)',
  'linear-gradient(90deg, #7c3aed, #00ff88)',
  'linear-gradient(90deg, #ff4d6d, #00d4ff)',
];

function buildThreadList() {
  threadList.innerHTML = '';
  const p = state.threads;
  for (let i = 0; i < p; i++) {
    const row = document.createElement('div');
    row.className = 'thread-row';
    row.id = `thread-row-${i}`;

    const id = document.createElement('div');
    id.className = 'thread-id';
    id.textContent = `TH-${String(i).padStart(2, '0')}`;

    const track = document.createElement('div');
    track.className = 'thread-bar-track';

    const fill = document.createElement('div');
    fill.className = 'thread-bar-fill';
    fill.id = `thread-bar-${i}`;
    fill.style.background = THREAD_COLORS[i % THREAD_COLORS.length];
    fill.style.width = '0%';
    track.appendChild(fill);

    const pct = document.createElement('div');
    pct.className = 'thread-pct';
    pct.id = `thread-pct-${i}`;
    pct.textContent = '0%';

    const status = document.createElement('div');
    status.className = 'thread-status idle';
    status.id = `thread-status-${i}`;

    row.appendChild(id);
    row.appendChild(track);
    row.appendChild(pct);
    row.appendChild(status);
    threadList.appendChild(row);
  }
  threadsActive.textContent = `0 / ${p} active`;
}

async function animateThreadBars(durationMs) {
  const p = state.threads;

  for (let i = 0; i < p; i++) {
    const bar = document.getElementById(`thread-bar-${i}`);
    const statEl = document.getElementById(`thread-status-${i}`);
    if (bar) { bar.style.width = '0%'; bar.classList.add('active'); }
    if (statEl) statEl.className = 'thread-status active';
  }
  threadsActive.textContent = `${p} / ${p} active`;

  const step = 50;
  const steps = Math.ceil(durationMs / step);

  for (let s = 0; s <= steps; s++) {
    const progress = s / steps;
    for (let i = 0; i < p; i++) {
      const bar = document.getElementById(`thread-bar-${i}`);
      const pctEl = document.getElementById(`thread-pct-${i}`);
      const tp = Math.min(1, Math.max(0, progress * (1 + 0.08 * (Math.random() - 0.5))));
      const pct = Math.floor(tp * 100);
      if (bar) bar.style.width = `${pct}%`;
      if (pctEl) pctEl.textContent = `${pct}%`;
    }
    await sleep(step);
  }

  for (let i = 0; i < p; i++) {
    const bar = document.getElementById(`thread-bar-${i}`);
    const pctEl = document.getElementById(`thread-pct-${i}`);
    const statEl = document.getElementById(`thread-status-${i}`);
    if (bar) { bar.style.width = '100%'; bar.classList.remove('active'); }
    if (pctEl) pctEl.textContent = '100%';
    if (statEl) statEl.className = 'thread-status done';
  }
  threadsActive.textContent = `0 / ${p} active`;
}

// ── Speedup Reference Table ───────────────────────────────────
function buildSpeedupTable() {
  const threadCounts = [1, 2, 4, 8, 12, 16];
  const { aRows: M, aCols: K, bCols: N } = state;
  const ts = seqTimeRect(M, K, N);
  speedupTable.innerHTML = '';

  threadCounts.forEach(p => {
    const tp = parTimeRect(M, K, N, p);
    const sp = (ts / tp).toFixed(2);
    const eff = ((ts / tp) / p * 100).toFixed(0);
    const color = p === 1 ? 'var(--text-muted)'
                : p <= 4 ? 'var(--green)'
                : p <= 8 ? 'var(--blue-primary)'
                : '#a78bfa';

    const item = document.createElement('div');
    item.className = 'speedup-item';
    item.innerHTML = `
      <div class="speedup-threads">${p} thread${p !== 1 ? 's' : ''}</div>
      <div class="speedup-val" style="color:${color}">${sp}×</div>
      <div class="speedup-eff">Eff: ${eff}%</div>
    `;
    speedupTable.appendChild(item);
  });
}

// ── Stat Cards ────────────────────────────────────────────────
function updateStats(timeMs, speedup, efficiency) {
  animateCounter(statTime, 0, timeMs, 1200, v => {
    if (timeMs < 0.001) return v.toExponential(2);
    if (timeMs < 0.1) return v.toFixed(4);
    if (timeMs < 1) return v.toFixed(3);
    if (timeMs < 100) return v.toFixed(2);
    return v.toFixed(1);
  });
  animateCounter(statSpeedup, 0, speedup, 1200, v => v.toFixed(2));
  animateCounter(statEff, 0, efficiency, 1200, v => v.toFixed(1));

  setTimeout(() => {
    barTime.style.width    = `${Math.min(100, (timeMs / 5000) * 100)}%`;
    barSpeedup.style.width = `${Math.min(100, (speedup / 16) * 100)}%`;
    barEff.style.width     = `${Math.min(100, efficiency)}%`;
  }, 200);
}

function animateCounter(el, from, to, duration, format) {
  const start = performance.now();
  function tick(now) {
    const t = Math.min(1, (now - start) / duration);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = format(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── Log Terminal ──────────────────────────────────────────────
function logTime() {
  const e = (Date.now() - state.logStart) / 1000;
  const m = String(Math.floor(e / 60)).padStart(2, '0');
  const s = (e % 60).toFixed(3).padStart(6, '0');
  return `${m}:${s}`;
}

function log(type, msg, extra = '') {
  const line = document.createElement('div');
  line.className = 'terminal-line';
  const tagClass = { ok: 'terminal-tag-ok', run: 'terminal-tag-run', info: 'terminal-tag-info', err: 'terminal-tag-err' }[type] || 'terminal-tag-info';
  const tagLabel = { ok: '[OK]', run: '[RUN]', info: '[INFO]', err: '[ERR]' }[type] || '[INFO]';
  line.innerHTML = `
    <span class="terminal-time">${logTime()}</span>
    <span class="${tagClass}">${tagLabel}</span>
    <span class="terminal-msg">${msg}${extra ? ` <span class="hl">${extra}</span>` : ''}</span>
  `;
  execLog.appendChild(line);
  execLog.scrollTop = execLog.scrollHeight;
}

btnClearLog.addEventListener('click', () => {
  execLog.innerHTML = '';
  log('info', 'Log cleared.');
});

// ── Toast ─────────────────────────────────────────────────────
let toastTimer = null;
function showToast(icon, msg, duration = 3000) {
  toastIcon.textContent = icon;
  toastMsg.textContent  = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), duration);
}

function formatSI(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'G';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toFixed(0);
}

// ── Run Parallel Multiplication ───────────────────────────────
btnRun.addEventListener('click', async () => {
  if (state.running) return;

  const M = state.aRows;
  const K = state.aCols;
  const B_rows = state.bRows;
  const N = state.bCols;
  const p = state.threads;

  if (K !== B_rows) {
    showToast('⚠️', `Cannot multiply: Cols of A (${K}) ≠ Rows of B (${B_rows})`);
    log('err', `Dimension mismatch: A is ${M}×${K}, B is ${B_rows}×${N}.`, 'Multiplication undefined');
    return;
  }

  state.running = true;
  state.runCount++;
  btnRun.disabled = true;
  btnRun.innerHTML = `<div class="spinner"></div><span>Computing…</span>`;

  // Collect manual values from input grids
  const matA = getMatrixValues('a', M, K);
  const matB = getMatrixValues('b', B_rows, N);
  const matC = multiplyMatrices(matA, matB, M, K, N);

  log('run', `Executing A(${M}×${K}) × B(${K}×${N}) → C(${M}×${N}) with`, `${p} thread${p > 1 ? 's' : ''}`);
  log('info', `FLOPs: ${formatSI(calcFlops(M, K, N))}, Parallel rows: ${M}, Rows/thread: ${Math.ceil(M / p)}`);

  // Log matrix contents for transparency
  if (M <= 5 && K <= 5) {
    matA.forEach((row, idx) => log('info', `Matrix A[${idx}] = [${row.join(', ')}]`));
  }
  if (B_rows <= 5 && N <= 5) {
    matB.forEach((row, idx) => log('info', `Matrix B[${idx}] = [${row.join(', ')}]`));
  }

  // Update visualizer with actual A and B values
  buildMatrixVisualizer(matA, matB);

  // Timing
  const ts = seqTimeRect(M, K, N);
  const tp = parTimeRect(M, K, N, p);
  const speedup    = ts / tp;
  const efficiency = (speedup / p) * 100;
  const timeMs     = tp * 1000;

  const animDuration = Math.min(2200, Math.max(500, 450 + (M * N) * 40));

  await Promise.all([
    animateMatrix(matC, M, N),
    animateThreadBars(animDuration),
  ]);

  updateStats(timeMs, speedup, efficiency);
  drawChart();

  state.history.push({ M, K, N, threads: p, timeMs, speedup, efficiency });

  log('ok', `Multiplication completed in ${timeMs.toFixed(4)} ms — Speedup:`, `${speedup.toFixed(2)}×`);
  log('ok', `Efficiency:`, `${efficiency.toFixed(1)}%`);

  if (M <= 6 && N <= 6) {
    matC.forEach((row, idx) => log('info', `Result C[${idx}] = [${row.map(v => fmtVal(v)).join(', ')}]`));
  }

  showToast('✅', `A(${M}×${K}) × B(${K}×${N}) computed! Result C is ${M}×${N}`);

  state.running = false;
  btnRun.disabled = false;
  btnRun.innerHTML = `<span class="btn-icon">▶</span><span>Run Parallel Multiplication</span>`;
});

// ── Reset ─────────────────────────────────────────────────────
btnReset.addEventListener('click', () => {
  dimARows.value = 2;
  dimACols.value = 3;
  dimBRows.value = 3;
  dimBCols.value = 4;
  sliderThreads.value = 4;
  linkDimsCb.checked = true;

  updateDimensions();

  statTime.textContent = statSpeedup.textContent = statEff.textContent = '—';
  barTime.style.width = barSpeedup.style.width = barEff.style.width = '0%';
  matrixStatus.textContent = 'Idle';
  matrixStatus.style.color = 'var(--text-muted)';

  log('info', 'Reset configuration to defaults: A(2×3), B(3×4), 4 threads');
  showToast('↺', 'Reset to defaults');
});

// ── Performance Chart (Canvas) ────────────────────────────────
let chartCanvas, ctx;

function initChart() {
  chartCanvas = document.getElementById('perf-chart');
  ctx = chartCanvas.getContext('2d');
  resizeChart();
  drawChart();
}

function resizeChart() {
  const dpr = window.devicePixelRatio || 1;
  const rect = chartCanvas.parentElement.getBoundingClientRect();
  if (ctx.resetTransform) ctx.resetTransform();
  chartCanvas.width  = rect.width * dpr;
  chartCanvas.height = 220 * dpr;
  ctx.scale(dpr, dpr);
  chartCanvas.style.width  = rect.width + 'px';
  chartCanvas.style.height = '220px';
}

function drawChart() {
  if (!ctx || !chartCanvas) return;
  const dpr = window.devicePixelRatio || 1;
  const W   = chartCanvas.width / dpr;
  const H   = chartCanvas.height / dpr;

  ctx.clearRect(0, 0, W, H);

  const pad = { top: 20, right: 20, bottom: 40, left: 60 };
  const cW  = W - pad.left - pad.right;
  const cH  = H - pad.top  - pad.bottom;

  const { aRows: M, aCols: K, bCols: N } = state;
  const threads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
  const ts      = seqTimeRect(M, K, N);
  const seqMs   = threads.map(() => ts * 1000);
  const parMs   = threads.map(p => parTimeRect(M, K, N, p) * 1000);
  const speedups = threads.map(p => ts / parTimeRect(M, K, N, p));

  const maxTime    = Math.max(...seqMs, ...parMs) * 1.15;
  const maxSpeedup = Math.max(...speedups) * 1.3;

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (cH / 5) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cW, y); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = 'rgba(0,212,255,0.2)'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + cH);
  ctx.lineTo(pad.left + cW, pad.top + cH);
  ctx.stroke();

  // X-axis labels
  ctx.fillStyle = 'rgba(125,165,194,0.8)';
  ctx.font = `10px 'JetBrains Mono', monospace`;
  ctx.textAlign = 'center';
  [1, 4, 8, 12, 16].forEach(t => {
    ctx.fillText(t, pad.left + ((t - 1) / 15) * cW, pad.top + cH + 18);
  });
  ctx.fillStyle = 'rgba(125,165,194,0.6)';
  ctx.fillText('Threads', pad.left + cW / 2, pad.top + cH + 34);

  // Y-axis label
  ctx.save();
  ctx.translate(14, pad.top + cH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(125,165,194,0.6)';
  ctx.font = `10px 'JetBrains Mono', monospace`;
  ctx.fillText('Time (ms)', 0, 0);
  ctx.restore();

  // Y-axis ticks
  ctx.textAlign = 'right';
  ctx.fillStyle = 'rgba(125,165,194,0.6)';
  ctx.font = `9px 'JetBrains Mono', monospace`;
  for (let i = 0; i <= 4; i++) {
    const val = maxTime * (1 - i / 4);
    const y   = pad.top + (cH / 4) * i;
    ctx.fillText(val < 0.001 ? val.toExponential(1) : val.toFixed(3), pad.left - 6, y + 3);
  }

  function drawLine(data, maxVal, color, dash = []) {
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round'; ctx.setLineDash(dash);
    data.forEach((v, i) => {
      const x = pad.left + (i / (data.length - 1)) * cW;
      const y = pad.top + cH - (v / maxVal) * cH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke(); ctx.setLineDash([]);
  }

  // Fill area under parallel line
  ctx.beginPath();
  parMs.forEach((v, i) => {
    const x = pad.left + (i / (parMs.length - 1)) * cW;
    const y = pad.top + cH - (v / maxTime) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(pad.left + cW, pad.top + cH);
  ctx.lineTo(pad.left,      pad.top + cH);
  ctx.closePath();
  ctx.fillStyle = 'rgba(0,212,255,0.06)'; ctx.fill();

  drawLine(seqMs,    maxTime,    '#7c3aed', [6, 3]);
  drawLine(parMs,    maxTime,    '#00d4ff');
  drawLine(speedups, maxSpeedup, '#00ff88');

  // Ideal speedup reference
  ctx.beginPath(); ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1; ctx.setLineDash([3, 5]);
  threads.forEach((p, i) => {
    const x = pad.left + (i / (threads.length - 1)) * cW;
    const y = pad.top + cH - (p / maxSpeedup) * cH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke(); ctx.setLineDash([]);

  // Marker at selected thread count
  const ti = state.threads - 1;
  const cx = pad.left + (ti / 15) * cW;
  ctx.beginPath(); ctx.strokeStyle = 'rgba(0,212,255,0.2)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.moveTo(cx, pad.top); ctx.lineTo(cx, pad.top + cH);
  ctx.stroke(); ctx.setLineDash([]);

  // Dots
  [[parMs, maxTime, '#00d4ff'], [seqMs, maxTime, '#7c3aed'], [speedups, maxSpeedup, '#00ff88']].forEach(([data, max, color]) => {
    const cy = pad.top + cH - (data[ti] / max) * cH;
    ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = 1.5; ctx.stroke();
  });
}

// ── Keyboard shortcuts ────────────────────────────────────────
document.addEventListener('keydown', e => {
  const tag = document.activeElement.tagName;
  if (tag === 'INPUT') return;
  if (e.key === 'Enter' && !state.running) btnRun.click();
  if (e.key === 'r' && !e.ctrlKey && !state.running) btnReset.click();
});

// Resize handler
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (chartCanvas && ctx) { resizeChart(); drawChart(); }
  }, 100);
});

// ── Init ──────────────────────────────────────────────────────
function init() {
  updateDimensions();
  initChart();
  log('ok', 'Dashboard ready. Configure dimensions (e.g. 2×3 & 3×4), enter values, and press', 'Run');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  setTimeout(init, 0);
}
