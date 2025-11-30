/* Pixel-grid painter that stitches fast mouse moves via line interpolation.
   - Uses Bresenham to fill gaps between sampled cursor grid coordinates.
   - Paints with a 1-pixel brush (you can modify paintAtCell brushSize param if needed).
   - Left-click paints (1), right-click erases (0).
   - Sends JSON { pixels: [[0|1,...], ...] } to /upload_and_compute_pixels.
*/

// DOM refs
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const gridSizeSelect = document.getElementById('gridSize');
const info = document.getElementById('info');

let GRID = 64;                // current grid side
let PIXELS = [];              // 2D array rows x cols (rows = GRID)
let cellPx = 8;               // pixel display size
let mouseDown = false;
let drawColor = 1;            // 1 -> paint, 0 -> erase
let lastCell = null;          // last painted cell {r,c}

// initialize grid & drawing
function initGrid(n) {
  GRID = n;
  PIXELS = Array.from({length: GRID}, () => new Uint8Array(GRID));
  cellPx = Math.floor(Math.min(canvas.width, canvas.height) / GRID);
  drawGrid();
}

function drawGrid() {
  // clear
  ctx.fillStyle = '#fff';
  ctx.fillRect(0,0,canvas.width, canvas.height);

  // compute grid origin (center)
  const totalSize = cellPx * GRID;
  const offsetX = Math.floor((canvas.width - totalSize) / 2);
  const offsetY = Math.floor((canvas.height - totalSize) / 2);

  // draw pixels
  for (let r=0;r<GRID;r++){
    for (let c=0;c<GRID;c++){
      if (PIXELS[r][c]) {
        ctx.fillStyle = '#000';
        ctx.fillRect(offsetX + c*cellPx, offsetY + r*cellPx, cellPx, cellPx);
      } else {
        ctx.fillStyle = '#fff';
        ctx.fillRect(offsetX + c*cellPx, offsetY + r*cellPx, cellPx, cellPx);
      }
    }
  }

  // faint grid lines
  ctx.strokeStyle = 'rgba(0,0,0,0.06)';
  ctx.lineWidth = 1;
  for (let i=0;i<=GRID;i++){
    const x = offsetX + i*cellPx + 0.5;
    ctx.beginPath(); ctx.moveTo(x, offsetY); ctx.lineTo(x, offsetY + totalSize); ctx.stroke();
    const y = offsetY + i*cellPx + 0.5;
    ctx.beginPath(); ctx.moveTo(offsetX, y); ctx.lineTo(offsetX + totalSize, y); ctx.stroke();
  }
}

// Map mouse event to grid cell (returns {r,c} or null)
function cellFromEvent(e) {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const totalSize = cellPx * GRID;
  const offsetX = Math.floor((canvas.width - totalSize) / 2);
  const offsetY = Math.floor((canvas.height - totalSize) / 2);
  const cx = Math.floor((x - offsetX) / cellPx);
  const cy = Math.floor((y - offsetY) / cellPx);
  if (cx < 0 || cx >= GRID || cy < 0 || cy >= GRID) return null;
  return { r: cy, c: cx };
}

// Bresenham integer line algorithm between two cells (r1,c1) -> (r2,c2)
// yields all integer grid cells along the line
function bresenhamLine(r1, c1, r2, c2) {
  const cells = [];

  let x0 = c1, y0 = r1, x1 = c2, y1 = r2;
  const dx = Math.abs(x1 - x0);
  const sx = (x0 < x1) ? 1 : -1;
  const dy = -Math.abs(y1 - y0);
  const sy = (y0 < y1) ? 1 : -1;
  let err = dx + dy;  // err = dx - dy in original form

  while (true) {
    cells.push({ r: y0, c: x0 });
    if (x0 === x1 && y0 === y1) break;
    const e2 = 2*err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
  return cells;
}

// Paint a brush-sized square centered on (r,c) with mode (1 or 0)
function paintAtCell(r, c, brushSize, mode) {
  const half = Math.floor(brushSize / 2);
  for (let dr = -half; dr <= half; dr++) {
    for (let dc = -half; dc <= half; dc++) {
      const rr = r + dr, cc = c + dc;
      if (rr >= 0 && rr < GRID && cc >= 0 && cc < GRID) {
        PIXELS[rr][cc] = mode ? 1 : 0;
      }
    }
  }
}

// Paint along line between two grid cells, respecting brush size and mode.
// Uses Bresenham to sample intermediate cells.
function paintLineBetween(a, b, brushSize, mode) {
  if (!a || !b) return;
  const list = bresenhamLine(a.r, a.c, b.r, b.c);
  for (let p of list) {
    paintAtCell(p.r, p.c, brushSize, mode);
  }
  drawGrid();
}

// Set up pointer events to use pencil interpolation
canvas.addEventListener('mousedown', (e) => {
  // left button -> paint, right -> erase
  if (e.button === 2) drawColor = 0; else drawColor = 1;
  mouseDown = true;
  const cell = cellFromEvent(e);
  lastCell = cell;
  if (cell) {
    paintAtCell(cell.r, cell.c, 1, drawColor);
    drawGrid();
  }
  // prevent context menu on right click
  if (e.button === 2) e.preventDefault();
});
canvas.addEventListener('mousemove', (e) => {
  if (!mouseDown) return;
  const cell = cellFromEvent(e);
  if (!cell) return;
  // If lastCell is null (rare), just paint this cell
  if (!lastCell) {
    paintAtCell(cell.r, cell.c, 1, drawColor);
    lastCell = cell;
    drawGrid();
    return;
  }
  // If cell changed (or even if same), draw interpolated line between last and this cell
  if (cell.r !== lastCell.r || cell.c !== lastCell.c) {
    paintLineBetween(lastCell, cell, 1, drawColor);
    lastCell = cell;
  } else {
    // still paint (keeps continuous stroke)
    paintAtCell(cell.r, cell.c, 1, drawColor);
    drawGrid();
  }
});

window.addEventListener('mouseup', () => {
  mouseDown = false;
  lastCell = null;
});

// prevent context menu to allow right-click erase easily
canvas.addEventListener('contextmenu', (e) => e.preventDefault());

// right mouse sets erase mode on mousedown; we already handle e.button !== 2 above.
// ensure drawColor resets when mouseup (global listener)
window.addEventListener('mousedown', (e) => {
  if (e.target !== canvas) return;
  if (e.button === 2) drawColor = 0;
});
window.addEventListener('mouseup', () => { drawColor = 1; });

// clear
document.getElementById('clearBtn').addEventListener('click', () => {
  initGrid(GRID);
  info.innerText = '';
});

// grid size change
gridSizeSelect.addEventListener('change', (e) => {
  initGrid(Number(e.target.value));
});

// Upload pixels as JSON to endpoint /upload_and_compute_pixels (server must have it)
document.getElementById('computeBtn').addEventListener('click', async () => {
  const payload = { pixels: [] };
  for (let r=0;r<GRID;r++){
    payload.pixels.push(Array.from(PIXELS[r]));
  }
  info.innerText = 'Uploading pixels...';
  try {
    const resp = await fetch('http://localhost:8000/upload_and_compute_pixels', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(text || resp.statusText);
    }
    const data = await resp.json();
    currentWalk = data.positions;
    info.innerText = `nodes=${data.n_nodes} edges=${data.n_edges} walk_len=${data.walk.length} total=${data.total_weight.toFixed(3)}`;
    drawResultGraph(data);
  } catch (err) {
    info.innerText = 'Upload error: ' + err;
  }
});

// Helper: draw a highlighted marker for the start node (posXY: [x, y] in canvas coords)
function drawStartMarkerAt(posXY) {
  const r = Math.max(6, cellPx * 0.6);
  ctx.beginPath();
  ctx.fillStyle = "#ff0000";       // bright red fill
  ctx.strokeStyle = "#880000";     // dark red border
  ctx.lineWidth = 2;
  ctx.arc(posXY[0], posXY[1], r, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

// draw returned graph overlay & animate
function drawResultGraph(data){
  drawGrid();
  const totalSize = cellPx * GRID;
  const offsetX = Math.floor((canvas.width - totalSize) / 2);
  const offsetY = Math.floor((canvas.height - totalSize) / 2);

  // nodes: {id: [x,y]} where x,y are grid coords (col,row)
  const nodes = data.nodes || {};
  const edges = data.edges || [];

  // build pixel-to-canvas mapping
  const pos = {};
  for (const [k, p] of Object.entries(nodes)) {
    const cx = offsetX + (p[0] + 0.5) * cellPx;
    const cy = offsetY + (p[1] + 0.5) * cellPx;
    pos[k] = [cx, cy];
  }

  // draw edges
  ctx.lineWidth = 2;
  for (const e of edges) {
    const u = String(e[0]), v = String(e[1]);
    if (!(u in pos) || !(v in pos)) continue;
    const a = pos[u], b = pos[v];
    ctx.beginPath();
    ctx.strokeStyle = '#2196f3';
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  // draw nodes
  for (const [k, p] of Object.entries(pos)) {
    ctx.beginPath();
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = '#000';
    ctx.arc(p[0], p[1], Math.max(2, cellPx*0.28), 0, Math.PI*2);
    ctx.fill();
    ctx.stroke();
  }

  // highlight start vertex if walk provided
  const startId = (data.walk && data.walk.length > 0) ? String(data.walk[0]) : null;
  if (startId && pos[startId]) {
    drawStartMarkerAt(pos[startId]);
  }

  if (data.walk && data.walk.length > 1) {
    animateWalkOnCanvas(pos, data.walk, data.edges || []);
  }
}

// animate walk (simple)
function animateWalkOnCanvas(pos, walk, edges) {
  const steps = [];
  for (let i=0;i<walk.length-1;i++){
    const a = pos[String(walk[i])], b = pos[String(walk[i+1])];
    if (!a || !b) continue;
    steps.push([a,b]);
  }
  let s = 0;
  function frame(){
    // redraw background for clarity
    drawGrid();

    // draw static graph overlay (edges)
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#ddd';
    for (const e of edges) {
      const u = String(e[0]), v = String(e[1]);
      if (!(u in pos) || !(v in pos)) continue;
      const a = pos[u], b = pos[v];
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }

    // draw trail
    for (let i=0;i<s;i++){
      const seg = steps[i]; if(!seg) continue;
      const t = i / Math.max(1, steps.length-1);
      ctx.strokeStyle = `hsl(${t*360},80%,50%)`;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(seg[0][0], seg[0][1]);
      ctx.lineTo(seg[1][0], seg[1][1]);
      ctx.stroke();
    }
    // current marker
    if (steps.length > 0) {
      const cur = steps[Math.min(s, steps.length-1)][1];
      ctx.beginPath(); ctx.fillStyle='red'; ctx.arc(cur[0], cur[1], Math.max(4, cellPx*0.32), 0, Math.PI*2); ctx.fill();
    }

    // keep start marker visible on top (draw last)
    if (walk && walk.length > 0) {
      const startId = String(walk[0]);
      if (startId && pos[startId]) {
        drawStartMarkerAt(pos[startId]);
      }
    }

    s++;
    if (s <= steps.length) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// initialize default grid
initGrid(Number(gridSizeSelect.value));
