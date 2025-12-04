// draw.js
// Vector stroke canvas + requestAnimationFrame rendering loop
// Placeholder smoothing function (no-op).
// Sends strokes to POST /upload_polylines
// Animation: arc-length (distance) based walk animation + rainbow trail + start marker.
// Added: optional client-side intersection-splitting preview; server-side splitting requested via payload.

(() => {
  const canvas = document.getElementById('canvas');
  const info = document.getElementById('info');
  const clearBtn = document.getElementById('clearBtn');
  const computeBtn = document.getElementById('computeBtn');
  const sendBtn = document.getElementById('sendBtn');
  const tolInput = document.getElementById('dpTolerance');

  const ctx = canvas.getContext('2d');

  // Application state
  let strokes = [];            // array of strokes; each stroke is an array of {x,y}
  let currentStroke = null;    // building stroke (array) while pointerdown
  let isDrawing = false;

  // Server / walk overlay & animation state
  window.lastServerResponse = null;
  let walkPositions = [];      // positions (copied from server) - array of {x,y}
  let segLengths = [];         // length per segment
  let nodeCum = [];            // cumulative distance at each node (nodeCum[0]=0)
  let totalLength = 0;
  let distanceTravelled = 0;   // in pixels
  let animRunning = false;
  let lastFrameTime = null;
  let animSpeed = 2000;         // default speed pixels/second (tunable)

  // CLIENT-SPLIT: preview graph produced by splitting intersections locally (preview only)
  window.mergedGraphForPreview = null; // { nodes: [{x,y,id}], edges: [{u,v}] }

  // Handle high-DPI: make backing store scale to devicePixelRatio
  function resizeCanvasToDisplaySize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const cssW = Math.max(1, Math.floor(rect.width));
    const cssH = Math.max(1, Math.floor(rect.height));

    const needResize = canvas.width !== cssW * dpr || canvas.height !== cssH * dpr;
    if (needResize) {
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
      // scale drawing ops so we can use CSS pixels coordinates
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  // Convert client coords to canvas CSS-pixel coordinates
  function getCanvasPointFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    return { x, y };
  }

  // --- placeholder smoothing function (no-op) ---
  function smoothStroke(points) {
    return points;
  }

  // --- pointer handlers ---
  canvas.style.touchAction = 'none'; // prevent panning on touch devices

  canvas.addEventListener('pointerdown', (e) => {
    if (e.button && e.button !== 0) return;
    canvas.setPointerCapture(e.pointerId);
    isDrawing = true;
    currentStroke = [];
    const pt = getCanvasPointFromEvent(e);
    currentStroke.push(pt);
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!isDrawing || !currentStroke) return;
    const pt = getCanvasPointFromEvent(e);
    const last = currentStroke[currentStroke.length - 1];
    if (!last || last.x !== pt.x || last.y !== pt.y) {
      currentStroke.push(pt);
    }
  });

  function endStroke(e) {
    if (!isDrawing) return;
    isDrawing = false;
    if (currentStroke && currentStroke.length > 0) {
      strokes.push(currentStroke.slice());
    }
    currentStroke = null;
    try { canvas.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
  }

  canvas.addEventListener('pointerup', endStroke);
  canvas.addEventListener('pointercancel', endStroke);
  window.addEventListener('blur', () => { isDrawing = false; currentStroke = null; });

  // --- geometry helpers (segment intersection + splitting) ---
  const EPS = 1e-9;
  function almostEqual(a,b,tol=1e-6){ return Math.abs(a-b) <= tol; }
  function keyForPoint(p){ return `${p.x.toFixed(6)},${p.y.toFixed(6)}`; }

  // Segment intersection test (proper intersection or at endpoints)
  // Returns {x,y, t, u} where t is param on segment AB, u is param on segment CD, or null if no intersection.
  function segmentIntersect(A, B, C, D) {
    const ax = A.x, ay = A.y;
    const bx = B.x, by = B.y;
    const cx = C.x, cy = C.y;
    const dx = D.x, dy = D.y;

    const r_x = bx - ax, r_y = by - ay;
    const s_x = dx - cx, s_y = dy - cy;

    const denom = r_x * s_y - r_y * s_x;
    if (Math.abs(denom) < EPS) {
      const cross = (cx - ax) * r_y - (cy - ay) * r_x;
      if (Math.abs(cross) < EPS) {
        function onSeg(P, Q, R) {
          return Math.min(P.x, R.x) - EPS <= Q.x && Q.x <= Math.max(P.x, R.x) + EPS &&
                 Math.min(P.y, R.y) - EPS <= Q.y && Q.y <= Math.max(P.y, R.y) + EPS;
        }
        if (onSeg(A, C, B)) return { x: C.x, y: C.y, t: ((Math.abs(r_x) > Math.abs(r_y)) ? ((C.x - ax) / r_x) : ((C.y - ay) / r_y)), u: 0 };
        if (onSeg(A, D, B)) return { x: D.x, y: D.y, t: ((Math.abs(r_x) > Math.abs(r_y)) ? ((D.x - ax) / r_x) : ((D.y - ay) / r_y)), u: 1 };
        if (onSeg(C, A, D)) return { x: A.x, y: A.y, t: 0, u: ((Math.abs(s_x) > Math.abs(s_y)) ? ((A.x - cx) / s_x) : ((A.y - cy) / s_y)) };
        if (onSeg(C, B, D)) return { x: B.x, y: B.y, t: 1, u: ((Math.abs(s_x) > Math.abs(s_y)) ? ((B.x - cx) / s_x) : ((B.y - cy) / s_y)) };
        return null;
      }
      return null;
    }

    const invDen = 1 / denom;
    const cx_ax = cx - ax, cy_ay = cy - ay;
    const t = (cx_ax * s_y - cy_ay * s_x) * invDen;
    const u = (cx_ax * r_y - cy_ay * r_x) * invDen;

    if (t >= -EPS && t <= 1 + EPS && u >= -EPS && u <= 1 + EPS) {
      const ix = ax + t * r_x;
      const iy = ay + t * r_y;
      return { x: ix, y: iy, t: Math.max(0, Math.min(1, t)), u: Math.max(0, Math.min(1, u)) };
    }
    return null;
  }

  // Build raw segment list from strokes: returns array of {a,b, strokeIndex, segIndex}
  function buildSegmentListFromStrokes(strokesArr) {
    const segs = [];
    for (let si = 0; si < strokesArr.length; si++) {
      const s = strokesArr[si];
      for (let i = 0; i < s.length - 1; i++) {
        segs.push({ a: s[i], b: s[i+1], strokeIndex: si, segIndex: i });
      }
    }
    return segs;
  }

  // Given segments, find intersections and split each segment at its local parameter t values
  // Returns array of subsegments: {a:{x,y}, b:{x,y}, originalStrokeIndex, originalSegIndex}
  function splitSegmentsAtIntersections(segs) {
    const tLists = new Array(segs.length);
    for (let i = 0; i < segs.length; i++) tLists[i] = [0, 1];

    for (let i = 0; i < segs.length; i++) {
      for (let j = i + 1; j < segs.length; j++) {
        const I = segmentIntersect(segs[i].a, segs[i].b, segs[j].a, segs[j].b);
        if (I) {
          const ti = Math.max(0, Math.min(1, I.t));
          const uj = Math.max(0, Math.min(1, I.u));
          if (!tLists[i].some(v => almostEqual(v, ti))) tLists[i].push(ti);
          if (!tLists[j].some(v => almostEqual(v, uj))) tLists[j].push(uj);
        }
      }
    }

    const outSubsegments = [];
    for (let i = 0; i < segs.length; i++) {
      const seg = segs[i];
      const ts = tLists[i].slice().sort((a,b) => a-b);
      for (let k = 0; k < ts.length - 1; k++) {
        const t0 = ts[k], t1 = ts[k+1];
        if (t1 - t0 < 1e-6) continue;
        const a = { x: seg.a.x + (seg.b.x - seg.a.x) * t0, y: seg.a.y + (seg.b.y - seg.a.y) * t0 };
        const b = { x: seg.a.x + (seg.b.x - seg.a.x) * t1, y: seg.a.y + (seg.b.y - seg.a.y) * t1 };
        if (Math.hypot(b.x - a.x, b.y - a.y) < 1e-6) continue;
        outSubsegments.push({ a, b, originalStrokeIndex: seg.strokeIndex, originalSegIndex: seg.segIndex });
      }
    }
    return outSubsegments;
  }

  // Build node list and edge index from subsegments (preview)
  function buildGraphFromSubsegments(subsegs) {
    const nodeIndex = new Map(); // key -> index
    const nodes = [];
    const edges = []; // [u,v]
    function addNode(p) {
      const k = keyForPoint(p);
      const existing = nodeIndex.get(k);
      if (existing !== undefined) return existing;
      const idx = nodes.length;
      nodes.push({ x: p.x, y: p.y });
      nodeIndex.set(k, idx);
      return idx;
    }
    for (const s of subsegs) {
      const u = addNode(s.a);
      const v = addNode(s.b);
      if (u === v) continue;
      edges.push([u, v]);
    }
    return { nodes, edges };
  }

  // Public helper: produce split subsegments (2-point polylines) and preview graph
  function computeClientSplitAndPreview() {
    const segs = buildSegmentListFromStrokes(strokes);
    if (segs.length === 0) {
      window.mergedGraphForPreview = null;
      return { subsegments: [], graph: null };
    }
    const subsegs = splitSegmentsAtIntersections(segs);
    const graph = buildGraphFromSubsegments(subsegs);
    window.mergedGraphForPreview = graph;
    return { subsegments: subsegs, graph: graph };
  }

  // --- animation helpers (arc-length based) ---
  function precomputeLengths(positions) {
    segLengths = [];
    nodeCum = [];
    totalLength = 0;
    if (!positions || positions.length === 0) {
      return;
    }
    nodeCum.push(0);
    for (let i = 0; i < positions.length - 1; i++) {
      const a = positions[i], b = positions[i + 1];
      const L = Math.hypot(b.x - a.x, b.y - a.y);
      segLengths.push(L);
      totalLength += L;
      nodeCum.push(totalLength);
    }
  }

  function stopWalkAnimation() {
    animRunning = false;
    distanceTravelled = 0;
    lastFrameTime = null;
    walkPositions = [];
    segLengths = [];
    nodeCum = [];
    totalLength = 0;
    // keep lastServerResponse for static display if needed
  }

  function startWalkAnimation(positions) {
    stopWalkAnimation(); // stop previous
    if (!positions || positions.length < 2) return;
    walkPositions = positions.map(p => ({ x: Number(p.x), y: Number(p.y) }));
    precomputeLengths(walkPositions);
    distanceTravelled = 0;
    lastFrameTime = performance.now();
    animRunning = true;
  }

  // find segment index and local t for given distance (binary search)
  function findSegmentForDistance(d) {
    if (!nodeCum || nodeCum.length === 0) return { segIndex: 0, t: 0 };
    if (d <= 0) return { segIndex: 0, t: 0 };
    if (d >= totalLength) {
      return { segIndex: Math.max(0, nodeCum.length - 2), t: 1 };
    }
    let left = 0, right = nodeCum.length - 1;
    while (left <= right) {
      const mid = (left + right) >> 1;
      if (nodeCum[mid] <= d) left = mid + 1;
      else right = mid - 1;
    }
    const segIndex = Math.min(Math.max(0, right), nodeCum.length - 2);
    const segStart = nodeCum[segIndex];
    const segLen = segLengths[segIndex] || 0;
    const rem = d - segStart;
    const t = segLen > 0 ? Math.max(0, Math.min(1, rem / segLen)) : 0;
    return { segIndex, t };
  }

  // interpolate between two points
  function lerp(a, b, t) {
    return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
  }

  // draw circular marker (filled + stroke)
  function drawMarker(pos, opts = {}) {
    const r = typeof opts.r === 'number' ? opts.r : 6;
    const fill = opts.fill || '#ff0000';
    const stroke = opts.stroke || '#880000';
    const lineW = typeof opts.lineWidth === 'number' ? opts.lineWidth : 2;
    ctx.beginPath();
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineW;
    ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // draw small node circle (background)
  function drawNode(pos, opts = {}) {
    const r = typeof opts.r === 'number' ? opts.r : Math.max(2, 3);
    ctx.beginPath();
    ctx.fillStyle = opts.fill || '#fff';
    ctx.strokeStyle = opts.stroke || '#000';
    ctx.lineWidth = opts.lineWidth || 0.8;
    ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // --- rendering loop ---
  function render(now) {
    if (typeof now !== 'number') now = performance.now();

    resizeCanvasToDisplaySize();

    // Clear
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Draw saved strokes
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#0ea5ff';
    ctx.lineWidth = 2;

    for (const s of strokes) {
      const toDraw = smoothStroke(s);
      if (!toDraw || toDraw.length === 0) continue;
      ctx.beginPath();
      ctx.moveTo(toDraw[0].x, toDraw[0].y);
      for (let i = 1; i < toDraw.length; i++) ctx.lineTo(toDraw[i].x, toDraw[i].y);
      ctx.stroke();
    }

    // Draw current stroke preview
    if (currentStroke && currentStroke.length > 0) {
      const preview = smoothStroke(currentStroke);
      ctx.strokeStyle = '#7c3aed';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(preview[0].x, preview[0].y);
      for (let i = 1; i < preview.length; i++) ctx.lineTo(preview[i].x, preview[i].y);
      ctx.stroke();
    }


    // Draw server-returned graph overlay and animated walk
    const resp = window.lastServerResponse;
    if (resp && resp.positions && Array.isArray(resp.positions) && resp.positions.length > 0) {
      // faint graph edges from resp.edges if present
      if (resp.edges && Array.isArray(resp.edges)) {
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(33,150,243,0.25)';
        for (const e of resp.edges) {
          const u = String(e[0]), v = String(e[1]);
          const nmap = resp.nodes || {};
          if (nmap[u] && nmap[v]) {
            const a = { x: nmap[u][0], y: nmap[u][1] };
            const b = { x: nmap[v][0], y: nmap[v][1] };
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
      }

      // draw nodes (either resp.nodes or fallback to resp.positions)
      if (resp.nodes) {
        for (const [id, p] of Object.entries(resp.nodes)) {
          drawNode({ x: p[0], y: p[1] }, { r: Math.max(2, (canvas.width / 800)), fill: '#fff', stroke: '#000', lineWidth: 0.6 });
        }
      } else {
        for (const p of resp.positions) drawNode({ x: p.x, y: p.y }, { r: 2, fill: '#fff', stroke: '#000' });
      }

      // highlight all nodes if requested
      const highlightAll = document.getElementById('highlightAllNodesChk') && document.getElementById('highlightAllNodesChk').checked;
      if (highlightAll) {
        for (const p of resp.positions) {
          drawMarker({ x: p.x, y: p.y }, { r: Math.max(3, 0.35 * Math.min(canvas.width, canvas.height) / 100), fill: '#00e5c1', stroke: '#008f6b', lineWidth: 1 });
        }
      }

      // start marker
      const start = resp.positions[0];
      if (start) drawMarker({ x: start.x, y: start.y }, { r: Math.max(6, 0.6 * Math.min(canvas.width, canvas.height) / 100), fill: '#ff3b30', stroke: '#7f1d1d', lineWidth: 2 });

      // Advance animation by time elapsed (arc-length model)
      if (animRunning) {
        if (!lastFrameTime) lastFrameTime = now;
        const dt = Math.max(0, (now - lastFrameTime) / 1000); // seconds
        lastFrameTime = now;
        // Update distance travelled by speed * dt
        distanceTravelled += animSpeed * dt;
        if (distanceTravelled >= totalLength) {
          distanceTravelled = totalLength;
          animRunning = false; // finished
        }
      } else {
        lastFrameTime = null;
      }

      // draw rainbow trail up to current travelled distance
      const pts = resp.positions;
      const nPts = pts.length;
      if (nPts > 1) {
        ctx.lineWidth = 3;
        for (let i = 0; i < nPts - 1; i++) {
          const segStart = nodeCum[i];
          const segEnd = nodeCum[i + 1];
          const segLen = segLengths[i] || 0;
          const segCount = Math.max(1, nPts - 1);
          const hueT = i / (segCount - 1 || 1);
          const hue = hueT * 360;
          ctx.strokeStyle = `hsl(${hue},80%,50%)`;

          if (distanceTravelled >= segEnd) {
            ctx.beginPath();
            ctx.moveTo(pts[i].x, pts[i].y);
            ctx.lineTo(pts[i + 1].x, pts[i + 1].y);
            ctx.stroke();
            continue;
          }

          if (distanceTravelled > segStart && distanceTravelled < segEnd) {
            const local = distanceTravelled - segStart;
            const t = segLen > 0 ? Math.max(0, Math.min(1, local / segLen)) : 0;
            const ip = lerp(pts[i], pts[i + 1], t);
            ctx.beginPath();
            ctx.moveTo(pts[i].x, pts[i].y);
            ctx.lineTo(ip.x, ip.y);
            ctx.stroke();
            break;
          }

          if (distanceTravelled <= segStart) {
            break;
          }
        }
      }

      // draw moving head (current animated marker) at the interpolated position
      let headPos = null;
      if (distanceTravelled <= 0) {
        headPos = pts[0];
      } else if (distanceTravelled >= totalLength) {
        headPos = pts[pts.length - 1];
      } else {
        const { segIndex, t } = findSegmentForDistance(distanceTravelled);
        const A = pts[segIndex];
        const B = pts[segIndex + 1];
        headPos = lerp(A, B, t);
      }

      if (headPos) {
        drawMarker(headPos, { r: Math.max(4, 0.4 * Math.min(canvas.width, canvas.height) / 100), fill: '#ff3b30', stroke: '#7f1d1d', lineWidth: 1 });
      }

      // If not animating and completed, draw full rainbow path & final marker
      if (!animRunning && distanceTravelled >= totalLength) {
        if (distanceTravelled >= totalLength) {
          const segCount = Math.max(1, pts.length - 1);
          ctx.lineWidth = 3;
          for (let i = 0; i < pts.length - 1; i++) {
            const hueT = i / (segCount - 1 || 1);
            const hue = hueT * 360;
            ctx.strokeStyle = `hsl(${hue},80%,50%)`;
            ctx.beginPath();
            ctx.moveTo(pts[i].x, pts[i].y);
            ctx.lineTo(pts[i + 1].x, pts[i + 1].y);
            ctx.stroke();
          }
          const last = pts[pts.length - 1];
          if (last) drawMarker(last, { r: Math.max(4, 0.4 * Math.min(canvas.width, canvas.height) / 100), fill: '#ff3b30', stroke: '#7f1d1d', lineWidth: 1 });
        }
      }
    }

    // draw client-side merged preview overlay if any
    const preview = window.mergedGraphForPreview;
    if (preview && preview.nodes && preview.edges) {
      // edges
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = 'rgba(0,200,180,0.9)';
      for (const e of preview.edges) {
        const a = preview.nodes[e[0]];
        const b = preview.nodes[e[1]];
        if (!a || !b) continue;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      // nodes
      for (let i = 0; i < preview.nodes.length; i++) {
        const p = preview.nodes[i];
        drawNode(p, { r: 2.4, fill: '#00ffc9', stroke: '#004d3a', lineWidth: 0.8 });
      }
    }

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);

  // --- UI: clear / send / compute ---
  clearBtn.addEventListener('click', () => {
    strokes = [];
    currentStroke = null;
    window.lastServerResponse = null;
    window.mergedGraphForPreview = null;
    info.innerText = 'Cleared.';
    // stop animation and clear overlay
    stopWalkAnimation();
  });

  // Build payload for server: send multiple polylines (or a single merged polyline if merge checkbox set)
  function buildPayload() {
    const tol = parseFloat(tolInput.value) || 0;
    let payloadStrokes = [];

    // send strokes individually (multiple polylines)
    payloadStrokes = strokes.map(s => s.map(p => ({ x: p.x, y: p.y })));
    const keepLargest = document.getElementById('keepLargestChk') && document.getElementById('keepLargestChk').checked;

    return {
      strokes: payloadStrokes,
      tol: tol,
      keep_largest: keepLargest
    };
  }

  async function postPolylines() {
    const endpoint = 'http://localhost:8000/upload_polylines';
    const payload = buildPayload();
    if (!payload.strokes || payload.strokes.length === 0 || payload.strokes.every(s => s.length === 0)) {
      info.innerText = 'No strokes to send.';
      return null;
    }

    // stop any current animation before sending
    stopWalkAnimation();

    info.innerText = 'Sending polylines...';
    try {
      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || resp.statusText);
      }
      const data = await resp.json();
      window.lastServerResponse = data;
      info.innerText = `Server OK — nodes=${data.n_nodes ?? '-'} edges=${data.n_edges ?? '-'} walk_len=${(data.walk && data.walk.length) || '-'} tol=${payload.tol} split=${payload.split_intersections}`;

      // start animation using returned positions (if present)
      if (data.positions && Array.isArray(data.positions) && data.positions.length > 1) {
        startWalkAnimation(data.positions);
      } else {
        stopWalkAnimation();
      }

      return data;
    } catch (err) {
      console.error(err);
      info.innerText = 'Send failed: ' + (err.message || err);
      return null;
    }
  }

  sendBtn.addEventListener('click', postPolylines);
  computeBtn.addEventListener('click', postPolylines);

  // Expose helpers for debugging
  window.vcanvas = {
    getStrokes: () => strokes,
    clear: () => { strokes = []; currentStroke = null; },
    lastServerResponse: () => window.lastServerResponse,
    startWalkAnimation: (positions) => startWalkAnimation(positions),
    stopWalkAnimation: () => stopWalkAnimation(),
    setAnimSpeed: (pxPerSec) => { animSpeed = Number(pxPerSec) || animSpeed; },
    computeClientSplitAndPreview: () => computeClientSplitAndPreview()
  };

  info.innerText = 'Ready — draw with mouse or touch. Use Compute/Send when ready.';
})();
