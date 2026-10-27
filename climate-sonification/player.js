(() => {
  "use strict";

  const SONG = window.SONG;
  if (!SONG) {
    document.body.innerHTML = "<p style='padding:40px;color:#fff'>player_data.js missing — run <code>python3 sonify.py</code> first.</p>";
    return;
  }

  const START = SONG.meta.start;
  const END = SONG.meta.end;
  const END_BOUND = START + SONG.bars.length; // 2025.0
  const DURATION = SONG.meta.duration;
  const $ = (s) => document.querySelector(s);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const chartEl = $("#chart");
  const rollEl = $("#roll");
  const playBtn = $("#play");
  const yearSlider = $("#year");
  const timeEl = $("#time");
  const yearNow = $("#yearNow");
  const tempNow = $("#tempNow");
  const co2Now = $("#co2Now");
  const tempoNow = $("#tempoNow");
  const hemiNow = $("#hemiNow");

  const state = {
    year: START,
    time: 0,
    scrubbing: false,
    audioReady: false,
    muted: new Set(),
    lastFrame: 0,
  };

  // ---------- time <-> year ----------
  function barIndexAt(t) {
    let lo = 0, hi = SONG.bars.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (SONG.bars[mid].t <= t) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  function timeToYear(t) {
    t = clamp(t, 0, DURATION);
    const i = barIndexAt(t);
    const b = SONG.bars[i];
    return clamp(b.year + (t - b.t) / b.d, START, END_BOUND);
  }

  function yearToTime(y) {
    y = clamp(y, START, END_BOUND);
    if (y >= END_BOUND) return DURATION;
    const i = clamp(Math.floor(y - START), 0, SONG.bars.length - 1);
    const b = SONG.bars[i];
    return b.t + (y - b.year) * b.d;
  }

  function formatTime(s) {
    s = Math.max(0, s);
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return `${m}:${String(r).padStart(2, "0")}`;
  }

  // ---------- precompute note positions ----------
  for (const n of SONG.notes) {
    const i = barIndexAt(n.t);
    const b = SONG.bars[i];
    n.ny = b.year + (n.t - b.t) / b.d;
    n.nw = n.d / b.d;
    n.bar = i;
  }

  function tempAt(year) {
    const a = SONG.series.temp;
    const i = clamp(Math.floor(year - START), 0, a.length - 1);
    const j = Math.min(i + 1, a.length - 1);
    const f = clamp(year - START - i, 0, 1);
    return a[i] * (1 - f) + a[j] * f;
  }

  function co2At(year) {
    const ys = SONG.series.years;
    const vs = SONG.series.co2;
    const i = clamp(Math.floor(year - START), 0, ys.length - 1);
    if (vs[i] == null) return null;
    const j = Math.min(i + 1, vs.length - 1);
    if (vs[j] == null) return vs[i];
    const f = clamp(year - START - i, 0, 1);
    return vs[i] * (1 - f) + vs[j] * f;
  }

  function tempoAt(year) {
    const bar = Math.floor(year - START);
    let bpm = SONG.tempo[0][1];
    for (const [b, v] of SONG.tempo) if (b <= bar) bpm = v;
    return bpm;
  }

  function contrastAt(year) {
    const vs = SONG.series.contrast;
    if (!vs) return null;
    const i = clamp(Math.floor(year - START), 0, vs.length - 1);
    if (vs[i] == null) return null;
    const j = Math.min(i + 1, vs.length - 1);
    if (vs[j] == null) return vs[i];
    const f = clamp(year - START - i, 0, 1);
    return vs[i] * (1 - f) + vs[j] * f;
  }

  // ---------- canvas helpers ----------
  const COLOR = {
    temp: "#ff8a5b",
    tempDeep: "#ff6b4a",
    co2: "#2dd4bf",
    bass: "#4f9cf9",
    motif: "#ffd166",
    hemi: "#b388ff",
    grid: "rgba(145,170,210,.12)",
    faint: "#5c6f8f",
    muted: "#8fa3c4",
    ink: "#eef4ff",
  };

  function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h, dpr };
  }

  function makeBuffer(w, h, dpr) {
    const c = document.createElement("canvas");
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { canvas: c, ctx };
  }

  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, h / 2, w / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // ---------- static chart ----------
  let chartBuf = null;
  let chartGeom = null;
  let rollBuf = null;
  let rollGeom = null;

  function buildChart() {
    const { ctx: live, w, h, dpr } = setupCanvas(chartEl);
    const { canvas, ctx } = makeBuffer(w, h, dpr);
    const L = 58, R = 66, T = 24, B = 30;
    const pw = w - L - R, ph = h - T - B;
    const x = (year) => L + ((year - START) / (END_BOUND - START)) * pw;
    const ty = (v) => T + ((1.5 - v) / 2.1) * ph;
    const cy = (v) => T + ((428 - v) / 116) * ph;
    // NH-SH contrast lives in its own band along the bottom (-0.4..0.8 -> bottom..top of band)
    const hy = (v) => T + (0.97 - 0.25 * ((v + 0.4) / 1.2)) * ph;

    chartGeom = { w, h, L, R, T, B, pw, ph, x, ty, cy, hy };

    // background bands per decade
    for (let i = 0; i < SONG.sections.length; i++) {
      const s = SONG.sections[i];
      const x0 = x(s.year), x1 = x(Math.min(s.year + 10, END_BOUND));
      if (i % 2 === 0) {
        ctx.fillStyle = "rgba(255,255,255,.018)";
        ctx.fillRect(x0, T, x1 - x0, ph);
      }
      ctx.strokeStyle = "rgba(145,170,210,.075)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(x0) + 0.5, T);
      ctx.lineTo(Math.round(x0) + 0.5, h - B);
      ctx.stroke();
    }

    // horizontal grid + left labels
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of [-0.5, 0, 0.5, 1.0, 1.5]) {
      const yy = ty(v);
      ctx.strokeStyle = v === 0 ? "rgba(255,255,255,.22)" : COLOR.grid;
      ctx.setLineDash(v === 0 ? [4, 5] : []);
      ctx.beginPath();
      ctx.moveTo(L, yy);
      ctx.lineTo(w - R, yy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = v === 0 ? COLOR.muted : COLOR.faint;
      ctx.fillText(`${v > 0 ? "+" : ""}${v.toFixed(1)}°`, L - 10, yy);
    }

    // right CO2 labels
    ctx.textAlign = "left";
    for (const v of [320, 360, 400]) {
      const yy = cy(v);
      ctx.fillStyle = "rgba(45,212,191,.55)";
      ctx.fillText(`${v}`, w - R + 10, yy);
    }
    ctx.fillStyle = "rgba(45,212,191,.75)";
    ctx.fillText("ppm", w - R + 10, T - 8);
    ctx.fillStyle = "rgba(255,138,91,.8)";
    ctx.textAlign = "left";
    ctx.fillText("°C", 14, T - 8);

    // x labels
    ctx.textAlign = "center";
    ctx.fillStyle = COLOR.faint;
    for (const y of [1925, 1950, 1975, 2000, 2024]) {
      ctx.fillText(`${y}`, x(y), h - 10);
    }

    // temperature fill
    const tGrad = ctx.createLinearGradient(0, T, 0, T + ph);
    tGrad.addColorStop(0, "rgba(255,107,74,.28)");
    tGrad.addColorStop(1, "rgba(255,107,74,0)");
    ctx.beginPath();
    SONG.series.years.forEach((yr, i) => {
      const xx = x(yr + 0.5), yy = ty(SONG.series.temp[i]);
      if (i === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    });
    ctx.lineTo(x(2024.5), ty(0));
    ctx.lineTo(x(1925), ty(0));
    ctx.closePath();
    ctx.fillStyle = tGrad;
    ctx.fill();

    // temperature line
    ctx.beginPath();
    SONG.series.years.forEach((yr, i) => {
      const xx = x(yr + 0.5), yy = ty(SONG.series.temp[i]);
      if (i === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    });
    ctx.strokeStyle = COLOR.temp;
    ctx.lineWidth = 2.4;
    ctx.lineJoin = "round";
    ctx.shadowColor = "rgba(255,107,74,.45)";
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // CO2 line
    let started = false;
    ctx.beginPath();
    SONG.series.years.forEach((yr, i) => {
      const v = SONG.series.co2[i];
      if (v == null) return;
      const xx = x(yr + 0.5), yy = cy(v);
      if (!started) { ctx.moveTo(xx, yy); started = true; }
      else ctx.lineTo(xx, yy);
    });
    ctx.strokeStyle = COLOR.co2;
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(45,212,191,.35)";
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // CO2 entry annotation
    const entry = SONG.meta.co2Entry;
    if (entry) {
      const xx = x(entry);
      ctx.strokeStyle = "rgba(45,212,191,.35)";
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.moveTo(xx, T);
      ctx.lineTo(xx, h - B);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(45,212,191,.8)";
      ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillText("CO₂ enters", xx + 7, T + 12);
    }

    // hemispheric contrast (NH - SH) in its own band along the bottom
    const hs = SONG.series.contrast;
    if (hs) {
      const zeroY = hy(0);
      ctx.strokeStyle = "rgba(179,136,255,.28)";
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(L, zeroY);
      ctx.lineTo(w - R, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);
      let started = false;
      ctx.beginPath();
      SONG.series.years.forEach((yr, i) => {
        const v = hs[i];
        if (v == null) return;
        const xx = x(yr + 0.5), yy = hy(v);
        if (!started) { ctx.moveTo(xx, yy); started = true; }
        else ctx.lineTo(xx, yy);
      });
      ctx.strokeStyle = "rgba(179,136,255,.85)";
      ctx.lineWidth = 1.6;
      ctx.shadowColor = "rgba(179,136,255,.35)";
      ctx.shadowBlur = 8;
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(179,136,255,.75)";
      ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillText("NH−SH", L + 4, T + 0.70 * ph - 6);
    }

    // title note (hidden on narrow screens where the readout moves in)
    if (w > 560) {
      ctx.fillStyle = "rgba(238,244,255,.6)";
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillText("GLOBAL TEMPERATURE ANOMALY", L, T + 12);
    }

    chartBuf = canvas;
    drawFrame();
  }

  // ---------- static piano roll ----------
  function buildRoll() {
    const { w, h, dpr } = setupCanvas(rollEl);
    const { canvas, ctx } = makeBuffer(w, h, dpr);
    const L = 58, R = 66, T = 16, B = 22;
    const pw = w - L - R, ph = h - T - B;
    const x = (year) => L + ((year - START) / (END_BOUND - START)) * pw;
    const y = (midi) => T + ((105 - midi) / (105 - 30)) * ph;

    rollGeom = { w, h, L, R, T, B, pw, ph, x, y };

    // bands
    for (let i = 0; i < SONG.sections.length; i++) {
      const s = SONG.sections[i];
      const x0 = x(s.year), x1 = x(Math.min(s.year + 10, END_BOUND));
      if (i % 2 === 0) {
        ctx.fillStyle = "rgba(255,255,255,.018)";
        ctx.fillRect(x0, T, x1 - x0, ph);
      }
      ctx.strokeStyle = "rgba(145,170,210,.07)";
      ctx.beginPath();
      ctx.moveTo(Math.round(x0) + 0.5, T);
      ctx.lineTo(Math.round(x0) + 0.5, h - B);
      ctx.stroke();
    }

    // note-name grid
    ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const [midi, name] of [[36, "C2"], [48, "C3"], [60, "C4"], [72, "C5"], [84, "C6"], [96, "C7"]]) {
      const yy = y(midi);
      ctx.strokeStyle = "rgba(145,170,210,.09)";
      ctx.beginPath();
      ctx.moveTo(L, yy);
      ctx.lineTo(w - R, yy);
      ctx.stroke();
      ctx.fillStyle = COLOR.faint;
      ctx.fillText(name, L - 9, yy);
    }

    // notes
    const colors = { 0: "#ff8a5b", 1: "#4f9cf9", 2: "#2dd4bf", 3: "#ffd166", 4: "#b388ff" };
    const alphas = { 0: 0.9, 1: 0.55, 2: 0.95, 3: 0.95, 4: 0.85 };
    for (const n of SONG.notes) {
      const x0 = x(n.ny), x1 = x(n.ny + Math.max(n.nw, 0.012));
      const yy = y(n.m);
      ctx.globalAlpha = alphas[n.vc];
      ctx.fillStyle = colors[n.vc];
      ctx.fillRect(x0, yy - 1.6, Math.max(1.4, x1 - x0), 3.2);
    }
    ctx.globalAlpha = 1;

    // bass root / CO2 register hints
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = "rgba(45,212,191,.22)";
    for (const midi of [50, 62]) {
      const yy = y(midi);
      ctx.beginPath();
      ctx.moveTo(L, yy);
      ctx.lineTo(w - R, yy);
      ctx.stroke();
    }
    // hemispheric contrast register hints
    ctx.strokeStyle = "rgba(179,136,255,.20)";
    for (const midi of [81, 105]) {
      const yy = y(midi);
      ctx.beginPath();
      ctx.moveTo(L, yy);
      ctx.lineTo(w - R, yy);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    rollBuf = canvas;
    drawFrame();
  }

  // ---------- frame ----------
  function drawFrame() {
    if (!chartBuf || !rollBuf) return;
    const year = state.year;
    const chart = chartEl.getContext("2d");
    const roll = rollEl.getContext("2d");
    const cdpr = Math.min(2, window.devicePixelRatio || 1);
    const rdpr = cdpr;

    chart.setTransform(1, 0, 0, 1, 0, 0);
    chart.clearRect(0, 0, chartEl.width, chartEl.height);
    chart.drawImage(chartBuf, 0, 0);
    chart.setTransform(cdpr, 0, 0, cdpr, 0, 0);

    roll.setTransform(1, 0, 0, 1, 0, 0);
    roll.clearRect(0, 0, rollEl.width, rollEl.height);
    roll.drawImage(rollBuf, 0, 0);
    roll.setTransform(rdpr, 0, 0, rdpr, 0, 0);

    if (!chartGeom || !rollGeom) return;

    // playhead
    const cx = chartGeom.x(year);
    chart.save();
    chart.strokeStyle = "rgba(255,255,255,.7)";
    chart.lineWidth = 1;
    chart.shadowColor = "rgba(255,255,255,.35)";
    chart.shadowBlur = 8;
    chart.beginPath();
    chart.moveTo(cx, chartGeom.T);
    chart.lineTo(cx, chartGeom.h - chartGeom.B);
    chart.stroke();
    chart.restore();

    // dots
    const tv = tempAt(year);
    const ty = chartGeom.ty(tv);
    chart.beginPath();
    chart.arc(cx, ty, 4.5, 0, Math.PI * 2);
    chart.fillStyle = COLOR.temp;
    chart.shadowColor = COLOR.temp;
    chart.shadowBlur = 14;
    chart.fill();
    chart.shadowBlur = 0;

    const cv = co2At(year);
    if (cv != null) {
      const cy = chartGeom.cy(cv);
      chart.beginPath();
      chart.arc(cx, cy, 3.8, 0, Math.PI * 2);
      chart.fillStyle = COLOR.co2;
      chart.shadowColor = COLOR.co2;
      chart.shadowBlur = 12;
      chart.fill();
      chart.shadowBlur = 0;
    }

    const hv = contrastAt(year);
    if (hv != null) {
      const hyy = chartGeom.hy(hv);
      chart.beginPath();
      chart.arc(cx, hyy, 3, 0, Math.PI * 2);
      chart.fillStyle = COLOR.hemi;
      chart.shadowColor = COLOR.hemi;
      chart.shadowBlur = 10;
      chart.fill();
      chart.shadowBlur = 0;
    }

    // year bubble
    const label = String(Math.min(END, Math.floor(year)));
    chart.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
    const tw = chart.measureText(label).width + 14;
    const bx = clamp(cx - tw / 2, chartGeom.L, chartGeom.w - chartGeom.R - tw);
    chart.fillStyle = "rgba(238,244,255,.94)";
    roundRect(chart, bx, chartGeom.T + 2, tw, 20, 6);
    chart.fill();
    chart.fillStyle = "#0a1120";
    chart.textAlign = "center";
    chart.textBaseline = "middle";
    chart.fillText(label, bx + tw / 2, chartGeom.T + 12.5);

    // roll playhead
    const rx = rollGeom.x(year);
    roll.strokeStyle = "rgba(255,255,255,.55)";
    roll.lineWidth = 1;
    roll.beginPath();
    roll.moveTo(rx, rollGeom.T);
    roll.lineTo(rx, rollGeom.h - rollGeom.B);
    roll.stroke();

    // active notes glow
    for (const n of SONG.notes) {
      if (n.t <= state.time && state.time < n.t + n.d) {
        const x0 = rollGeom.x(n.ny);
        const x1 = rollGeom.x(n.ny + Math.max(n.nw, 0.012));
        const yy = rollGeom.y(n.m);
        roll.fillStyle = "rgba(255,255,255,.92)";
        roll.fillRect(x0, yy - 2.3, Math.max(2, x1 - x0), 4.6);
      }
    }
  }

  // ---------- readout ----------
  function updateReadout() {
    const year = state.year;
    yearNow.textContent = String(Math.min(END, Math.floor(year)));
    const t = tempAt(year);
    tempNow.textContent = `${t >= 0 ? "+" : ""}${t.toFixed(2)}°C`;
    const c = co2At(year);
    co2Now.textContent = c == null ? "—" : `${c.toFixed(0)} ppm`;
    tempoNow.textContent = `${tempoAt(year)} BPM`;
    const h = contrastAt(year);
    hemiNow.textContent = h == null ? "—" : `${h >= 0 ? "+" : ""}${h.toFixed(2)}°`;

    if (!state.scrubbing) {
      yearSlider.value = String(year);
    }
    const p = ((year - START) / (END_BOUND - START)) * 100;
    yearSlider.style.setProperty("--p", `${clamp(p, 0, 100)}%`);
    timeEl.textContent = `${formatTime(state.time)} / ${formatTime(DURATION)}`;
  }

  // ---------- audio ----------
  const audio = { synths: {}, parts: [], bases: { 0: -7, 1: -10, 2: -11, 3: -14, 4: -16 } };

  function buildAudio() {
    const master = new Tone.Volume(-3).toDestination();
    const meter = new Tone.Meter({ normalRange: false });
    master.connect(meter);
    audio.meter = meter;
    const limiter = new Tone.Limiter(-1).connect(master);
    const reverb = new Tone.Reverb({ decay: 4.4, preDelay: 0.02, wet: 0.34 }).connect(limiter);

    const melody = new Tone.PolySynth(Tone.FMSynth, {
      harmonicity: 1.6,
      modulationIndex: 2.2,
      oscillator: { type: "triangle" },
      envelope: { attack: 0.025, decay: 0.18, sustain: 0.45, release: 0.7 },
      modulation: { type: "sine" },
      modulationEnvelope: { attack: 0.01, decay: 0.2, sustain: 0.2, release: 0.5 },
    }).connect(reverb);
    melody.volume.value = audio.bases[0];

    const bass = new Tone.MonoSynth({
      oscillator: { type: "sawtooth" },
      filter: { Q: 1.8, type: "lowpass", rolloff: -24 },
      envelope: { attack: 0.012, decay: 0.28, sustain: 0.45, release: 0.5 },
      filterEnvelope: { attack: 0.01, decay: 0.22, sustain: 0.25, release: 0.6, baseFrequency: 110, octaves: 2.6 },
    }).connect(reverb);
    bass.volume.value = audio.bases[1];

    const co2 = new Tone.AMSynth({
      harmonicity: 1.5,
      oscillator: { type: "sine" },
      envelope: { attack: 0.55, decay: 0.35, sustain: 0.85, release: 2.6 },
      modulation: { type: "triangle" },
      modulationEnvelope: { attack: 0.9, decay: 0.4, sustain: 0.6, release: 2.2 },
    }).connect(reverb);
    co2.volume.value = audio.bases[2];

    const motif = new Tone.PolySynth(Tone.FMSynth, {
      harmonicity: 3.01,
      modulationIndex: 9,
      oscillator: { type: "sine" },
      envelope: { attack: 0.002, decay: 1.1, sustain: 0, release: 1.2 },
      modulation: { type: "sine" },
      modulationEnvelope: { attack: 0.002, decay: 0.6, sustain: 0, release: 0.8 },
    }).connect(reverb);
    motif.volume.value = audio.bases[3];

    const hemi = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "sine" },
      envelope: { attack: 0.45, decay: 0.5, sustain: 0.85, release: 2.4 },
    }).connect(reverb);
    hemi.volume.value = audio.bases[4];

    audio.synths = { 0: melody, 1: bass, 2: co2, 3: motif, 4: hemi };

    // staccato melody notes (art === -1) get a crisp, plucky attack:
    // articulation maps climate velocity, distinct from rhythm and tempo
    const staccato = new Tone.PolySynth(Tone.FMSynth, {
      harmonicity: 1.6,
      modulationIndex: 2.2,
      oscillator: { type: "triangle" },
      envelope: { attack: 0.004, decay: 0.09, sustain: 0.08, release: 0.28 },
      modulation: { type: "sine" },
      modulationEnvelope: { attack: 0.005, decay: 0.08, sustain: 0.05, release: 0.25 },
    }).connect(reverb);
    staccato.volume.value = audio.bases[0] + 1.5;
    audio.staccato = staccato;

    for (const [vc, synth] of Object.entries(audio.synths)) {
      const events = SONG.notes
        .filter((n) => n.vc === Number(vc) && !(Number(vc) === 0 && n.art === -1))
        .map((n) => ({ time: n.t, midi: n.m, dur: n.d, vel: n.v }));
      const part = new Tone.Part((time, e) => {
        synth.triggerAttackRelease(
          Tone.Frequency(e.midi, "midi").toFrequency(),
          Math.max(0.06, e.dur * 0.96),
          time,
          e.vel / 127
        );
      }, events).start(0);
      audio.parts.push(part);
    }

    const staccatoEvents = SONG.notes
      .filter((n) => n.vc === 0 && n.art === -1)
      .map((n) => ({ time: n.t, midi: n.m, dur: n.d, vel: n.v }));
    const staccatoPart = new Tone.Part((time, e) => {
      audio.staccato.triggerAttackRelease(
        Tone.Frequency(e.midi, "midi").toFrequency(),
        Math.max(0.05, e.dur * 1.1),
        time,
        e.vel / 127
      );
    }, staccatoEvents).start(0);
    audio.parts.push(staccatoPart);

    // apply mutes chosen before audio existed
    for (const vc of state.muted) {
      if (audio.synths[vc]) audio.synths[vc].volume.value = -60;
      if (vc === 0 && audio.staccato) audio.staccato.volume.value = -60;
    }
    state.audioReady = true;
  }

  async function togglePlay() {
    if (!state.audioReady) {
      await Tone.start();
      buildAudio();
      Tone.Transport.seconds = state.time;
    }
    if (Tone.Transport.state === "started") {
      Tone.Transport.pause();
      playBtn.classList.remove("playing");
    } else {
      if (state.time >= DURATION - 0.05) {
        seek(START);
      }
      Tone.Transport.start();
      playBtn.classList.add("playing");
    }
  }

  function seek(year) {
    state.year = clamp(year, START, END_BOUND);
    state.time = yearToTime(state.year);
    if (state.audioReady) {
      Tone.Transport.seconds = state.time;
    }
    updateReadout();
    drawFrame();
  }

  // ---------- events ----------
  playBtn.addEventListener("click", () => {
    togglePlay().catch((err) => console.error(err));
  });

  yearSlider.addEventListener("pointerdown", () => { state.scrubbing = true; });
  yearSlider.addEventListener("pointerup", () => { state.scrubbing = false; });
  yearSlider.addEventListener("input", () => {
    seek(parseFloat(yearSlider.value));
  });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !["INPUT", "BUTTON"].includes(document.activeElement.tagName)) {
      e.preventDefault();
      togglePlay().catch(console.error);
    }
  });

  chartEl.addEventListener("click", (e) => {
    if (!chartGeom) return;
    const rect = chartEl.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const year = START + ((px - chartGeom.L) / chartGeom.pw) * (END_BOUND - START);
    seek(year);
  });

  for (const chip of document.querySelectorAll(".chip")) {
    chip.addEventListener("click", () => {
      const vc = Number(chip.dataset.voice);
      if (state.muted.has(vc)) state.muted.delete(vc);
      else state.muted.add(vc);
      chip.classList.toggle("off", state.muted.has(vc));
      if (audio.synths[vc]) {
        audio.synths[vc].volume.value = state.muted.has(vc) ? -60 : audio.bases[vc];
      }
      if (vc === 0 && audio.staccato) {
        audio.staccato.volume.value = state.muted.has(vc) ? -60 : audio.bases[0] + 1.5;
      }
    });
  }

  // ---------- loop ----------
  function loop() {
    if (Tone.Transport.state === "started") {
      const t = Tone.Transport.seconds;
      if (t >= DURATION - 0.02) {
        Tone.Transport.pause();
        state.time = DURATION;
        state.year = END_BOUND;
        playBtn.classList.remove("playing");
        updateReadout();
        drawFrame();
      } else {
        state.time = t;
        state.year = timeToYear(t);
        updateReadout();
        drawFrame();
      }
    }
    requestAnimationFrame(loop);
  }

  // ---------- boot ----------
  let resizeTimer = 0;
  function rebuild() {
    buildChart();
    buildRoll();
    updateReadout();
    drawFrame();
  }
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(rebuild, 120);
  });

  // expose a tiny debug handle (useful for testing / tinkering)
  window.__climate = { state, audio, SONG, seek, togglePlay };

  // wait for fonts/layout, then draw
  requestAnimationFrame(() => {
    rebuild();
    requestAnimationFrame(loop);
  });
})();
