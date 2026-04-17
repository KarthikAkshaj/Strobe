'use strict';

// ─── Scale ceilings for bar normalization ─────────────────────────────────────
// CLIP ViT-B/32 cosine similarity to a 55-caption bank rarely exceeds 0.40.
// Bars are normalized to these maxima so typical values fill ~60-80% of the bar.
const CONF_MAX = 0.40;
const GAP_MAX  = 0.05;
const SIM_MAX  = 1.00;

// ─── State ───────────────────────────────────────────────────────────────────
let captions      = [];   // sorted by timestamp ascending
let metadata      = {};
let searchResults = [];   // from search_video.py output (optional)
let searchQuery   = '';
let videoDuration = 0;
let currentIdx    = -2;   // -2 = not yet computed
let tCanvasW      = 0;
let isDragging    = false;
let tlListenersAdded = false;

// ─── Element refs ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const loadScreen  = $('load-screen');
const viewer      = $('viewer');
const videoEl     = $('video-el');
const overlayEl   = $('caption-overlay');
const overlayTxt  = $('overlay-text');
const overlayConf = $('overlay-conf');
const overlayTs   = $('overlay-ts');
const capText     = $('cap-text');
const confBar     = $('conf-bar');
const confVal     = $('conf-val');
const gapBar      = $('gap-bar');
const gapVal      = $('gap-val');
const simBar      = $('sim-bar');
const simVal      = $('sim-val');
const altsList    = $('alts-list');
const altsLabel   = $('alts-label');
const metaGrid    = $('meta-grid');
const eventsList  = $('events-list');
const evCount     = $('ev-count');
const canvas      = $('timeline-canvas');
const playhead    = $('playhead');
const launchBtn   = $('launch-btn');

// ─── File loading ─────────────────────────────────────────────────────────────
let videoURL   = null;
let videoReady = false;
let jsonReady  = false;

function setupDropZone(zoneId, inputId, labelId, onFile) {
    const zone  = $(zoneId);
    const input = $(inputId);
    const label = $(labelId);

    // Click-to-pick
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', e => {
        if (e.target.files[0]) onFile(e.target.files[0], label, zone);
    });

    // Drag-and-drop
    zone.addEventListener('dragover', e => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const f = e.dataTransfer.files[0];
        if (f) onFile(f, label, zone);
    });
}

setupDropZone('video-zone', 'video-file', 'video-label', (f, label, zone) => {
    if (videoURL) URL.revokeObjectURL(videoURL);
    videoURL = URL.createObjectURL(f);
    label.textContent = f.name;
    zone.classList.add('ready');
    videoReady = true;
    checkReady();
});

setupDropZone('json-zone', 'json-file', 'json-label', (f, label, zone) => {
    const reader = new FileReader();
    reader.onload = ev => {
        try {
            const data = JSON.parse(ev.target.result);
            captions = (data.captions || []).sort((a, b) => a.timestamp - b.timestamp);
            metadata = data.metadata || {};
            label.textContent = f.name;
            zone.classList.add('ready');
            jsonReady = true;
            checkReady();
        } catch (err) {
            alert('Could not parse JSON: ' + err.message);
        }
    };
    reader.readAsText(f);
});

// Optional: search results
setupDropZone('search-zone', 'search-file', 'search-label', (f, label, zone) => {
    const reader = new FileReader();
    reader.onload = ev => {
        try {
            const data = JSON.parse(ev.target.result);
            searchResults = data.results || [];
            searchQuery = data.query || '';
            label.textContent = f.name;
            zone.classList.add('ready');
        } catch (err) {
            alert('Could not parse search JSON: ' + err.message);
        }
    };
    reader.readAsText(f);
});

function checkReady() {
    launchBtn.disabled = !(videoReady && jsonReady);
}

launchBtn.addEventListener('click', launch);

// ─── Launch ───────────────────────────────────────────────────────────────────
function launch() {
    loadScreen.hidden = true;
    viewer.hidden = false;

    videoEl.src = videoURL;
    videoEl.addEventListener('loadedmetadata', () => {
        videoDuration = videoEl.duration;

        buildMeta();
        buildEventsList();
        buildSearchResults();
        evCount.textContent = `(${captions.length})`;

        // Defer until layout is computed so canvas.offsetWidth is valid
        requestAnimationFrame(() => {
            buildTimeline();
            startLoop();
        });
    }, { once: true });
}

// ─── Sidebar: meta panel ──────────────────────────────────────────────────────
function buildMeta() {
    const m = metadata;
    const modelPath = m.model || '';
    const modelShort = modelPath.includes('/')
        ? modelPath.split('/').slice(-2).join('/')
        : modelPath || '—';

    const rows = [
        ['model',      modelShort],
        ['threshold',  m.change_threshold  != null ? m.change_threshold.toFixed(3)  : '—'],
        ['cap thresh', m.caption_threshold != null ? m.caption_threshold.toFixed(3) : '—'],
        ['hysteresis', m.hysteresis_count  ?? '—'],
        ['top-K',      m.top_k             ?? '—'],
        ['events',     captions.length],
        ['frames',     m.frames_processed  ?? '—'],
        ['time',       m.processing_ms != null ? `${Math.round(m.processing_ms)} ms` : '—'],
    ];
    if (m.blip_model) {
        const blipShort = m.blip_model.split('/').pop();
        rows.push(['captioner', blipShort]);
        rows.push(['enhanced', `${m.blip_enhanced ?? 0} / ${captions.length}`]);
    }

    metaGrid.innerHTML = rows.map(([k, v]) =>
        `<div class="meta-key">${k}</div><div class="meta-val">${v}</div>`
    ).join('');
}

// ─── Sidebar: events list ─────────────────────────────────────────────────────
function buildEventsList() {
    eventsList.innerHTML = captions.map((ev, i) =>
        `<div class="ev-item" data-i="${i}" id="ei${i}">
            <span class="ev-time">${fmtTime(ev.timestamp)}</span>
            <span class="ev-caption">${escapeHtml(ev.caption)}</span>
            <span class="ev-conf">${(ev.confidence * 100).toFixed(1)}%</span>
        </div>`
    ).join('');

    eventsList.addEventListener('click', e => {
        const item = e.target.closest('.ev-item');
        if (item) videoEl.currentTime = captions[+item.dataset.i].timestamp;
    });
}

// ─── Timeline ─────────────────────────────────────────────────────────────────
function buildTimeline() {
    const dpr = window.devicePixelRatio || 1;
    tCanvasW  = canvas.offsetWidth;
    const H   = canvas.offsetHeight;

    canvas.width  = tCanvasW * dpr;
    canvas.height = H * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#141414';
    ctx.fillRect(0, 0, tCanvasW, H);

    if (!videoDuration) return;

    // Subtle tick marks every 10 s
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let t = 10; t < videoDuration; t += 10) {
        const x = (t / videoDuration) * tCanvasW;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }

    // Caption event markers — brightness encodes confidence
    captions.forEach(ev => {
        const x     = (ev.timestamp / videoDuration) * tCanvasW;
        const alpha = 0.35 + 0.65 * Math.min(ev.confidence / CONF_MAX, 1);
        ctx.fillStyle = `rgba(79, 195, 247, ${alpha.toFixed(2)})`;
        ctx.fillRect(x - 2, 5, 4, H - 10);
    });

    // Search result markers (gold) — drawn on top
    if (searchResults.length) {
        searchResults.forEach(r => {
            const x = (r.timestamp / videoDuration) * tCanvasW;
            const alpha = 0.4 + 0.6 * Math.min(r.similarity / CONF_MAX, 1);
            ctx.fillStyle = `rgba(255, 213, 79, ${alpha.toFixed(2)})`;
            ctx.fillRect(x - 1.5, 2, 3, H - 4);
        });
    }

    // Register seek listeners once
    if (!tlListenersAdded) {
        const tl = $('timeline');
        tl.addEventListener('mousedown', e => { isDragging = true; seekFromMouse(e); });
        tl.addEventListener('mousemove', e => { if (isDragging) seekFromMouse(e); });
        window.addEventListener('mouseup', () => { isDragging = false; });
        window.addEventListener('resize', handleResize);
        tlListenersAdded = true;
    }
}

function seekFromMouse(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    videoEl.currentTime = (x / rect.width) * videoDuration;
}

function handleResize() {
    if (!viewer.hidden) buildTimeline();
}

// ─── Main rAF loop ────────────────────────────────────────────────────────────
function startLoop() {
    function tick() {
        const t = videoEl.currentTime;

        // Move playhead
        if (videoDuration > 0 && tCanvasW > 0) {
            playhead.style.transform = `translateX(${(t / videoDuration) * tCanvasW}px)`;
        }

        // Timestamp in overlay
        overlayTs.textContent = fmtTime(t);

        // Sync caption
        const idx = findIdx(t);
        if (idx !== currentIdx) {
            currentIdx = idx;
            updateCaption(idx);
            highlightEvent(idx);
        }

        requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

// Binary search: return index of last caption with timestamp <= t, or -1
function findIdx(t) {
    let lo = 0, hi = captions.length - 1, res = -1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (captions[mid].timestamp <= t) { res = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    return res;
}

// ─── Caption update ───────────────────────────────────────────────────────────
function updateCaption(idx) {
    if (idx < 0) {
        overlayEl.hidden = true;
        capText.textContent = '\u2014';
        setBar(confBar, confVal, null, CONF_MAX);
        setBar(gapBar,  gapVal,  null, GAP_MAX);
        setBar(simBar,  simVal,  null, SIM_MAX);
        altsLabel.hidden = true;
        altsList.innerHTML = '';
        return;
    }

    const ev = captions[idx];

    // Overlay
    overlayEl.hidden = false;
    overlayTxt.textContent  = ev.caption;
    overlayConf.textContent = `conf ${(ev.confidence * 100).toFixed(1)}%`;

    // Sidebar — caption text with source badge
    const source = ev.caption_source || 'clip';
    const badge = `<span class="source-badge ${source}">${source}</span>`;
    capText.innerHTML = escapeHtml(ev.caption) + badge;

    // Show original CLIP caption if BLIP-enhanced
    const clipOrigEl = $('clip-original');
    if (clipOrigEl) {
        if (ev.clip_caption && ev.caption_source === 'blip') {
            clipOrigEl.textContent = `CLIP: "${ev.clip_caption}"`;
            clipOrigEl.hidden = false;
        } else {
            clipOrigEl.hidden = true;
        }
    }

    setBar(confBar, confVal, ev.confidence,       CONF_MAX);
    setBar(gapBar,  gapVal,  ev.confidence_gap,   GAP_MAX);
    setBar(simBar,  simVal,  ev.change_similarity, SIM_MAX);

    // Alternatives
    const alts = (ev.alternatives || []).slice(0, 4);
    if (alts.length > 0) {
        altsLabel.hidden = false;
        altsList.innerHTML = alts.map(a =>
            `<div class="alt-row">
                <span class="alt-text">${escapeHtml(a.text)}</span>
                <span class="alt-score">${a.score.toFixed(3)}</span>
            </div>`
        ).join('');
    } else {
        altsLabel.hidden = true;
        altsList.innerHTML = '';
    }
}

// Set a metric bar to value/maxValue, displaying the raw value in the label
function setBar(barEl, valEl, value, maxValue) {
    if (value == null || !isFinite(value)) {
        barEl.style.width = '0';
        valEl.textContent = '\u2014';
    } else {
        barEl.style.width = (Math.min(value / maxValue, 1) * 100) + '%';
        valEl.textContent = value.toFixed(4);
    }
}

// ─── Event highlight + scroll ─────────────────────────────────────────────────
function highlightEvent(idx) {
    document.querySelectorAll('.ev-item.active')
        .forEach(el => el.classList.remove('active'));

    if (idx >= 0) {
        const el = $(`ei${idx}`);
        if (el) {
            el.classList.add('active');
            el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function fmtTime(sec) {
    if (!isFinite(sec) || sec < 0) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ─── Search Results ─────────────���───────────────────────────��─────────────────
function buildSearchResults() {
    const section = $('search-section');
    if (!searchResults.length) {
        section.hidden = true;
        return;
    }

    section.hidden = false;
    $('search-query').textContent = `"${searchQuery}"`;
    $('search-count').textContent = `(${searchResults.length})`;

    const list = $('search-list');
    list.innerHTML = searchResults.map(r =>
        `<div class="sr-item" data-ts="${r.timestamp}">
            <span class="sr-rank">#${r.rank}</span>
            <span class="sr-time">${fmtTime(r.timestamp)}</span>
            <span class="sr-sim">${r.similarity.toFixed(4)}</span>
        </div>`
    ).join('');

    list.addEventListener('click', e => {
        const item = e.target.closest('.sr-item');
        if (item) videoEl.currentTime = parseFloat(item.dataset.ts);
    });
}
