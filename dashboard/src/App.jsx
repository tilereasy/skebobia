import { useCallback, useEffect, useMemo, useRef, useState, memo } from "react";
import ForceGraph2D from "react-force-graph-2d";

const API_PROTOCOL = window.location.protocol === "https:" ? "https" : "http";
const API_HOST = window.location.hostname || "localhost";
const API_BASE = import.meta.env.VITE_API_BASE || `${API_PROTOCOL}://${API_HOST}:8000`;
const WS_PROTOCOL = window.location.protocol === "https:" ? "wss" : "ws";
const WS_HOST = window.location.hostname || "localhost";
const WS_URL = import.meta.env.VITE_WS_URL || `${WS_PROTOCOL}://${WS_HOST}:8000/ws/stream`;

const MAX_EVENTS = 300;
const RECONNECT_DELAY = 1500;
const MIN_GRAPH_WIDTH = 240;
const MIN_GRAPH_HEIGHT = 240;
const RECENT_EVENTS_LIMIT = 10;
const FEEDBACK_TIMEOUT = 5000;
const GRAPH_NODE_SIZE = 8;
const GRAPH_NODE_FONT_SIZE = 12;
const WORLD_ANCHOR_LABELS = {
  tracks: "у путей",
  path: "на тропе",
  vending_machines: "у автоматов",
  bench: "у лавки",
  stone_circle: "у круга камней",
  signpost: "у указателя",
  bushes_right: "у правых кустов",
};

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;900&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

  :root {
    --bg-void:       #050a10;
    --bg-panel:      #090f1a;
    --bg-panel-alt:  #0c1422;
    --bg-glass:      rgba(9,15,26,0.85);
    --border-dim:    rgba(0,220,255,0.15);
    --border-glow:   rgba(0,220,255,0.5);
    --cyan:          #00dcff;
    --cyan-dim:      rgba(0,220,255,0.6);
    --green:         #00ff9d;
    --green-dim:     rgba(0,255,157,0.6);
    --world-micro:   #7fff6b;
    --amber:         #ffb627;
    --red:           #ff3d5a;
    --text-primary:  #f0f8ff;
    --text-secondary:#7aa8c2;
    --text-muted:    #3d6882;
    --font-display:  'Orbitron', monospace;
    --font-mono:     'Space Mono', monospace;
    --glow-cyan:     0 0 8px rgba(0,220,255,0.6), 0 0 20px rgba(0,220,255,0.2);
    --glow-green:    0 0 8px rgba(0,255,157,0.6), 0 0 20px rgba(0,255,157,0.2);
    --glow-amber:    0 0 8px rgba(255,182,39,0.6), 0 0 20px rgba(255,182,39,0.2);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  select option {
  background: #0c1422;
  color: #e8f4ff;
  }

  html, body, #root {
  height: auto;
  background: var(--bg-void);
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.6;
  overflow: auto;
}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 2px; }

  /* ── Scanline overlay ── */
  #root::before {
    content: '';
    position: fixed; inset: 0; z-index: 9999; pointer-events: none;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.04) 2px,
      rgba(0,0,0,0.04) 4px
    );
  }

  /* ── Grid texture ── */
  #root::after {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
      linear-gradient(rgba(0,220,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,220,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
  }

  /* ── App Shell ── */
  .app-shell {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  overflow: visible;
  }

  /* ── Topbar ── */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px;
    background: linear-gradient(180deg, rgba(0,220,255,0.04) 0%, transparent 100%);
    border-bottom: 1px solid var(--border-glow);
    box-shadow: 0 1px 30px rgba(0,220,255,0.08);
    flex-shrink: 0;
  }

  .topbar-brand { display: flex; align-items: center; gap: 14px; }

  .topbar-logo {
    width: 36px; height: 36px;
    background: conic-gradient(from 180deg, var(--cyan), var(--green), var(--cyan));
    clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);
    animation: spin-slow 12s linear infinite;
    flex-shrink: 0;
  }

  @keyframes spin-slow { to { transform: rotate(360deg); } }

  .topbar h1 {
    font-family: var(--font-display);
    font-size: 16px; font-weight: 900;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--cyan);
    text-shadow: var(--glow-cyan);
  }

  .topbar-subtitle {
    font-size: 10px; letter-spacing: 0.12em;
    color: var(--text-muted); text-transform: uppercase;
    margin-top: 1px;
  }

  .topbar-actions { display: flex; align-items: center; gap: 10px; }

  .scene-link {
    font-family: var(--font-display); font-size: 10px; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--green); text-decoration: none;
    border: 1px solid var(--green-dim);
    padding: 5px 12px; border-radius: 2px;
    transition: all 0.2s;
    text-shadow: var(--glow-green);
    box-shadow: inset 0 0 12px rgba(0,255,157,0.04);
  }
  .scene-link:hover {
    background: rgba(0,255,157,0.08);
    box-shadow: var(--glow-green), inset 0 0 12px rgba(0,255,157,0.08);
  }

  /* ── WS Pill ── */
  .ws-pill {
    font-family: var(--font-display); font-size: 9px; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase;
    padding: 4px 10px; border-radius: 2px;
    border: 1px solid currentColor;
    display: flex; align-items: center; gap: 6px;
  }
  .ws-pill::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: currentColor; animation: blink 1.4s ease-in-out infinite;
  }
  @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.2; } }

  .ws-open    { color: var(--green);  border-color: var(--green-dim);  text-shadow: var(--glow-green); background-color: rgba(0,255,157,0.04);}
  .ws-connecting,.ws-reconnecting { color: var(--amber); border-color: rgba(255,182,39,0.4); text-shadow: var(--glow-amber); background-color: rgba(0,255,157,0.04);}
  .ws-error   { color: var(--red);    border-color: rgba(255,61,90,0.4); background-color: rgba(0,255,157,0.04);}

  /* ── Dashboard Grid ── */
  .dashboard-grid {
  flex: 1;
  overflow: visible;
  display: grid;
  grid-template-columns: 1fr 1fr 320px;
  grid-template-rows: 1fr;
  gap: 1px;
  background: var(--border-dim);
  padding: 1px;
  }


  /* ── Panel Base ── */
  .panel {
    background: var(--bg-panel);
    display: flex; flex-direction: column;
    overflow: hidden;
    position: relative;
    isolation: isolate;
  }

  .panel::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan-dim), transparent);
    opacity: 0.6;
    pointer-events: none;
    z-index: -1;
  }

  .panel-inner { padding: 14px; overflow-y: auto; flex: 1; }

  /* ── Panel Title Row ── */
  .panel-title-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px 8px;
    border-bottom: 1px solid var(--border-dim);
    flex-shrink: 0;
  }

  .panel-title-row h2, .panel-title-row h3 {
    font-family: var(--font-display);
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--cyan);
    text-shadow: var(--glow-cyan);
  }

  .panel-title-section { display: flex; align-items: center; gap: 8px; }

  /* ── Inline Controls ── */
  .inline-controls {
    display: flex; align-items: center; gap: 10px;
    flex-wrap: wrap;
  }

  label { color: var(--text-secondary); font-size: 11px; }

  select, textarea, input[type="range"], input[type="text"], input[type="number"] {
    background: rgba(0,220,255,0.04);
    border: 1px solid var(--border-glow);
    color: var(--text-primary);
    font-family: var(--font-mono); font-size: 11px;
    border-radius: 2px; outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  select:focus, textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
    border-color: var(--cyan);
    box-shadow: var(--glow-cyan);
  }
  select { padding: 3px 6px; margin-left: 6px; cursor: pointer; }
  input[type="text"], input[type="number"] { padding: 4px 6px; width: 100%; }
  textarea { width: 100%; padding: 6px 8px; resize: vertical; min-height: 52px; }

  input[type="range"] {
    width: 100%; height: 4px; cursor: pointer;
    -webkit-appearance: none; padding: 0; border: none;
    background: linear-gradient(90deg, var(--cyan), var(--green));
    border-radius: 2px;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px; height: 12px; border-radius: 50%;
    background: var(--cyan); box-shadow: var(--glow-cyan);
    cursor: pointer;
  }

  button[type="submit"], button[type="button"]:not(.collapse-btn):not(.inspect-close) {
    font-family: var(--font-display); font-size: 9px; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase;
    background: transparent; color: var(--cyan);
    border: 1px solid var(--border-glow);
    padding: 5px 12px; border-radius: 2px; cursor: pointer;
    transition: all 0.2s;
    text-shadow: var(--glow-cyan);
  }
  button[type="submit"]:hover, button[type="button"]:not(.collapse-btn):not(.inspect-close):hover {
    background: rgba(0,220,255,0.08);
    border-color: var(--cyan);
    box-shadow: var(--glow-cyan);
  }

  .collapse-btn {
    background: transparent; border: none; cursor: pointer;
    color: var(--text-muted); font-size: 10px;
    padding: 2px 6px; transition: color 0.2s;
  }
  .collapse-btn:hover { color: var(--cyan); }

  /* ── Checkbox Row ── */
  .checkbox-row {
    display: flex; align-items: center; gap: 5px; cursor: pointer;
  }
  input[type="checkbox"] {
    accent-color: var(--cyan); cursor: pointer;
  }

  /* ── Event Filters ── */
  .event-filters {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    padding: 6px 14px;
    border-bottom: 1px solid var(--border-dim);
    background: rgba(0,220,255,0.02);
    flex-shrink: 0;
  }
  .filter-label {
    font-size: 9px; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--text-muted);
  }
  .filter-checkbox {
    display: flex; align-items: center; gap: 4px;
    font-size: 11px; cursor: pointer; color: var(--text-secondary);
    transition: color 0.2s;
  }
  .filter-checkbox:hover { color: var(--cyan); }

  /* ── Event List ── */
  .event-list {
  flex: 1;
  overflow-y: auto;
  max-height: 600px;
  padding: 8px 0;
  background: var(--bg-panel);
  }

  .event-item {
    padding: 8px 14px;
    border-bottom: 1px solid rgba(0,220,255,0.04);
    transition: background 0.15s;
    animation: fadeSlideIn 0.3s ease;
    position: relative;
    z-index: 2;
    isolation: isolate;
    background: var(--bg-panel);
  }
  @keyframes fadeSlideIn {
    from { opacity: 0; transform: translateX(-6px); }
    to   { opacity: 1; transform: translateX(0); }
  }
  .event-item:hover { background: rgba(0,220,255,0.03); }

  .event-type-dialogue { border-left: 2px solid var(--cyan); padding-left: 12px; }
  .event-type-world    { border-left: 2px solid var(--green); padding-left: 12px; }
  .event-type-system   { border-left: 2px solid var(--amber); padding-left: 12px; }
  .event-type-other    { border-left: 2px solid var(--text-muted); padding-left: 12px; }
  .event-world-micro {
    border-left: 2px solid var(--world-micro);
    background: linear-gradient(90deg, rgba(127,255,107,0.08), rgba(0,0,0,0) 28%);
  }

  .event-meta {
    display: flex; align-items: center; gap: 8px; margin-bottom: 3px;
    font-size: 10px; color: var(--text-muted);
    flex-wrap: wrap;
  }
  .event-time { color: var(--text-muted); }
  .event-source {
    font-family: var(--font-display); font-size: 9px; font-weight: 600;
    letter-spacing: 0.1em; color: var(--cyan); text-shadow: 0 0 6px rgba(0,220,255,0.5);
  }
  .event-tags {
    font-size: 9px; color: var(--text-muted);
    background: rgba(0,220,255,0.06); padding: 1px 5px; border-radius: 2px;
  }
  .event-badge {
    font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase;
    border: 1px solid transparent; border-radius: 2px;
    padding: 1px 5px;
  }
  .event-badge-world {
    color: var(--world-micro);
    border-color: rgba(127,255,107,0.45);
    background: rgba(127,255,107,0.1);
  }
  .event-anchor {
    font-size: 10px;
    color: #d4ffbf;
    border: 1px solid rgba(127,255,107,0.35);
    background: rgba(127,255,107,0.08);
    border-radius: 2px;
    padding: 1px 6px;
  }
  .event-item p {
    color: #ffffff !important;
    font-size: 12.5px;
    line-height: 1.55;
    position: relative;
    z-index: 2;
  }

  .empty-state {
    padding: 32px 20px; text-align: center;
    color: var(--text-muted); font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  /* ── Graph Panel ── */
  .graph-panel-compact {
  background: var(--bg-panel-alt);
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  grid-column: 1 / 3;
  }   

  .graph-body {
    flex: 1;
    min-height: 0;
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: center;
  }

  .muted { font-size: 10px; color: var(--text-muted); }

  /* ── Side Panel ── */
  
  .side-panel {
  overflow-y: auto;
  max-height: 100vh;
  gap: 0;
  }

  .panel-block {
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 4px;
  }
  .panel-block:last-child { border-bottom: none; }

  /* ── Agent Cards ── */
  .agent-cards { padding: 8px 10px; display: flex; flex-direction: column; gap: 8px; }

  .agent-card {
    background: rgba(0,220,255,0.03);
    border: 1px solid var(--border-dim);
    border-radius: 3px; padding: 10px 12px;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative; overflow: auto;
  }
  .agent-card::before {
    content: '';
    position: absolute; top: 0; left: 0; bottom: 0; width: 2px;
    background: linear-gradient(180deg, var(--cyan), var(--green));
    opacity: 0.7;
  }
  .agent-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 0 16px rgba(0,220,255,0.08);
  }

  .agent-card-head {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 4px;
  }
  .agent-card-head strong {
    font-family: var(--font-display); font-size: 10px; font-weight: 600;
    letter-spacing: 0.12em; color: var(--text-primary);
  }
  .agent-card p {
    font-size: 11px; color: var(--text-secondary);
    margin-bottom: 8px; line-height: 1.4;
  }
  .agent-card button {
    font-size: 9px; padding: 3px 9px;
  }

  /* ── Mood Badge ── */
  .mood {
    font-family: var(--font-display); font-size: 8px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 2px 6px; border-radius: 2px;
  }
  .mood {
    font-family: var(--font-display); font-size: 8px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 2px 6px; border-radius: 2px;
    color: var(--text-muted); border: 1px solid var(--border-dim);
  }

  .mood {
    font-family: var(--font-display);
    font-size: 8px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 2px;
    color: var(--text-muted);
    border: 1px solid var(--border-dim);
  }

  .mood-happy {
    color: #00ff9d;
    border-color: rgba(0,255,157,0.5);
    background: rgba(0,255,157,0.08);
    text-shadow: 0 0 8px rgba(0,255,157,0.7);
  }

  .mood-excited {
    color: #ffe066;
    border-color: rgba(255,224,102,0.5);
    background: rgba(255,224,102,0.08);
    text-shadow: 0 0 8px rgba(255,224,102,0.7);
  }

  .mood-neutral {
    color: #00dcff;
    border-color: rgba(0,220,255,0.4);
    background: rgba(0,220,255,0.06);
    text-shadow: 0 0 8px rgba(0,220,255,0.5);
  }

  .mood-calm {
    color: #7ecfff;
    border-color: rgba(126,207,255,0.4);
    background: rgba(126,207,255,0.06);
    text-shadow: 0 0 8px rgba(126,207,255,0.5);
  }

  .mood-sad {
    color: #7b9fff;
    border-color: rgba(123,159,255,0.4);
    background: rgba(123,159,255,0.06);
    text-shadow: 0 0 8px rgba(123,159,255,0.5);
  }

  .mood-angry {
    color: #ff3d5a;
    border-color: rgba(255,61,90,0.5);
    background: rgba(255,61,90,0.08);
    text-shadow: 0 0 8px rgba(255,61,90,0.7);
  }

  .mood-anxious {
    color: #ffb627;
    border-color: rgba(255,182,39,0.5);
    background: rgba(255,182,39,0.08);
    text-shadow: 0 0 8px rgba(255,182,39,0.6);
  }

  .mood-afraid {
    color: #ff7eb3;
    border-color: rgba(255,126,179,0.4);
    background: rgba(255,126,179,0.06);
    text-shadow: 0 0 8px rgba(255,126,179,0.5);
  }

  /* ── Control Panel ── */
  .control-form {
    display: flex; flex-direction: column; gap: 5px;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-dim);
  }
  .control-form label {
    font-family: var(--font-display); font-size: 9px; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted);
  }
  .control-feedback {
    margin: 6px 12px;
    font-size: 10px; color: var(--green);
    text-shadow: var(--glow-green);
    animation: fadeSlideIn 0.3s ease;
    letter-spacing: 0.05em;
  }

  /* ── Inspect Drawer ── */
  .inspect-drawer {
    position: fixed; right: 0; top: 0; bottom: 0;
    width: 360px; z-index: 100;
    background: var(--bg-panel);
    border-left: 1px solid var(--border-glow);
    box-shadow: -8px 0 40px rgba(0,220,255,0.1);
    display: flex; flex-direction: column;
    animation: slideInRight 0.25s cubic-bezier(0.22,1,0.36,1);
    overflow: hidden;
  }
  @keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to   { transform: translateX(0);    opacity: 1; }
  }

  .inspect-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border-glow);
    background: rgba(0,220,255,0.04);
    flex-shrink: 0;
  }
  .inspect-header h3 {
    font-family: var(--font-display); font-size: 11px; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--cyan); text-shadow: var(--glow-cyan);
  }

  .inspect-close {
    background: transparent; border: 1px solid rgba(255,61,90,0.4) !important;
    color: var(--red) !important;
    font-family: var(--font-display) !important; font-size: 9px !important;
    letter-spacing: 0.12em !important; padding: 4px 10px !important;
    cursor: pointer; border-radius: 2px;
    transition: all 0.2s !important;
  }
  .inspect-close:hover {
    background: rgba(255,61,90,0.1) !important;
    box-shadow: 0 0 8px rgba(255,61,90,0.4) !important;
  }

  .inspect-content {
    flex: 1; overflow-y: auto; padding: 14px 16px;
    display: flex; flex-direction: column; gap: 10px;
  }

  .inspect-field { display: flex; flex-direction: column; gap: 2px; }
  .inspect-field .field-label {
    font-family: var(--font-display); font-size: 8px; font-weight: 600;
    letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted);
  }
  .inspect-field .field-value { font-size: 12px; color: var(--text-primary); }

  .inspect-section-title {
    font-family: var(--font-display); font-size: 9px; font-weight: 600;
    letter-spacing: 0.2em; text-transform: uppercase; color: var(--amber);
    text-shadow: var(--glow-amber);
    margin-top: 6px; padding-bottom: 4px;
    border-bottom: 1px solid rgba(255,182,39,0.2);
  }

  .inspect-list {
    list-style: none; display: flex; flex-direction: column; gap: 6px;
  }
  .inspect-list li {
    font-size: 11px; color: var(--text-secondary);
    padding: 5px 8px;
    background: rgba(0,220,255,0.03);
    border: 1px solid var(--border-dim);
    border-radius: 2px;
    line-height: 1.5;
  }
  .inspect-list li span {
    font-size: 10px; color: var(--text-muted); margin-right: 6px;
  }

  .error-message { color: var(--red); font-size: 11px; padding: 10px 16px; }

  /* ── Loading Screen ── */
  .loading-screen {
    display: flex; align-items: center; justify-content: center;
    height: 100vh; flex-direction: column; gap: 16px;
  }
  .loading-spinner {
    width: 40px; height: 40px;
    border: 2px solid var(--border-dim);
    border-top-color: var(--cyan);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-text {
    font-family: var(--font-display); font-size: 11px; font-weight: 600;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--cyan); text-shadow: var(--glow-cyan);
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  @media (max-width: 1100px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
    grid-template-areas:
    "feed"
    "graph"
    "side";
  }

  .feed-panel         { grid-area: feed;}
  .graph-panel-compact { grid-area: graph; }
  .side-panel         { grid-area: side; }
  .graph-panel-compact { grid-column: auto; }
}
`;

function apiPath(path) { return `${API_BASE}${path}`; }
function trimEvents(events) { return events.slice(-MAX_EVENTS); }

function appendEvent(events, nextEvent) {
  if (!nextEvent || typeof nextEvent !== "object") return events;
  if (nextEvent.id && events.some((e) => e.id === nextEvent.id)) return events;
  return trimEvents([...events, nextEvent]);
}

async function fetchJson(url, init) {
  const response = await fetch(url, init);
  const raw = await response.text();
  let data = null;
  if (raw) {
    try { data = JSON.parse(raw); } catch (e) { data = null; }
  }
  if (!response.ok) {
    const detail = (data && typeof data === "object" && data.detail) || raw || `HTTP ${response.status}`;
    throw new Error(String(detail));
  }
  if (data === null) throw new Error("Response is not valid JSON");
  return data;
}

function postJson(path, body) {
  return fetchJson(apiPath(path), {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
}

function fallbackNodeColor(id) {
  const palette = ["#00dcff", "#00ff9d", "#ffb627", "#ff3d5a", "#8b5cf6", "#06b6d4"];
  const key = String(id || "agent");
  let hash = 0;
  for (let i = 0; i < key.length; i++) { hash = (hash << 5) - hash + key.charCodeAt(i); hash |= 0; }
  return palette[Math.abs(hash) % palette.length];
}

function moodClass(moodLabel) { return `mood mood-${moodLabel || "neutral"}`; }

function hasTag(event, expectedTag) {
  if (!event || !Array.isArray(event.tags)) return false;
  const needle = String(expectedTag || "").toLowerCase();
  return event.tags.some((tag) => String(tag || "").toLowerCase() === needle);
}

function extractEventAnchor(event) {
  if (event && typeof event.anchor === "string" && event.anchor) return event.anchor;
  if (!event || !Array.isArray(event.tags)) return "";
  const anchorTag = event.tags.find((tag) => typeof tag === "string" && tag.startsWith("anchor:"));
  return anchorTag ? String(anchorTag).split(":", 2)[1] : "";
}

function anchorLabel(anchor) {
  if (!anchor) return "";
  return WORLD_ANCHOR_LABELS[anchor] || anchor.replaceAll("_", " ");
}

function isWorldMicroEvent(event) {
  return hasTag(event, "world") && hasTag(event, "micro");
}

function sourceLabel(event, agentById) {
  if (event.source_id && agentById.has(event.source_id)) return agentById.get(event.source_id).name;
  if (isWorldMicroEvent(event)) return "World Micro";
  if (event.source_type === "world") return "World";
  return "Unknown";
}

function getEventType(event) {
  if (Array.isArray(event.tags)) {
    if (event.tags.includes("dialogue")) return "dialogue";
    if (event.tags.includes("system")) return "system";
  }
  if (event.source_type === "world") return "world";
  if (event.text?.includes("said:") || event.text?.includes("сказал:")) return "dialogue";
  return "other";
}

const AgentCard = memo(({ agent, onInspect }) => (
  <article className="agent-card">
    <div className="agent-card-head">
      <strong>{agent.name}</strong>
      <span className={moodClass(agent.mood_label)}>{agent.mood_label}</span>
    </div>
    <p>{agent.current_plan || "—"}</p>
    <button type="button" onClick={() => onInspect(agent.id)}>Inspect</button>
  </article>
));
AgentCard.displayName = "AgentCard";

export default function App() {
  const [agents, setAgents] = useState([]);
  const [relations, setRelations] = useState({ nodes: [], edges: [] });
  const [events, setEvents] = useState([]);
  const [simStats, setSimStats] = useState(null);
  const [filterAgentId, setFilterAgentId] = useState("all");
  const [autoScroll, setAutoScroll] = useState(true);
  const [wsStatus, setWsStatus] = useState("connecting");
  const [inspectAgentId, setInspectAgentId] = useState("");
  const [inspectData, setInspectData] = useState(null);
  const [inspectLoading, setInspectLoading] = useState(false);
  const [inspectError, setInspectError] = useState("");
  const [worldEventText, setWorldEventText] = useState("");
  const [messageText, setMessageText] = useState("");
  const [messageAgentId, setMessageAgentId] = useState("");
  const [newAgentName, setNewAgentName] = useState("");
  const [newAgentTraits, setNewAgentTraits] = useState("нейтральный, любопытный");
  const [newAgentMood, setNewAgentMood] = useState(0);
  const [newAgentAvatar, setNewAgentAvatar] = useState("");
  const [removeAgentId, setRemoveAgentId] = useState("");
  const [speed, setSpeed] = useState(1);
  const [controlFeedback, setControlFeedback] = useState("");
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [eventFilters, setEventFilters] = useState({ dialogue: true, world: true, system: true, other: true });
  const [collapsedPanels, setCollapsedPanels] = useState({ graph: false, controls: false });

  const feedRef = useRef(null);
  const graphWrapRef = useRef(null);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(true);
  const [graphSize, setGraphSize] = useState({ width: MIN_GRAPH_WIDTH, height: MIN_GRAPH_HEIGHT });

  const agentById = useMemo(() => new Map(agents.map((a) => [a.id, a])), [agents]);

  const filteredEvents = useMemo(() => {
    let f = events;
    if (filterAgentId !== "all") f = f.filter((e) => e.source_id === filterAgentId);
    f = f.filter((e) => eventFilters[getEventType(e)] !== false);
    return f;
  }, [events, filterAgentId, eventFilters]);

  const graphData = useMemo(() => ({
    nodes: relations.nodes.filter((n) => agentById.has(n.id)).map((n) => ({
      ...n, color: agentById.get(n.id)?.avatar || fallbackNodeColor(n.id),
    })),
    links: relations.edges.map((e) => ({ ...e })),
  }), [relations, agentById]);

  const buildLocalInspectData = useCallback((agentId) => {
    const agent = agentById.get(agentId);
    if (!agent) return null;
    const relationItems = (relations.edges || [])
      .filter((e) => e.from === agentId)
      .map((e) => ({
        agent_id: e.to,
        name: agentById.get(e.to)?.name || e.to,
        value: Number(e.value || 0),
      }))
      .sort((a, b) => b.value - a.value);
    return {
      id: agent.id, name: agent.name, traits: "n/a (local fallback)",
      mood: agent.mood, mood_label: agent.mood_label, current_plan: agent.current_plan,
      key_memories: [{ text: "Inspector fallback from live stream.", score: null }],
      recent_events: events.filter((e) => e.source_id === agentId).slice(-RECENT_EVENTS_LIMIT),
      relations_snapshot: {
        top_positive: relationItems.filter((i) => i.value >= 0).slice(0, 3),
        top_negative: relationItems.filter((i) => i.value < 0).slice(-3).reverse(),
      },
    };
  }, [agentById, events, relations.edges]);

  const fetchInspect = useCallback(async (agentId) => {
    if (!agentId) return;
    setInspectAgentId(agentId);
    setInspectLoading(true);
    setInspectError("");
    try {
      const data = await fetchJson(apiPath(`/api/agents/${encodeURIComponent(agentId)}`), {
        method: "GET", headers: { Accept: "application/json" },
      });
      setInspectData(data); setInspectError("");
    } catch (error) {
      const fallback = buildLocalInspectData(agentId);
      if (fallback) { setInspectData(fallback); setControlFeedback(`Inspector fallback: ${error}`); }
      else { setInspectData(null); setInspectError(`Failed: ${error}`); }
    } finally { setInspectLoading(false); }
  }, [buildLocalInspectData]);

  const toggleEventFilter = useCallback((t) => setEventFilters((p) => ({ ...p, [t]: !p[t] })), []);
  const togglePanel = useCallback((n) => setCollapsedPanels((p) => ({ ...p, [n]: !p[n] })), []);

  useEffect(() => {
    if (controlFeedback) { const t = setTimeout(() => setControlFeedback(""), FEEDBACK_TIMEOUT); return () => clearTimeout(t); }
  }, [controlFeedback]);

  useEffect(() => {
    async function load() {
      try {
        const s = await fetchJson(apiPath("/api/state"), { method: "GET", headers: { Accept: "application/json" } });
        setAgents(Array.isArray(s.agents) ? s.agents : []);
        setRelations(s.relations || { nodes: [], edges: [] });
        setEvents(trimEvents(Array.isArray(s.events) ? s.events : []));
        if (typeof s.speed === "number") setSpeed(s.speed);
        setSimStats({
          tick: s.tick,
          speed: s.speed,
          runtime: s.runtime || null,
          llm_stats: s.llm_stats || null,
          world_event_stats: s.world_event_stats || null,
          memory_stats: s.memory_stats || null,
        });
      } catch (e) { setControlFeedback(`Load failed: ${e}`); }
      finally { setIsInitialLoading(false); }
    }
    load();
  }, []);

  useEffect(() => {
    if (agents.length === 0) return;
    if (!messageAgentId || !agents.some((a) => a.id === messageAgentId)) setMessageAgentId(agents[0].id);
    if (!removeAgentId || !agents.some((a) => a.id === removeAgentId)) setRemoveAgentId(agents[0].id);
  }, [agents, messageAgentId, removeAgentId]);

  useEffect(() => {
    if (filterAgentId !== "all" && !agents.some((a) => a.id === filterAgentId)) setFilterAgentId("all");
  }, [agents, filterAgentId]);

  useEffect(() => {
    if (autoScroll && feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [filteredEvents, autoScroll]);

  useEffect(() => {
    if (!graphWrapRef.current) return;
    const target = graphWrapRef.current;
    const update = () => {
      const r = target.getBoundingClientRect();
      setGraphSize({ width: Math.max(MIN_GRAPH_WIDTH, Math.floor(r.width)), height: Math.max(MIN_GRAPH_HEIGHT, Math.floor(r.height)) });
    };
    update();
    if (typeof ResizeObserver === "undefined") { window.addEventListener("resize", update); return () => window.removeEventListener("resize", update); }
    const obs = new ResizeObserver(update);
    obs.observe(target);
    return () => obs.disconnect();
  }, [collapsedPanels.graph]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    function connect() {
      setWsStatus("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => setWsStatus("open");
      ws.onmessage = (msg) => {
        try {
          const p = JSON.parse(msg.data);
          if (!p.type || p.payload === undefined) return;
          if (p.type === "agents_state" && Array.isArray(p.payload)) { setAgents(p.payload); return; }
          if (p.type === "event" && p.payload) { setEvents((prev) => appendEvent(prev, p.payload)); return; }
          if (p.type === "relations" && p.payload) { setRelations(p.payload); return; }
        } catch (e) { setControlFeedback("WS: malformed message"); }
      };
      ws.onerror = () => setWsStatus("error");
      ws.onclose = () => {
        if (!shouldReconnectRef.current) return;
        setWsStatus("reconnecting");
        reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY);
      };
    }
    connect();
    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  useEffect(() => {
    if (isInitialLoading) return undefined;
    let alive = true;
    const refresh = async () => {
      try {
        const s = await fetchJson(apiPath("/api/state"), { method: "GET", headers: { Accept: "application/json" } });
        if (!alive) return;
        setSimStats({
          tick: s.tick,
          speed: s.speed,
          runtime: s.runtime || null,
          llm_stats: s.llm_stats || null,
          world_event_stats: s.world_event_stats || null,
          memory_stats: s.memory_stats || null,
        });
      } catch (_e) {
        // periodic stats refresh
      }
    };
    const timer = setInterval(refresh, 3000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [isInitialLoading]);

  async function submitWorldEvent(e) {
    e.preventDefault();
    const text = worldEventText.trim();
    if (!text) return;
    try { await postJson("/api/control/event", { text }); setWorldEventText(""); setControlFeedback("World event sent"); }
    catch (err) { setControlFeedback(`Error: ${err}`); }
  }

  async function submitMessage(e) {
    e.preventDefault();
    const text = messageText.trim();
    if (!text || !messageAgentId) return;
    try { await postJson("/api/control/message", { agent_id: messageAgentId, text }); setMessageText(""); setControlFeedback("Message sent"); }
    catch (err) { setControlFeedback(`Error: ${err}`); }
  }

  async function submitSpeed(e) {
    e.preventDefault();
    try {
      const data = await postJson("/api/control/speed", { speed });
      if (typeof data.speed === "number") setSpeed(data.speed);
      setControlFeedback(`Speed → ${Number(speed).toFixed(1)}x`);
    } catch (err) { setControlFeedback(`Error: ${err}`); }
  }

  async function submitAddAgent(e) {
    e.preventDefault();
    const name = newAgentName.trim();
    if (!name) return;
    try {
      await postJson("/api/control/agent/add", {
        name,
        traits: newAgentTraits.trim() || "нейтральный",
        mood: Number(newAgentMood),
        avatar: newAgentAvatar.trim() || null,
      });
      setNewAgentName("");
      setControlFeedback("Agent added");
    } catch (err) {
      setControlFeedback(`Error: ${err}`);
    }
  }

  async function submitRemoveAgent(e) {
    e.preventDefault();
    if (!removeAgentId) return;
    try {
      await postJson("/api/control/agent/remove", { agent_id: removeAgentId });
      setControlFeedback("Agent removed");
    } catch (err) {
      setControlFeedback(`Error: ${err}`);
    }
  }

  if (isInitialLoading) {
    return (
      <>
        <style>{STYLES}</style>
        <div className="app-shell loading-screen">
          <div className="loading-spinner" />
          <div className="loading-text">Initializing Skebobia</div>
        </div>
      </>
    );
  }

  return (
    <>
      <style>{STYLES}</style>
      <div className="app-shell">

        {/* ── Header ── */}
        <header className="topbar">
          <div className="topbar-brand">
            <div className="topbar-logo" aria-hidden="true" />
            <div>
              <h1>Skebobia</h1>
              <div className="topbar-subtitle">Realtime Simulation Dashboard</div>
            </div>
          </div>
          <div className="topbar-actions">
            <a href="/scene/" target="_blank" rel="noopener noreferrer" className="scene-link">
              🎮 Unity Scene
            </a>
            <div className={`ws-pill ws-${wsStatus}`} aria-label={`WebSocket: ${wsStatus}`}>
              {wsStatus}
            </div>
          </div>
        </header>

        <main className="dashboard-grid">

          <section className="panel feed-panel">
            <div className="panel-title-row">
              <div className="panel-title-section">
                <h2>Event Feed</h2>
                <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                  [{filteredEvents.length}]
                </span>
              </div>
              <div className="inline-controls">
                <label>
                  Agent:
                  <select value={filterAgentId} onChange={(e) => setFilterAgentId(e.target.value)}>
                    <option value="all">All</option>
                    {agents.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
                  </select>
                </label>
                <label className="checkbox-row">
                  <input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} />
                  auto-scroll
                </label>
              </div>
            </div>

            <div className="event-filters">
              <span className="filter-label">Show:</span>
              {[
                ["dialogue", "Dialogue"],
                ["world", "World"],
                ["system", "System"],
              ].map(([key, label]) => (
                <label key={key} className="filter-checkbox">
                  <input type="checkbox" checked={eventFilters[key]} onChange={() => toggleEventFilter(key)} />
                  {label}
                </label>
              ))}
            </div>

            <div className="event-list" ref={feedRef} role="log" aria-live="polite">
              {filteredEvents.length === 0 && <div className="empty-state">No events match filters</div>}
              {filteredEvents.map((event) => {
                const type = getEventType(event);
                const worldMicro = isWorldMicroEvent(event);
                const anchor = extractEventAnchor(event);
                return (
                  <article key={event.id} className={`event-item event-type-${type}${worldMicro ? " event-world-micro" : ""}`}>
                    <div className="event-meta">
                      <span className="event-time">{event.ts || "—"}</span>
                      <span className="event-source">{sourceLabel(event, agentById)}</span>
                      {worldMicro && <span className="event-badge event-badge-world">world micro</span>}
                      {anchor && <span className="event-anchor">{anchorLabel(anchor)}</span>}
                      {Array.isArray(event.tags) && event.tags.length > 0 && (
                        <span className="event-tags">{event.tags.join(", ")}</span>
                      )}
                    </div>
                    <p>{event.text}</p>
                    {Array.isArray(event.evidence_ids) && event.evidence_ids.length > 0 && (
                      <div className="event-meta" style={{ marginTop: 4 }}>
                        <span className="event-tags">evidence: {event.evidence_ids.join(", ")}</span>
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
          </section>

          <section className="panel graph-panel-compact" style={{ minHeight: 320, position: "relative", display: "flex", flexDirection: "column" }}>
            <div className="panel-title-row">
              <h2>Relations Graph</h2>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="muted">click node → inspect</span>
                <button type="button" className="collapse-btn" onClick={() => togglePanel("graph")}>
                  {collapsedPanels.graph ? "▼" : "▲"}
                </button>
              </div>
            </div>
            {!collapsedPanels.graph && (
            <div className="graph-body" ref={graphWrapRef}>
              <ForceGraph2D
                graphData={graphData}
                width={graphSize.width}
                height={graphSize.height}
                linkSource="from"
                linkTarget="to"
                backgroundColor="transparent"
                nodeLabel={(n) => n.name}
                linkColor={(l) => l.value >= 0 ? "rgba(0,255,157,0.8)" : "rgba(255,61,90,0.8)"}
                linkWidth={(l) => 1.5 + Math.abs(l.value || 0) / 20}
                enableNodeDrag={true}
                enableZoomInteraction={true}
                nodeCanvasObject={(node, ctx, globalScale) => {
                  const label = node.name || node.id;
                  const fontSize = GRAPH_NODE_FONT_SIZE / globalScale;
                  ctx.shadowColor = node.color || fallbackNodeColor(node.id);
                  ctx.shadowBlur = 12;
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, GRAPH_NODE_SIZE, 0, 2 * Math.PI);
                  ctx.fillStyle = node.color || fallbackNodeColor(node.id);
                  ctx.fill();
                  ctx.shadowBlur = 0;
                  ctx.strokeStyle = "rgba(0,220,255,0.4)";
                  ctx.lineWidth = 1;
                  ctx.stroke();
                  ctx.font = `${fontSize}px 'Space Mono'`;
                  ctx.fillStyle = "#e8f4ff";
                  ctx.fillText(label, node.x + GRAPH_NODE_SIZE + 3, node.y + fontSize / 3);
                }}
                onNodeClick={(node) => fetchInspect(node.id)}
              />
            </div>
          )}
          </section>

          <section className="panel side-panel">
            <div className="panel-block">
              <div className="panel-title-row">
                <h2>Agent Cards</h2>
                <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{agents.length} active</span>
              </div>
              <div className="agent-cards">
                {agents.map((a) => <AgentCard key={a.id} agent={a} onInspect={fetchInspect} />)}
              </div>
            </div>

            <div className="panel-block">
              <div className="panel-title-row">
                <h2>Control Panel</h2>
                <button type="button" className="collapse-btn" onClick={() => togglePanel("controls")}>
                  {collapsedPanels.controls ? "▼" : "▲"}
                </button>
              </div>

              {!collapsedPanels.controls && (
                <>
                  <form onSubmit={submitWorldEvent} className="control-form">
                    <label htmlFor="world-event">World Event</label>
                    <textarea
                      id="world-event"
                      rows={3}
                      value={worldEventText}
                      onChange={(e) => setWorldEventText(e.target.value)}
                      placeholder="Meteor shower over the market..."
                    />
                    <button type="submit">↗ Broadcast</button>
                  </form>

                  <form onSubmit={submitMessage} className="control-form">
                    <label htmlFor="msg-agent">Direct Message</label>
                    <select id="msg-agent" value={messageAgentId} onChange={(e) => setMessageAgentId(e.target.value)}>
                      {agents.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
                    </select>
                    <textarea
                      rows={2}
                      value={messageText}
                      onChange={(e) => setMessageText(e.target.value)}
                      placeholder="Patrol the square..."
                    />
                    <button type="submit">↗ Send</button>
                  </form>

                  <form onSubmit={submitSpeed} className="control-form">
                    <label htmlFor="speed-slider">Sim Speed — {Number(speed).toFixed(1)}×</label>
                    <input
                      id="speed-slider"
                      type="range" min="0.1" max="5" step="0.1"
                      value={speed}
                      onChange={(e) => setSpeed(Number(e.target.value))}
                    />
                    <button type="submit">↗ Apply</button>
                  </form>

                  <form onSubmit={submitAddAgent} className="control-form">
                    <label htmlFor="agent-name">Add Agent</label>
                    <input
                      id="agent-name"
                      type="text"
                      value={newAgentName}
                      onChange={(e) => setNewAgentName(e.target.value)}
                      placeholder="Имя агента"
                    />
                    <input
                      type="text"
                      value={newAgentTraits}
                      onChange={(e) => setNewAgentTraits(e.target.value)}
                      placeholder="Черты"
                    />
                    <div style={{ display: "flex", gap: 8 }}>
                      <input
                        type="number"
                        min={-100}
                        max={100}
                        value={newAgentMood}
                        onChange={(e) => setNewAgentMood(Number(e.target.value))}
                        style={{ width: 84 }}
                        placeholder="Mood"
                      />
                      <input
                        type="text"
                        value={newAgentAvatar}
                        onChange={(e) => setNewAgentAvatar(e.target.value)}
                        placeholder="#hex (опц.)"
                      />
                    </div>
                    <button type="submit">↗ Add Agent</button>
                  </form>

                  <form onSubmit={submitRemoveAgent} className="control-form">
                    <label htmlFor="remove-agent">Remove Agent</label>
                    <select id="remove-agent" value={removeAgentId} onChange={(e) => setRemoveAgentId(e.target.value)}>
                      {agents.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
                    </select>
                    <button type="submit">↗ Remove Agent</button>
                  </form>
                </>
              )}

              {controlFeedback && (
                <p className="control-feedback" role="status" aria-live="polite">
                  ✓ {controlFeedback}
                </p>
              )}
            </div>

            <div className="panel-block">
              <div className="panel-title-row">
                <h2>Live Metrics</h2>
              </div>
              <div className="inspect-content" style={{ gap: 6 }}>
                <div className="inspect-field">
                  <span className="field-label">Tick</span>
                  <span className="field-value">{simStats?.tick ?? "—"}</span>
                </div>
                <div className="inspect-field">
                  <span className="field-label">Queue Pending</span>
                  <span className="field-value">{simStats?.llm_stats?.reply_queue?.pending ?? "—"}</span>
                </div>
                <div className="inspect-field">
                  <span className="field-label">Avg Tick ms</span>
                  <span className="field-value">{simStats?.runtime?.avg_tick_ms ?? "—"}</span>
                </div>
                <div className="inspect-field">
                  <span className="field-label">World / 100 ticks</span>
                  <span className="field-value">{simStats?.world_event_stats?.metrics?.world_events_per_100_ticks ?? "—"}</span>
                </div>
                <div className="inspect-field">
                  <span className="field-label">World evidence ratio</span>
                  <span className="field-value">{simStats?.world_event_stats?.metrics?.agent_world_evidence_ratio_100_ticks ?? "—"}</span>
                </div>
                <div className="inspect-field">
                  <span className="field-label">Repeat ratio (50)</span>
                  <span className="field-value">{simStats?.world_event_stats?.metrics?.dialogue_repeat_ratio_recent_50 ?? "—"}</span>
                </div>
                <div className="inspect-field">
                  <span className="field-label">Memory entries</span>
                  <span className="field-value">{simStats?.memory_stats?.entries ?? "—"}</span>
                </div>
              </div>
            </div>
          </section>
        </main>

        {inspectAgentId && (
          <aside className="inspect-drawer" role="dialog" aria-label="Agent inspector">
            <div className="inspect-header">
              <h3>// Inspector: {inspectData?.name || inspectAgentId}</h3>
              <button
                type="button"
                className="inspect-close"
                onClick={() => { setInspectAgentId(""); setInspectData(null); setInspectError(""); }}
              >
                ✕ Close
              </button>
            </div>

            {inspectLoading && (
              <div style={{ padding: 20, display: "flex", gap: 10, alignItems: "center" }}>
                <div className="loading-spinner" style={{ width: 20, height: 20 }} />
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Loading...</span>
              </div>
            )}
            {inspectError && <p className="error-message">{inspectError}</p>}

            {inspectData && !inspectLoading && (
              <div className="inspect-content">
                <div className="inspect-field">
                  <span className="field-label">Traits</span>
                  <span className="field-value">{inspectData.traits || "—"}</span>
                </div>
                <div className="inspect-field">
                  <span className="field-label">Mood</span>
                  <span className="field-value">
                    <span className={moodClass(inspectData.mood_label)}>{inspectData.mood_label}</span>
                    {" "}
                    <span style={{ color: "var(--text-muted)", fontSize: 11 }}>({inspectData.mood})</span>
                  </span>
                </div>
                <div className="inspect-field">
                  <span className="field-label">Current Plan</span>
                  <span className="field-value">{inspectData.current_plan || "—"}</span>
                </div>

                <div className="inspect-section-title">Key Memories</div>
                <ul className="inspect-list">
                  {(inspectData.key_memories || []).map((m, i) => (
                    <li key={`mem-${i}`}>
                      {m.text}
                      {typeof m.score === "number" && (
                        <span style={{ color: "var(--amber)", marginLeft: 6, fontSize: 10 }}>
                          [{m.score.toFixed(2)}]
                        </span>
                      )}
                    </li>
                  ))}
                </ul>

                <div className="inspect-section-title">Recent Events</div>
                <ul className="inspect-list">
                  {(inspectData.recent_events || []).map((e) => (
                    <li key={e.id}>
                      <span>{e.ts || "—"}</span>{e.text}
                    </li>
                  ))}
                </ul>

                <div className="inspect-section-title">Relations</div>
                <ul className="inspect-list">
                  {(inspectData.relations_snapshot?.top_positive || []).map((item) => (
                    <li key={`rel-pos-${item.agent_id}`}>
                      <span>+{item.value}</span>{item.name}
                    </li>
                  ))}
                  {(inspectData.relations_snapshot?.top_negative || []).map((item) => (
                    <li key={`rel-neg-${item.agent_id}`}>
                      <span>{item.value}</span>{item.name}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </aside>
        )}
      </div>
    </>
  );
}
