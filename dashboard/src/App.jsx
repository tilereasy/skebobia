import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

const API_PROTOCOL = window.location.protocol === "https:" ? "https" : "http";
const API_HOST = window.location.hostname || "localhost";
const API_BASE = import.meta.env.VITE_API_BASE || `${API_PROTOCOL}://${API_HOST}:8000`;
const WS_PROTOCOL = window.location.protocol === "https:" ? "wss" : "ws";
const WS_HOST = window.location.hostname || "localhost";
const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${WS_PROTOCOL}://${WS_HOST}:8000/ws/stream`;
const MAX_EVENTS = 300;

function apiPath(path) {
  return `${API_BASE}${path}`;
}

function trimEvents(events) {
  return events.slice(-MAX_EVENTS);
}

function appendEvent(events, nextEvent) {
  if (!nextEvent || typeof nextEvent !== "object") {
    return events;
  }
  if (nextEvent.id && events.some((event) => event.id === nextEvent.id)) {
    return events;
  }
  return trimEvents([...events, nextEvent]);
}

async function fetchJson(url, init) {
  const response = await fetch(url, init);
  const raw = await response.text();
  let data = null;

  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch {
      data = null;
    }
  }

  if (!response.ok) {
    const detail =
      (data && typeof data === "object" && data.detail) || raw || `HTTP ${response.status}`;
    throw new Error(String(detail));
  }

  if (data === null) {
    throw new Error("Response is not valid JSON");
  }

  return data;
}

function postJson(path, body) {
  return fetchJson(apiPath(path), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
}

function fallbackNodeColor(id) {
  const palette = ["#f4a261", "#2a9d8f", "#e76f51", "#457b9d", "#8ecae6", "#f28482"];
  const key = String(id || "agent");
  let hash = 0;
  for (let i = 0; i < key.length; i += 1) {
    hash = (hash << 5) - hash + key.charCodeAt(i);
    hash |= 0;
  }
  return palette[Math.abs(hash) % palette.length];
}

function moodClass(moodLabel) {
  return `mood mood-${moodLabel || "neutral"}`;
}

function sourceLabel(event, agentById) {
  if (event.source_id && agentById.has(event.source_id)) {
    return agentById.get(event.source_id).name;
  }
  if (event.source_type === "world") {
    return "World";
  }
  return "Unknown";
}

export default function App() {
  const [agents, setAgents] = useState([]);
  const [relations, setRelations] = useState({ nodes: [], edges: [] });
  const [events, setEvents] = useState([]);
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
  const [speed, setSpeed] = useState(1);
  const [controlFeedback, setControlFeedback] = useState("");

  const feedRef = useRef(null);
  const graphWrapRef = useRef(null);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(true);
  const [graphSize, setGraphSize] = useState({ width: 600, height: 320 });

  const agentById = useMemo(() => {
    return new Map(agents.map((agent) => [agent.id, agent]));
  }, [agents]);

  const filteredEvents = useMemo(() => {
    if (filterAgentId === "all") {
      return events;
    }
    return events.filter((event) => event.source_id === filterAgentId);
  }, [events, filterAgentId]);

  const graphData = useMemo(() => {
    return {
      nodes: relations.nodes.map((node) => ({
        ...node,
        color: agentById.get(node.id)?.avatar || fallbackNodeColor(node.id),
      })),
      links: relations.edges.map((edge) => ({ ...edge })),
    };
  }, [relations, agentById]);

  const buildLocalInspectData = useCallback(
    (agentId) => {
      const agent = agentById.get(agentId);
      if (!agent) {
        return null;
      }
      const recentEvents = events.filter((event) => event.source_id === agentId).slice(-10);
      return {
        id: agent.id,
        name: agent.name,
        traits: "n/a (local fallback)",
        mood: agent.mood,
        mood_label: agent.mood_label,
        current_plan: agent.current_plan,
        key_memories: [{ text: "Inspector fallback from live stream.", score: null }],
        recent_events: recentEvents,
      };
    },
    [agentById, events]
  );

  const fetchInspect = useCallback(
    async (agentId) => {
      if (!agentId) {
        return;
      }
      setInspectAgentId(agentId);
      setInspectLoading(true);
      setInspectError("");
      try {
        const data = await fetchJson(apiPath(`/api/agents/${encodeURIComponent(agentId)}`), {
          method: "GET",
          headers: { Accept: "application/json" },
        });
        setInspectData(data);
        setInspectError("");
      } catch (error) {
        const fallback = buildLocalInspectData(agentId);
        if (fallback) {
          setInspectData(fallback);
          setInspectError("");
          setControlFeedback(`Inspector API issue, showing local fallback: ${String(error)}`);
        } else {
          setInspectData(null);
          setInspectError(`Failed to load inspector: ${String(error)}`);
        }
      } finally {
        setInspectLoading(false);
      }
    },
    [buildLocalInspectData]
  );

  useEffect(() => {
    async function loadInitialState() {
      try {
        const state = await fetchJson(apiPath("/api/state"), {
          method: "GET",
          headers: { Accept: "application/json" },
        });
        setAgents(Array.isArray(state.agents) ? state.agents : []);
        setRelations(state.relations || { nodes: [], edges: [] });
        setEvents(trimEvents(Array.isArray(state.events) ? state.events : []));
        if (typeof state.speed === "number") {
          setSpeed(state.speed);
        }
      } catch (error) {
        setControlFeedback(`Initial state load failed: ${String(error)}`);
      }
    }
    loadInitialState();
  }, []);

  useEffect(() => {
    if (agents.length === 0) {
      return;
    }
    if (!messageAgentId || !agents.some((agent) => agent.id === messageAgentId)) {
      setMessageAgentId(agents[0].id);
    }
  }, [agents, messageAgentId]);

  useEffect(() => {
    if (filterAgentId === "all") {
      return;
    }
    if (!agents.some((agent) => agent.id === filterAgentId)) {
      setFilterAgentId("all");
    }
  }, [agents, filterAgentId]);

  useEffect(() => {
    if (!autoScroll || !feedRef.current) {
      return;
    }
    feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [filteredEvents, autoScroll]);

  useEffect(() => {
    if (!graphWrapRef.current) {
      return;
    }
    const target = graphWrapRef.current;

    const updateSize = () => {
      const rect = target.getBoundingClientRect();
      setGraphSize({
        width: Math.max(240, Math.floor(rect.width)),
        height: Math.max(260, Math.floor(rect.height)),
      });
    };

    updateSize();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateSize);
      return () => window.removeEventListener("resize", updateSize);
    }

    const observer = new ResizeObserver(updateSize);
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    shouldReconnectRef.current = true;

    function connect() {
      setWsStatus("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsStatus("open");
      };

      ws.onmessage = (message) => {
        try {
          const parsed = JSON.parse(message.data);
          if (parsed.type === "agents_state" && Array.isArray(parsed.payload)) {
            setAgents(parsed.payload);
            return;
          }
          if (parsed.type === "event" && parsed.payload) {
            setEvents((prev) => appendEvent(prev, parsed.payload));
            return;
          }
          if (parsed.type === "relations" && parsed.payload) {
            setRelations(parsed.payload);
          }
        } catch {
          setControlFeedback("WS: received malformed message");
        }
      };

      ws.onerror = () => {
        setWsStatus("error");
      };

      ws.onclose = () => {
        if (!shouldReconnectRef.current) {
          return;
        }
        setWsStatus("reconnecting");
        reconnectTimerRef.current = setTimeout(connect, 1500);
      };
    }

    connect();

    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  async function submitWorldEvent(event) {
    event.preventDefault();
    const text = worldEventText.trim();
    if (!text) {
      return;
    }

    try {
      await postJson("/api/control/event", { text });
      setWorldEventText("");
      setControlFeedback("World event sent");
    } catch (error) {
      setControlFeedback(`Failed to send world event: ${String(error)}`);
    }
  }

  async function submitMessage(event) {
    event.preventDefault();
    const text = messageText.trim();
    if (!text || !messageAgentId) {
      return;
    }

    try {
      await postJson("/api/control/message", { agent_id: messageAgentId, text });
      setMessageText("");
      setControlFeedback("Message sent");
    } catch (error) {
      setControlFeedback(`Failed to send message: ${String(error)}`);
    }
  }

  async function submitSpeed(event) {
    event.preventDefault();
    try {
      const data = await postJson("/api/control/speed", { speed });
      if (typeof data.speed === "number") {
        setSpeed(data.speed);
      }
      setControlFeedback(`Speed updated to ${Number(speed).toFixed(1)}x`);
    } catch (error) {
      setControlFeedback(`Failed to update speed: ${String(error)}`);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Skebobia Dashboard</h1>
          <p>Realtime control + telemetry</p>
        </div>
        <div className={`ws-pill ws-${wsStatus}`}>WS: {wsStatus}</div>
      </header>

      <main className="dashboard-grid">
        <section className="panel feed-panel">
          <div className="panel-title-row">
            <h2>Event Feed</h2>
            <div className="inline-controls">
              <label>
                Agent:
                <select value={filterAgentId} onChange={(event) => setFilterAgentId(event.target.value)}>
                  <option value="all">All</option>
                  {agents.map((agent) => (
                    <option key={agent.id} value={agent.id}>
                      {agent.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={autoScroll}
                  onChange={(event) => setAutoScroll(event.target.checked)}
                />
                auto-scroll
              </label>
            </div>
          </div>

          <div className="event-list" ref={feedRef}>
            {filteredEvents.length === 0 && <div className="empty-state">No events yet</div>}
            {filteredEvents.map((event) => (
              <article key={event.id} className="event-item">
                <div className="event-meta">
                  <span>{event.ts || "-"}</span>
                  <span>{sourceLabel(event, agentById)}</span>
                  <span>{Array.isArray(event.tags) ? event.tags.join(", ") : ""}</span>
                </div>
                <p>{event.text}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="panel graph-panel">
          <div className="panel-title-row">
            <h2>Relations Graph</h2>
            <span className="muted">Click node to inspect</span>
          </div>
          <div className="graph-wrap" ref={graphWrapRef}>
            <ForceGraph2D
              graphData={graphData}
              width={graphSize.width}
              height={graphSize.height}
              linkSource="from"
              linkTarget="to"
              nodeLabel={(node) => node.name}
              linkLabel={(link) => `value: ${link.value}`}
              linkColor={(link) => (link.value >= 0 ? "rgba(24, 164, 113, 0.95)" : "rgba(214, 58, 58, 0.95)")}
              linkWidth={(link) => 2 + Math.abs(link.value || 0) / 18}
              nodeCanvasObject={(node, ctx, globalScale) => {
                const label = node.name || node.id;
                const size = 8;
                const fontSize = 12 / globalScale;
                ctx.beginPath();
                ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
                ctx.fillStyle = node.color || fallbackNodeColor(node.id);
                ctx.fill();
                ctx.strokeStyle = "#13213a";
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.font = `${fontSize}px Trebuchet MS`;
                ctx.fillStyle = "#13213a";
                ctx.fillText(label, node.x + size + 2, node.y + size / 2);
              }}
              onNodeClick={(node) => fetchInspect(node.id)}
            />
          </div>
        </section>

        <section className="panel side-panel">
          <div className="panel-block">
            <h2>Agent Cards</h2>
            <div className="agent-cards">
              {agents.map((agent) => (
                <article key={agent.id} className="agent-card">
                  <div className="agent-card-head">
                    <strong>{agent.name}</strong>
                    <span className={moodClass(agent.mood_label)}>{agent.mood_label}</span>
                  </div>
                  <p>{agent.current_plan || "-"}</p>
                  <button type="button" onClick={() => fetchInspect(agent.id)}>
                    Inspect
                  </button>
                </article>
              ))}
            </div>
          </div>

          <div className="panel-block">
            <h2>Control Panel</h2>

            <form onSubmit={submitWorldEvent} className="control-form">
              <label>World event</label>
              <textarea
                rows={3}
                value={worldEventText}
                onChange={(event) => setWorldEventText(event.target.value)}
                placeholder="Meteor shower started over the market..."
              />
              <button type="submit">Send Event</button>
            </form>

            <form onSubmit={submitMessage} className="control-form">
              <label>Message to agent</label>
              <select value={messageAgentId} onChange={(event) => setMessageAgentId(event.target.value)}>
                {agents.map((agent) => (
                  <option key={agent.id} value={agent.id}>
                    {agent.name}
                  </option>
                ))}
              </select>
              <textarea
                rows={2}
                value={messageText}
                onChange={(event) => setMessageText(event.target.value)}
                placeholder="You should patrol the square."
              />
              <button type="submit">Send Message</button>
            </form>

            <form onSubmit={submitSpeed} className="control-form">
              <label>Speed: {Number(speed).toFixed(1)}x</label>
              <input
                type="range"
                min="0.1"
                max="5"
                step="0.1"
                value={speed}
                onChange={(event) => setSpeed(Number(event.target.value))}
              />
              <button type="submit">Apply Speed</button>
            </form>

            {controlFeedback && <p className="control-feedback">{controlFeedback}</p>}
          </div>
        </section>
      </main>

      {inspectAgentId && (
        <aside className="inspect-drawer">
          <div className="inspect-header">
            <h3>Inspector: {inspectData?.name || inspectAgentId}</h3>
            <button
              type="button"
              onClick={() => {
                setInspectAgentId("");
                setInspectData(null);
                setInspectError("");
              }}
            >
              Close
            </button>
          </div>

          {inspectLoading && <p>Loading...</p>}
          {inspectError && <p>{inspectError}</p>}
          {inspectData && !inspectLoading && (
            <div className="inspect-content">
              <p>
                <strong>Traits:</strong> {inspectData.traits || "-"}
              </p>
              <p>
                <strong>Mood:</strong> {inspectData.mood_label} ({inspectData.mood})
              </p>
              <p>
                <strong>Plan:</strong> {inspectData.current_plan || "-"}
              </p>

              <h4>Key memories</h4>
              <ul>
                {(inspectData.key_memories || []).map((memory, index) => (
                  <li key={`mem-${index}`}>
                    {memory.text} {typeof memory.score === "number" ? `(${memory.score})` : ""}
                  </li>
                ))}
              </ul>

              <h4>Recent events</h4>
              <ul>
                {(inspectData.recent_events || []).map((event) => (
                  <li key={event.id}>
                    <span>{event.ts || "-"}</span> {event.text}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </aside>
      )}
    </div>
  );
}
