import { useCallback, useEffect, useMemo, useRef, useState, memo } from "react";
import ForceGraph2D from "react-force-graph-2d";

// Конфигурация API
const API_PROTOCOL = window.location.protocol === "https:" ? "https" : "http";
const API_HOST = window.location.hostname || "localhost";
const API_BASE = import.meta.env.VITE_API_BASE || `${API_PROTOCOL}://${API_HOST}:8000`;
const WS_PROTOCOL = window.location.protocol === "https:" ? "wss" : "ws";
const WS_HOST = window.location.hostname || "localhost";
const WS_URL = import.meta.env.VITE_WS_URL || `${WS_PROTOCOL}://${WS_HOST}:8000/ws/stream`;

// Константы
const MAX_EVENTS = 300;
const RECONNECT_DELAY = 1500;
const MIN_GRAPH_WIDTH = 240;
const MIN_GRAPH_HEIGHT = 200; // Уменьшено с 260
const RECENT_EVENTS_LIMIT = 10;
const FEEDBACK_TIMEOUT = 5000;
const GRAPH_NODE_SIZE = 8;
const GRAPH_NODE_FONT_SIZE = 12;

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
    } catch (error) {
      console.error("JSON parse error:", error);
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
    <p>{agent.current_plan || "-"}</p>
    <button 
      type="button" 
      onClick={() => onInspect(agent.id)}
      aria-label={`Inspect ${agent.name}`}
    >
      Inspect
    </button>
  </article>
));

AgentCard.displayName = "AgentCard";

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
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  
  // Новые состояния для фильтров событий
  const [eventFilters, setEventFilters] = useState({
    dialogue: true,
    world: true,
    system: true,
    other: true,
  });
  
  // Состояние для сворачивания панелей
  const [collapsedPanels, setCollapsedPanels] = useState({
    graph: false,
    controls: false,
  });

  const feedRef = useRef(null);
  const graphWrapRef = useRef(null);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(true);
  const [graphSize, setGraphSize] = useState({ 
    width: MIN_GRAPH_WIDTH, 
    height: MIN_GRAPH_HEIGHT 
  });

  const agentById = useMemo(() => {
    return new Map(agents.map((agent) => [agent.id, agent]));
  }, [agents]);

  const filteredEvents = useMemo(() => {
    let filtered = events;
    
    // Фильтр по агенту
    if (filterAgentId !== "all") {
      filtered = filtered.filter((event) => event.source_id === filterAgentId);
    }
    
    // Фильтр по типу события
    filtered = filtered.filter((event) => {
      const type = getEventType(event);
      return eventFilters[type] !== false;
    });
    
    return filtered;
  }, [events, filterAgentId, eventFilters]);

  const graphData = useMemo(() => {
    return {
      nodes: relations.nodes
        .filter(node => agentById.has(node.id))
        .map((node) => ({
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
      const recentEvents = events
        .filter((event) => event.source_id === agentId)
        .slice(-RECENT_EVENTS_LIMIT);
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
        console.error("Inspector API error:", error);
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

  const toggleEventFilter = useCallback((filterType) => {
    setEventFilters(prev => ({
      ...prev,
      [filterType]: !prev[filterType]
    }));
  }, []);

  const togglePanel = useCallback((panelName) => {
    setCollapsedPanels(prev => ({
      ...prev,
      [panelName]: !prev[panelName]
    }));
  }, []);

  useEffect(() => {
    if (controlFeedback) {
      const timer = setTimeout(() => setControlFeedback(""), FEEDBACK_TIMEOUT);
      return () => clearTimeout(timer);
    }
  }, [controlFeedback]);

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
        console.error("Initial state load failed:", error);
        setControlFeedback(`Initial state load failed: ${String(error)}`);
      } finally {
        setIsInitialLoading(false);
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
        width: Math.max(MIN_GRAPH_WIDTH, Math.floor(rect.width)),
        height: Math.max(MIN_GRAPH_HEIGHT, Math.floor(rect.height)),
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
        console.log("WebSocket connected");
        setWsStatus("open");
      };

      ws.onmessage = (message) => {
        try {
          const parsed = JSON.parse(message.data);
          
          if (!parsed.type || parsed.payload === undefined) {
            console.warn("Invalid WebSocket message format:", parsed);
            return;
          }

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
            return;
          }
          if (parsed.type === "ping") {
            return;
          }
        } catch (error) {
          console.error("WebSocket message parse error:", error);
          setControlFeedback("WS: received malformed message");
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setWsStatus("error");
      };

      ws.onclose = () => {
        console.log("WebSocket closed");
        if (!shouldReconnectRef.current) {
          return;
        }
        setWsStatus("reconnecting");
        reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY);
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
      console.error("Failed to send world event:", error);
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
      console.error("Failed to send message:", error);
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
      console.error("Failed to update speed:", error);
      setControlFeedback(`Failed to update speed: ${String(error)}`);
    }
  }

  if (isInitialLoading) {
    return (
      <div className="app-shell">
        <header className="topbar">
          <div>
            <h1>Skebobia Dashboard</h1>
            <p>Loading...</p>
          </div>
        </header>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Skebobia Dashboard</h1>
          <p>Realtime control + telemetry</p>
        </div>
        <div className="topbar-actions">
          <a href="/scene/" target="_blank" rel="noopener noreferrer" className="scene-link">
            🎮 Open Unity Scene
          </a>
          <div className={`ws-pill ws-${wsStatus}`} aria-label={`WebSocket status: ${wsStatus}`}>
            WS: {wsStatus}
          </div>
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="panel feed-panel">
          <div className="panel-title-row">
            <h2>Event Feed</h2>
            <div className="inline-controls">
              <label>
                Agent:
                <select 
                  value={filterAgentId} 
                  onChange={(event) => setFilterAgentId(event.target.value)}
                  aria-label="Filter events by agent"
                >
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
                  aria-label="Enable auto-scroll"
                />
                auto-scroll
              </label>
            </div>
          </div>

          <div className="event-filters">
            <span className="filter-label">Show:</span>
            <label className="filter-checkbox">
              <input
                type="checkbox"
                checked={eventFilters.dialogue}
                onChange={() => toggleEventFilter("dialogue")}
              />
              Dialogues
            </label>
            <label className="filter-checkbox">
              <input
                type="checkbox"
                checked={eventFilters.world}
                onChange={() => toggleEventFilter("world")}
              />
              World
            </label>
            <label className="filter-checkbox">
              <input
                type="checkbox"
                checked={eventFilters.system}
                onChange={() => toggleEventFilter("system")}
              />
              System
            </label>
          </div>

          <div className="event-list" ref={feedRef} role="log" aria-live="polite">
            {filteredEvents.length === 0 && <div className="empty-state">No events match filters</div>}
            {filteredEvents.map((event) => {
              const eventType = getEventType(event);
              return (
                <article key={event.id} className={`event-item event-type-${eventType}`}>
                  <div className="event-meta">
                    <span className="event-time">{event.ts || "-"}</span>
                    <span className="event-source">{sourceLabel(event, agentById)}</span>
                    <span className="event-tags">{Array.isArray(event.tags) ? event.tags.join(", ") : ""}</span>
                  </div>
                  <p>{event.text}</p>
                </article>
              );
            })}
          </div>
        </section>

        <section className="panel graph-panel-compact">
          <div className="panel-title-row">
            <h2>Relations Graph</h2>
            <div>
              <span className="muted">Click node to inspect</span>
              <button 
                type="button" 
                className="collapse-btn"
                onClick={() => togglePanel("graph")}
                aria-label={collapsedPanels.graph ? "Expand graph" : "Collapse graph"}
              >
                {collapsedPanels.graph ? "▼" : "▲"}
              </button>
            </div>
          </div>
          {!collapsedPanels.graph && (
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
                  const fontSize = GRAPH_NODE_FONT_SIZE / globalScale;
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, GRAPH_NODE_SIZE, 0, 2 * Math.PI, false);
                  ctx.fillStyle = node.color || fallbackNodeColor(node.id);
                  ctx.fill();
                  ctx.strokeStyle = "#13213a";
                  ctx.lineWidth = 1;
                  ctx.stroke();
                  ctx.font = `${fontSize}px Trebuchet MS`;
                  ctx.fillStyle = "#13213a";
                  ctx.fillText(label, node.x + GRAPH_NODE_SIZE + 2, node.y + GRAPH_NODE_SIZE / 2);
                }}
                onNodeClick={(node) => fetchInspect(node.id)}
              />
            </div>
          )}
        </section>

        <section className="panel side-panel">
          <div className="panel-block">
            <h2>Agent Cards</h2>
            <div className="agent-cards">
              {agents.map((agent) => (
                <AgentCard 
                  key={agent.id} 
                  agent={agent} 
                  onInspect={fetchInspect}
                />
              ))}
            </div>
          </div>

          <div className="panel-block">
            <div className="panel-title-row">
              <h2>Control Panel</h2>
              <button 
                type="button" 
                className="collapse-btn"
                onClick={() => togglePanel("controls")}
                aria-label={collapsedPanels.controls ? "Expand controls" : "Collapse controls"}
              >
                {collapsedPanels.controls ? "▼" : "▲"}
              </button>
            </div>

            {!collapsedPanels.controls && (
              <>
                <form onSubmit={submitWorldEvent} className="control-form">
                  <label htmlFor="world-event-text">World event</label>
                  <textarea
                    id="world-event-text"
                    rows={3}
                    value={worldEventText}
                    onChange={(event) => setWorldEventText(event.target.value)}
                    placeholder="Meteor shower started over the market..."
                    aria-label="World event text"
                  />
                  <button type="submit">Send Event</button>
                </form>

                <form onSubmit={submitMessage} className="control-form">
                  <label htmlFor="message-agent">Message to agent</label>
                  <select 
                    id="message-agent"
                    value={messageAgentId} 
                    onChange={(event) => setMessageAgentId(event.target.value)}
                    aria-label="Select agent to message"
                  >
                    {agents.map((agent) => (
                      <option key={agent.id} value={agent.id}>
                        {agent.name}
                      </option>
                    ))}
                  </select>
                  <textarea
                    id="message-text"
                    rows={2}
                    value={messageText}
                    onChange={(event) => setMessageText(event.target.value)}
                    placeholder="You should patrol the square."
                    aria-label="Message text"
                  />
                  <button type="submit">Send Message</button>
                </form>

                <form onSubmit={submitSpeed} className="control-form">
                  <label htmlFor="speed-slider">Speed: {Number(speed).toFixed(1)}x</label>
                  <input
                    id="speed-slider"
                    type="range"
                    min="0.1"
                    max="5"
                    step="0.1"
                    value={speed}
                    onChange={(event) => setSpeed(Number(event.target.value))}
                    aria-label="Simulation speed"
                  />
                  <button type="submit">Apply Speed</button>
                </form>
              </>
            )}

            {controlFeedback && (
              <p className="control-feedback" role="status" aria-live="polite">
                {controlFeedback}
              </p>
            )}
          </div>
        </section>
      </main>

      {inspectAgentId && (
        <aside className="inspect-drawer" role="dialog" aria-label="Agent inspector">
          <div className="inspect-header">
            <h3>Inspector: {inspectData?.name || inspectAgentId}</h3>
            <button
              type="button"
              onClick={() => {
                setInspectAgentId("");
                setInspectData(null);
                setInspectError("");
              }}
              aria-label="Close inspector"
            >
              Close
            </button>
          </div>

          {inspectLoading && <p>Loading...</p>}
          {inspectError && <p className="error-message">{inspectError}</p>}
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