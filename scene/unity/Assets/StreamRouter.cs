using System;
using UnityEngine;

public sealed class StreamRouter : MonoBehaviour
{
    private const int MaxIncomingJsonChars = 120_000;
    private const int MaxBubbleChars = 280;

    [SerializeField] private WsClient wsClient;
    [SerializeField] private AgentRegistry agentRegistry;
    [SerializeField] private StateLoader stateLoader;
    private bool subscribed;

    private void Awake()
    {
        if (wsClient == null)
        {
            wsClient = FindAnyObjectByType<WsClient>();
        }

        if (agentRegistry == null)
        {
            agentRegistry = FindAnyObjectByType<AgentRegistry>();
        }

        if (stateLoader == null)
        {
            stateLoader = FindAnyObjectByType<StateLoader>();
        }
    }

    private void OnEnable()
    {
        TrySubscribe();
    }

    private void OnDisable()
    {
        if (subscribed && wsClient != null)
        {
            wsClient.OnMessage -= HandleMessage;
            wsClient.OnConnected -= HandleConnected;
            subscribed = false;
        }
    }

    private void Update()
    {
        if (!subscribed)
        {
            TrySubscribe();
        }
    }

    private void TrySubscribe()
    {
        if (subscribed)
        {
            return;
        }

        if (wsClient == null)
        {
            wsClient = FindAnyObjectByType<WsClient>();
        }

        if (wsClient == null)
        {
            return;
        }

        wsClient.OnMessage += HandleMessage;
        wsClient.OnConnected += HandleConnected;
        subscribed = true;

        ReplayLastAgentsStateIfAvailable();
    }

    private void HandleConnected()
    {
        if (stateLoader == null)
        {
            stateLoader = FindAnyObjectByType<StateLoader>();
        }

        if (stateLoader != null)
        {
            stateLoader.LoadStateIfNeeded();
        }

        ReplayLastAgentsStateIfAvailable();
    }

    private void HandleMessage(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return;
        }

        if (json.Length > MaxIncomingJsonChars)
        {
            Debug.LogWarning($"WS message dropped: too large ({json.Length} chars)");
            return;
        }

        MessageHeader header;
        try
        {
            header = JsonUtility.FromJson<MessageHeader>(json);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"WS header parse failed: {ex.Message}");
            return;
        }
        if (header == null || string.IsNullOrWhiteSpace(header.type))
        {
            Debug.LogWarning($"WS message without type: {json}");
            return;
        }

        switch (header.type)
        {
            case "agents_state":
                HandleAgentsState(json);
                break;
            case "event":
                HandleEvent(json);
                break;
            case "relations":
                break;
            default:
                Debug.Log($"WS unknown type: {header.type}");
                break;
        }
    }

    private void HandleAgentsState(string json)
    {
        if (agentRegistry == null)
        {
            agentRegistry = FindAnyObjectByType<AgentRegistry>();
            if (agentRegistry == null)
            {
                Debug.LogError("StreamRouter: AgentRegistry is missing.");
                return;
            }
        }

        AgentsStateMessage message;
        try
        {
            message = JsonUtility.FromJson<AgentsStateMessage>(json);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"agents_state parse failed: {ex.Message}");
            return;
        }
        if (message == null || message.payload == null)
        {
            return;
        }

        int updated = 0;
        for (int i = 0; i < message.payload.Length; i++)
        {
            AgentState state = message.payload[i];
            if (state == null || string.IsNullOrWhiteSpace(state.id))
            {
                continue;
            }

            agentRegistry.Upsert(state);
            updated += 1;
        }

        Debug.Log($"WS agents_state: {updated} agents");
    }

    private void ReplayLastAgentsStateIfAvailable()
    {
        if (wsClient == null)
        {
            return;
        }

        if (!wsClient.TryGetLastAgentsState(out string json))
        {
            return;
        }

        HandleAgentsState(json);
    }

    private void HandleEvent(string json)
    {
        EventMessage message;
        try
        {
            message = JsonUtility.FromJson<EventMessage>(json);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"event parse failed: {ex.Message}");
            return;
        }
        string text = message != null && message.payload != null && !string.IsNullOrWhiteSpace(message.payload.text)
            ? message.payload.text
            : json;
        if (text.Length > MaxBubbleChars)
        {
            text = text.Substring(0, MaxBubbleChars);
        }
        Debug.Log($"WS event: {text}");

        if (message == null || message.payload == null)
        {
            return;
        }

        EventPayload payload = message.payload;
        bool sourceIsAgent = string.Equals(payload.source_type, "agent", StringComparison.OrdinalIgnoreCase);
        if (!sourceIsAgent || string.IsNullOrWhiteSpace(payload.source_id))
        {
            return;
        }

        if (ContainsTag(payload.tags, "memory"))
        {
            return;
        }

        bool hasDialogueTag = ContainsTag(payload.tags, "dialogue");
        bool looksLikeDialogue = LooksLikeDialogue(text);
        if (!hasDialogueTag && !looksLikeDialogue)
        {
            return;
        }

        if (agentRegistry == null)
        {
            agentRegistry = FindAnyObjectByType<AgentRegistry>();
            if (agentRegistry == null)
            {
                Debug.LogError("StreamRouter: AgentRegistry is missing.");
                return;
            }
        }

        if (!agentRegistry.ShowBubble(payload.source_id, text, 3f))
        {
            Debug.LogWarning($"StreamRouter: agent not found for event source_id={payload.source_id}");
        }
    }

    private static bool ContainsTag(string[] tags, string tag)
    {
        if (tags == null || tags.Length == 0 || string.IsNullOrWhiteSpace(tag))
        {
            return false;
        }

        for (int i = 0; i < tags.Length; i++)
        {
            if (string.Equals(tags[i], tag, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static bool LooksLikeDialogue(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        string normalized = text.Trim();
        if (normalized.Contains(":"))
        {
            return true;
        }

        if (normalized.Contains("\"") || normalized.Contains("«") || normalized.Contains("»"))
        {
            return true;
        }

        return normalized.EndsWith("?") || normalized.EndsWith("!") || normalized.EndsWith("...");
    }

    [Serializable]
    private sealed class MessageHeader
    {
        public string type;
    }

    [Serializable]
    private sealed class AgentsStateMessage
    {
        public string type;
        public AgentState[] payload;
    }

    [Serializable]
    private sealed class EventMessage
    {
        public string type;
        public EventPayload payload;
    }

    [Serializable]
    private sealed class EventPayload
    {
        public string id;
        public string text;
        public string source_type;
        public string source_id;
        public string[] tags;
    }
}
