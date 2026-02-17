using System;
using UnityEngine;

public sealed class StreamRouter : MonoBehaviour
{
    [SerializeField] private WsClient wsClient;
    [SerializeField] private AgentRegistry agentRegistry;
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
        subscribed = true;
    }

    private void HandleMessage(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return;
        }

        MessageHeader header = JsonUtility.FromJson<MessageHeader>(json);
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

        AgentsStateMessage message = JsonUtility.FromJson<AgentsStateMessage>(json);
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

    private void HandleEvent(string json)
    {
        EventMessage message = JsonUtility.FromJson<EventMessage>(json);
        string text = message != null && message.payload != null && !string.IsNullOrWhiteSpace(message.payload.text)
            ? message.payload.text
            : json;
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
