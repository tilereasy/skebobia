using UnityEngine;
using System.Collections.Generic;

public sealed class AgentRegistry : MonoBehaviour
{
    [SerializeField] private Transform agentsRoot;
    [SerializeField] private Vector3 agentScale = new Vector3(0.8f, 0.8f, 1f);
    [SerializeField] private bool removeMissingAgents = true;

    private readonly Dictionary<string, GameObject> agentObjects = new Dictionary<string, GameObject>();
    private readonly HashSet<string> seenIds = new HashSet<string>();
    private Sprite fallbackSprite;

    private void Awake()
    {
        if (agentsRoot == null)
        {
            GameObject root = new GameObject("Agents");
            root.transform.SetParent(transform, false);
            agentsRoot = root.transform;
        }

        fallbackSprite = CreateFallbackSprite();
    }

    public void CreateOrUpdate(IReadOnlyList<AgentState> agents)
    {
        if (agents == null)
        {
            return;
        }

        seenIds.Clear();
        for (int i = 0; i < agents.Count; i++)
        {
            AgentState state = agents[i];
            if (state == null || string.IsNullOrWhiteSpace(state.id))
            {
                continue;
            }

            seenIds.Add(state.id);

            if (!agentObjects.TryGetValue(state.id, out GameObject agentObject))
            {
                agentObject = CreateAgentObject(state.id);
                agentObjects[state.id] = agentObject;
            }

            UpdateAgentObject(agentObject, state);
        }

        if (!removeMissingAgents)
        {
            return;
        }

        List<string> missingIds = new List<string>();
        foreach (KeyValuePair<string, GameObject> entry in agentObjects)
        {
            if (!seenIds.Contains(entry.Key))
            {
                missingIds.Add(entry.Key);
                Destroy(entry.Value);
            }
        }

        for (int i = 0; i < missingIds.Count; i++)
        {
            agentObjects.Remove(missingIds[i]);
        }
    }

    private GameObject CreateAgentObject(string agentId)
    {
        GameObject go = new GameObject($"Agent_{agentId}");
        go.transform.SetParent(agentsRoot, false);
        go.transform.localScale = agentScale;

        SpriteRenderer renderer = go.AddComponent<SpriteRenderer>();
        renderer.sprite = fallbackSprite;
        renderer.sortingOrder = 10;
        return go;
    }

    private void UpdateAgentObject(GameObject agentObject, AgentState state)
    {
        Vector2 world2D = state.Position2D();
        agentObject.transform.position = new Vector3(world2D.x, world2D.y, 0f);

        string displayName = string.IsNullOrWhiteSpace(state.name) ? state.id : state.name;
        agentObject.name = $"Agent_{state.id}_{displayName}";

        if (!agentObject.TryGetComponent(out SpriteRenderer renderer))
        {
            return;
        }

        if (!string.IsNullOrWhiteSpace(state.avatar) && ColorUtility.TryParseHtmlString(state.avatar, out Color parsed))
        {
            renderer.color = parsed;
            return;
        }

        float t = Mathf.InverseLerp(-100f, 100f, state.mood);
        renderer.color = Color.Lerp(new Color(0.9f, 0.35f, 0.35f), new Color(0.35f, 0.85f, 0.4f), t);
    }

    private static Sprite CreateFallbackSprite()
    {
        Texture2D texture = Texture2D.whiteTexture;
        Rect rect = new Rect(0f, 0f, texture.width, texture.height);
        return Sprite.Create(texture, rect, new Vector2(0.5f, 0.5f), texture.width);
    }
}
