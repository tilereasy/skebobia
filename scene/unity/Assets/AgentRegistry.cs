using UnityEngine;
using System.Collections.Generic;
using TMPro;

public sealed class AgentRegistry : MonoBehaviour
{
    [SerializeField] private Transform agentsRoot;
    [SerializeField] private GameObject agentPrefab; 
    [SerializeField] private GameObject speechBubblePrefab;
    [SerializeField] private Vector3 agentScale = new Vector3(0.8f, 0.8f, 1f);
    [SerializeField, Min(0.1f)] private float moveSpeed = 5f;
    [SerializeField] private Vector3 nameOffset = new Vector3(0f, 0.9f, 0f);
    [SerializeField] private Vector3 speechBubbleOffset = new Vector3(0.8f, 0.6f, 0f);
    [SerializeField, Min(0.1f)] private float nameFontSize = 2.5f;
    [SerializeField] private Color nameColor = Color.white;
    [SerializeField] private bool removeMissingAgents = true;

    [Header("Emotion Icons")]
    [SerializeField] private Vector3 emotionOffset = Vector3.zero;
    [SerializeField] private int emotionSortingOrder = 15;
    [SerializeField, Min(0.01f)] private float emotionScale = 0.13f; // Новый масштаб по умолчанию

    private Dictionary<string, Sprite> emotionSpriteMap;
    private readonly Dictionary<string, GameObject> agentObjects = new Dictionary<string, GameObject>();
    private readonly Dictionary<string, AgentView> agentViews = new Dictionary<string, AgentView>();
    private readonly Dictionary<string, SpriteRenderer> agentRenderers = new Dictionary<string, SpriteRenderer>();
    private readonly Dictionary<string, TMP_Text> agentLabels = new Dictionary<string, TMP_Text>();
    private readonly Dictionary<string, SpriteRenderer> agentEmotionRenderers = new Dictionary<string, SpriteRenderer>();
    private readonly HashSet<string> seenIds = new HashSet<string>();

    private Sprite circleSprite;

    private void Awake()
    {
        if (agentsRoot == null)
        {
            GameObject root = new GameObject("Agents");
            root.transform.SetParent(transform, false);
            agentsRoot = root.transform;
        }

        circleSprite = CreateCircleSprite();
        LoadEmotionSprites();
    }

    private void LoadEmotionSprites()
    {
        emotionSpriteMap = new Dictionary<string, Sprite>();
        string[] emotionNames = { "angry", "sad", "neutral", "happy", "excited" };
        foreach (string name in emotionNames)
        {
            Sprite sprite = Resources.Load<Sprite>($"Emotions/{name}");
            if (sprite != null)
            {
                emotionSpriteMap[name] = sprite;
                Debug.Log($"Loaded emotion sprite: {name}");
            }
            else
            {
                Debug.LogWarning($"Emotion sprite 'Emotions/{name}' not found in Resources. Place it in Assets/Resources/Emotions/");
            }
        }
    }

    public void CreateOrUpdate(IReadOnlyList<AgentState> agents)
    {
        if (agents == null) return;

        seenIds.Clear();
        foreach (var state in agents)
        {
            if (state == null || string.IsNullOrWhiteSpace(state.id)) continue;
            seenIds.Add(state.id);
            Upsert(state);
        }

        if (!removeMissingAgents) return;

        List<string> missingIds = new List<string>();
        foreach (var kvp in agentObjects)
        {
            if (!seenIds.Contains(kvp.Key))
            {
                missingIds.Add(kvp.Key);
                Destroy(kvp.Value);
            }
        }

        foreach (string id in missingIds)
        {
            agentObjects.Remove(id);
            agentViews.Remove(id);
            agentRenderers.Remove(id);
            agentLabels.Remove(id);
            agentEmotionRenderers.Remove(id);
        }
    }

    public void Upsert(AgentState state)
    {
        if (state == null || string.IsNullOrWhiteSpace(state.id)) return;

        bool isNew = !agentObjects.TryGetValue(state.id, out GameObject agentObject);
        if (isNew)
        {
            agentObject = CreateAgentObject(state.id);
            agentObjects[state.id] = agentObject;
        }

        UpdateAgentObject(agentObject, state, snapPosition: isNew);
    }

    public void SetSpeechBubblePrefab(GameObject prefab)
    {
        speechBubblePrefab = prefab;
    }

    public TMP_FontAsset cyrillicFont; 

    public bool ShowBubble(string agentId, string text, float ttlSec)
    {
        if (string.IsNullOrWhiteSpace(agentId) || string.IsNullOrWhiteSpace(text)) return false;
        if (!agentObjects.TryGetValue(agentId, out GameObject agentObject) || agentObject == null) return false;

        AgentView agentView = EnsureAgentView(agentId, agentObject);
        agentView.SetSpeechBubblePrefab(speechBubblePrefab);
        agentView.SetSpeechBubbleOffset(speechBubbleOffset);
        agentView.ShowBubble(text, ttlSec);
        return true;
    }

    private GameObject CreateAgentObject(string agentId)
    {
        GameObject go = new GameObject($"Agent_{agentId}");
        go.transform.SetParent(agentsRoot, false);
        go.transform.localScale = agentScale;

        EnsureAgentView(agentId, go);
        EnsureRenderer(agentId, go);
        EnsureLabel(agentId, go);
        return go;
    }

    private void UpdateAgentObject(GameObject agentObject, AgentState state, bool snapPosition)
    {
        AgentView agentView = EnsureAgentView(state.id, agentObject);

        Vector2 world2D = state.Position2D();
        agentView.SetMoveSpeed(moveSpeed);
        agentView.SetSpeechBubblePrefab(speechBubblePrefab);
        agentView.SetSpeechBubbleOffset(speechBubbleOffset);
        agentView.SetTargetPosition(world2D, snapPosition);

        string displayName = string.IsNullOrWhiteSpace(state.name) ? state.id : state.name;
        agentObject.name = $"Agent_{state.id}_{displayName}";

        TMP_Text label = EnsureLabel(state.id, agentObject);
        label.text = displayName;
        label.color = nameColor;
        label.fontSize = nameFontSize;
        
        label.transform.localPosition = nameOffset;

        
        SpriteRenderer renderer = EnsureRenderer(state.id, agentObject);

        if (!string.IsNullOrWhiteSpace(state.avatar) && ColorUtility.TryParseHtmlString(state.avatar, out Color parsed))
        {
            renderer.color = parsed;
        }
        else
        {
            float t = Mathf.InverseLerp(-100f, 100f, state.mood);
            renderer.color = Color.Lerp(new Color(0.9f, 0.35f, 0.35f), new Color(0.35f, 0.85f, 0.4f), t);
        }

        SpriteRenderer emotionRenderer = EnsureEmotionRenderer(state.id, agentObject);
        string moodKey = state.mood_label?.ToLowerInvariant();
        if (!string.IsNullOrEmpty(moodKey) && emotionSpriteMap.TryGetValue(moodKey, out Sprite emotionSprite))
        {
            emotionRenderer.sprite = emotionSprite;
            emotionRenderer.enabled = true;
            emotionRenderer.color = Color.white;
        }
        else
        {
            emotionRenderer.enabled = false;
        }
    }

    private AgentView EnsureAgentView(string agentId, GameObject agentObject)
    {
        if (!agentViews.TryGetValue(agentId, out AgentView agentView) || agentView == null)
        {
            if (!agentObject.TryGetComponent(out agentView))
                agentView = agentObject.AddComponent<AgentView>();
            agentViews[agentId] = agentView;
        }
        return agentView;
    }

    private SpriteRenderer EnsureRenderer(string agentId, GameObject agentObject)
    {
        if (!agentRenderers.TryGetValue(agentId, out SpriteRenderer renderer) || renderer == null)
        {
            renderer = agentObject.GetComponentInChildren<SpriteRenderer>();
            if (renderer == null)
            {
                GameObject body = new GameObject("Body");
                body.transform.SetParent(agentObject.transform, false);
                body.transform.localPosition = Vector3.zero;
                renderer = body.AddComponent<SpriteRenderer>();
            }
            agentRenderers[agentId] = renderer;
        }

        renderer.sprite = circleSprite;
        renderer.sortingOrder = 10;
        renderer.transform.localPosition = Vector3.zero;
        renderer.transform.localRotation = Quaternion.identity;
        return renderer;
    }

    private SpriteRenderer EnsureEmotionRenderer(string agentId, GameObject agentObject)
    {
        if (!agentEmotionRenderers.TryGetValue(agentId, out SpriteRenderer emotionRenderer) || emotionRenderer == null)
        {
            Transform existing = agentObject.transform.Find("Emotion");
            if (existing != null)
            {
                emotionRenderer = existing.GetComponent<SpriteRenderer>();
            }
            else
            {
                GameObject emotionObj = new GameObject("Emotion");
                emotionObj.transform.SetParent(agentObject.transform, false);
                emotionRenderer = emotionObj.AddComponent<SpriteRenderer>();
            }

            emotionRenderer.sortingOrder = emotionSortingOrder;
            emotionRenderer.transform.localPosition = emotionOffset;
            emotionRenderer.transform.localScale = Vector3.one * emotionScale;
            agentEmotionRenderers[agentId] = emotionRenderer;
        }
        return emotionRenderer;
    }

    private TMP_Text EnsureLabel(string agentId, GameObject agentObject)
    {
        if (!agentLabels.TryGetValue(agentId, out TMP_Text label) || label == null)
        {
            label = agentObject.GetComponentInChildren<TMP_Text>(true);
            if (label == null)
            {
                GameObject labelObject = new GameObject("NameLabel");
                labelObject.transform.SetParent(agentObject.transform, false);
                label = labelObject.AddComponent<TextMeshPro>();
            }

            if (cyrillicFont != null)
                label.font = cyrillicFont;

            label.alignment = TextAlignmentOptions.Center;
            label.enableWordWrapping = false;
            label.overflowMode = TextOverflowModes.Overflow;
            label.color = nameColor;
            label.fontSize = nameFontSize;
            label.transform.localPosition = nameOffset;
            label.transform.localRotation = Quaternion.identity;
            label.transform.localScale = Vector3.one;

            MeshRenderer textRenderer = label.GetComponent<MeshRenderer>();
            if (textRenderer != null) textRenderer.sortingOrder = 20;

            agentLabels[agentId] = label;
        }
        return label;
    }

    private static Sprite CreateCircleSprite()
    {
        int size = 64;
        float radius = (size - 1) * 0.5f;
        Vector2 center = new Vector2(radius, radius);

        Texture2D texture = new Texture2D(size, size, TextureFormat.RGBA32, false);
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.filterMode = FilterMode.Bilinear;

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float distance = Vector2.Distance(new Vector2(x, y), center);
                float alpha = Mathf.Clamp01(radius + 1f - distance);
                texture.SetPixel(x, y, new Color(1f, 1f, 1f, alpha));
            }
        }
        texture.Apply();

        return Sprite.Create(texture, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f), size);
    }
}