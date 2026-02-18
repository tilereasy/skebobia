using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public sealed class StateLoader : MonoBehaviour
{
    [SerializeField] private NetConfig netConfig;
    [SerializeField] private AgentRegistry agentRegistry;
    [SerializeField] private bool loadOnStart = true;
    [SerializeField, Min(1)] private int timeoutSec = 10;

    private readonly List<AgentState> agents = new List<AgentState>();
    private bool isLoading;
    private bool hasLoadedOnce;

    public IReadOnlyList<AgentState> Agents => agents;
    public bool HasLoadedState => hasLoadedOnce;

    public event Action<IReadOnlyList<AgentState>> StateLoaded;
    public event Action<string> StateLoadFailed;

    private void Awake()
    {
        if (netConfig == null)
        {
            netConfig = FindAnyObjectByType<NetConfig>();
        }

        if (agentRegistry == null)
        {
            agentRegistry = FindAnyObjectByType<AgentRegistry>();
        }
    }

    private void Start()
    {
        if (loadOnStart)
        {
            LoadStateIfNeeded();
        }
    }

    public void LoadState()
    {
        if (isLoading)
        {
            return;
        }

        StartCoroutine(LoadStateRoutine());
    }

    public void LoadStateIfNeeded()
    {
        if (hasLoadedOnce || isLoading)
        {
            return;
        }

        LoadState();
    }

    private IEnumerator LoadStateRoutine()
    {
        isLoading = true;
        try
        {
            if (netConfig == null)
            {
                NotifyFailed("StateLoader: NetConfig is missing.");
                yield break;
            }

            string url = netConfig.StateUrl;
            using UnityWebRequest request = UnityWebRequest.Get(url);
            request.timeout = timeoutSec;
            request.SetRequestHeader("Accept", "application/json");

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                NotifyFailed($"StateLoader: GET {url} failed: {request.error}");
                yield break;
            }

            string json = request.downloadHandler.text;
            StatePayload payload;
            try
            {
                payload = JsonUtility.FromJson<StatePayload>(json);
            }
            catch (Exception ex)
            {
                NotifyFailed($"StateLoader: invalid JSON from {url}: {ex.Message}");
                yield break;
            }

            agents.Clear();
            if (payload?.agents != null)
            {
                agents.AddRange(payload.agents);
            }

            if (agentRegistry != null)
            {
                agentRegistry.CreateOrUpdate(agents);
            }

            hasLoadedOnce = true;
            Debug.Log($"Loaded state: {agents.Count} agents");
            StateLoaded?.Invoke(agents);
        }
        finally
        {
            isLoading = false;
        }
    }

    private void NotifyFailed(string error)
    {
        Debug.LogError(error);
        StateLoadFailed?.Invoke(error);
    }

    [Serializable]
    private sealed class StatePayload
    {
        public AgentState[] agents;
    }
}

[Serializable]
public sealed class AgentState
{
    public string id;
    public string name;
    public string avatar;
    public int mood;
    public string mood_label;
    public string current_plan;
    public AgentPos pos;
    public AgentPos look_at;
    public string last_action;
    public string last_say;
    public string target_id;

    public Vector2 Position2D()
    {
        float x = pos != null ? pos.x : 0f;
        float z = pos != null ? pos.z : 0f;
        return new Vector2(x, z);
    }
}

[Serializable]
public sealed class AgentPos
{
    public float x;
    public float y;
    public float z;
}
