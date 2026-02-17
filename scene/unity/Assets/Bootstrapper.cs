using UnityEngine;

public sealed class Bootstrapper : MonoBehaviour
{
    [SerializeField] private GameObject speechBubblePrefab;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void CreateBootstrapperIfMissing()
    {
        if (FindAnyObjectByType<StateLoader>() != null &&
            FindAnyObjectByType<NetConfig>() != null &&
            FindAnyObjectByType<AgentRegistry>() != null &&
            FindAnyObjectByType<WsClient>() != null &&
            FindAnyObjectByType<StreamRouter>() != null)
        {
            return;
        }

        if (FindAnyObjectByType<Bootstrapper>() != null)
        {
            return;
        }

        GameObject go = new GameObject("SceneBootstrapper");
        go.AddComponent<Bootstrapper>();
    }

    private void Awake()
    {
        if (GetComponent<NetConfig>() == null)
        {
            gameObject.AddComponent<NetConfig>();
        }

        AgentRegistry agentRegistry = GetComponent<AgentRegistry>();
        if (agentRegistry == null)
        {
            agentRegistry = gameObject.AddComponent<AgentRegistry>();
        }
        agentRegistry.SetSpeechBubblePrefab(speechBubblePrefab);

        if (GetComponent<StateLoader>() == null)
        {
            gameObject.AddComponent<StateLoader>();
        }

        if (GetComponent<WsClient>() == null)
        {
            gameObject.AddComponent<WsClient>();
        }

        if (GetComponent<StreamRouter>() == null)
        {
            gameObject.AddComponent<StreamRouter>();
        }
    }
}
