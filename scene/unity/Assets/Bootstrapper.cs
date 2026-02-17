using UnityEngine;

public sealed class Bootstrapper : MonoBehaviour
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void CreateBootstrapperIfMissing()
    {
        if (FindAnyObjectByType<StateLoader>() != null &&
            FindAnyObjectByType<NetConfig>() != null &&
            FindAnyObjectByType<AgentRegistry>() != null)
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

        if (GetComponent<AgentRegistry>() == null)
        {
            gameObject.AddComponent<AgentRegistry>();
        }

        if (GetComponent<StateLoader>() == null)
        {
            gameObject.AddComponent<StateLoader>();
        }
    }
}
