using System;
using System.Collections;
using System.Text;
using System.Threading.Tasks;
using NativeWebSocket;
using UnityEngine;

public sealed class WsClient : MonoBehaviour
{
    [SerializeField] private NetConfig netConfig;
    [SerializeField] private bool connectOnStart = true;
    [SerializeField] private bool reconnectOnClose = true;
    [SerializeField, Min(1f)] private float reconnectDelaySec = 2f;

    private WebSocket socket;
    private bool isQuitting;
    private Coroutine reconnectCoroutine;

    public event Action<string> OnMessage;
    public event Action OnConnected;
    public event Action<string> OnError;
    public event Action<string> OnClosed;

    private void Awake()
    {
        if (netConfig == null)
        {
            netConfig = FindAnyObjectByType<NetConfig>();
        }
    }

    private async void Start()
    {
        if (connectOnStart)
        {
            await Connect();
        }
    }

    private void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (socket != null)
        {
            socket.DispatchMessageQueue();
        }
#endif
    }

    private async void OnDestroy()
    {
        await Shutdown();
    }

    private async void OnApplicationQuit()
    {
        await Shutdown();
    }

    public async Task Connect()
    {
        if (socket != null && (socket.State == WebSocketState.Open || socket.State == WebSocketState.Connecting))
        {
            return;
        }

        StopReconnectCoroutine();
        await CloseSocket();

        string streamUrl = netConfig != null ? netConfig.StreamWsUrl : "ws://localhost/ws/stream";
        socket = new WebSocket(streamUrl);

        socket.OnOpen += () =>
        {
            Debug.Log($"WS connected: {streamUrl}");
            OnConnected?.Invoke();
        };

        socket.OnError += error =>
        {
            Debug.LogError($"WS error: {error}");
            OnError?.Invoke(error);
        };

        socket.OnClose += code =>
        {
            string codeText = code.ToString();
            Debug.LogWarning($"WS closed: {codeText}");
            OnClosed?.Invoke(codeText);
            if (reconnectOnClose && !isQuitting)
            {
                ScheduleReconnect();
            }
        };

        socket.OnMessage += bytes =>
        {
            string json = Encoding.UTF8.GetString(bytes);
            OnMessage?.Invoke(json);
        };

        try
        {
            await socket.Connect();
        }
        catch (Exception ex)
        {
            Debug.LogError($"WS connect failed: {ex.Message}");
            OnError?.Invoke(ex.Message);
            if (reconnectOnClose && !isQuitting)
            {
                ScheduleReconnect();
            }
        }
    }

    private void ScheduleReconnect()
    {
        if (reconnectCoroutine != null)
        {
            return;
        }

        reconnectCoroutine = StartCoroutine(ReconnectRoutine());
    }

    private IEnumerator ReconnectRoutine()
    {
        yield return new WaitForSeconds(reconnectDelaySec);
        reconnectCoroutine = null;
        if (!isQuitting)
        {
            _ = Connect();
        }
    }

    private void StopReconnectCoroutine()
    {
        if (reconnectCoroutine == null)
        {
            return;
        }

        StopCoroutine(reconnectCoroutine);
        reconnectCoroutine = null;
    }

    private async Task Shutdown()
    {
        if (isQuitting)
        {
            return;
        }

        isQuitting = true;
        StopReconnectCoroutine();
        await CloseSocket();
    }

    private async Task CloseSocket()
    {
        if (socket == null)
        {
            return;
        }

        WebSocket current = socket;
        socket = null;

        if (current.State == WebSocketState.Open || current.State == WebSocketState.Connecting)
        {
            try
            {
                await current.Close();
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"WS close failed: {ex.Message}");
            }
        }
    }
}
