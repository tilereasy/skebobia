using System.Collections;
using TMPro;
using UnityEngine;

public sealed class AgentView : MonoBehaviour
{
    [SerializeField, Min(0.1f)] private float moveSpeed = 5f;
    [SerializeField] private GameObject speechBubblePrefab;
    [SerializeField] private Vector3 speechBubbleOffset = new Vector3(0.2f, 1.6f, 0f);

    private Vector3 targetPos;
    private bool hasTargetPos;
    private GameObject bubbleInstance;
    private TMP_Text bubbleText;
    private Coroutine hideBubbleCoroutine;

    public void SetMoveSpeed(float unitsPerSecond)
    {
        if (unitsPerSecond > 0f)
        {
            moveSpeed = unitsPerSecond;
        }
    }

    public void SetSpeechBubblePrefab(GameObject prefab)
    {
        speechBubblePrefab = prefab;
    }

    public void SetSpeechBubbleOffset(Vector3 offset)
    {
        speechBubbleOffset = offset;
        if (bubbleInstance != null)
        {
            bubbleInstance.transform.localPosition = speechBubbleOffset;
        }
    }

    public void ShowBubble(string text, float ttlSec)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        if (!EnsureBubble())
        {
            return;
        }

        bubbleText.text = text.Trim();
        bubbleInstance.SetActive(true);

        if (hideBubbleCoroutine != null)
        {
            StopCoroutine(hideBubbleCoroutine);
        }

        float ttl = ttlSec > 0f ? ttlSec : 3f;
        hideBubbleCoroutine = StartCoroutine(HideBubbleAfter(ttl));
    }

    public void SetTargetPosition(Vector2 pos2D, bool snap)
    {
        targetPos = new Vector3(pos2D.x, pos2D.y, 0f);
        if (snap || !hasTargetPos)
        {
            transform.position = targetPos;
        }

        hasTargetPos = true;
    }

    private void Update()
    {
        if (!hasTargetPos)
        {
            return;
        }

        transform.position = Vector3.MoveTowards(transform.position, targetPos, moveSpeed * Time.deltaTime);
    }

    private bool EnsureBubble()
    {
        if (bubbleInstance == null)
        {
            if (speechBubblePrefab == null)
            {
                Debug.LogWarning("AgentView: speechBubblePrefab is not assigned.");
                return false;
            }

            bubbleInstance = Instantiate(speechBubblePrefab, transform);
            bubbleInstance.name = "SpeechBubble";
            bubbleInstance.transform.localPosition = speechBubbleOffset;
            bubbleInstance.transform.localRotation = Quaternion.identity;
            bubbleInstance.transform.localScale = Vector3.one;
            bubbleInstance.SetActive(false);
        }

        if (bubbleText == null)
        {
            bubbleText = bubbleInstance.GetComponentInChildren<TMP_Text>(true);
        }

        if (bubbleText == null)
        {
            Debug.LogWarning("AgentView: speech bubble prefab has no TMP_Text.");
            return false;
        }

        return true;
    }

    private IEnumerator HideBubbleAfter(float ttlSec)
    {
        yield return new WaitForSeconds(ttlSec);
        if (bubbleInstance != null)
        {
            bubbleInstance.SetActive(false);
        }

        hideBubbleCoroutine = null;
    }
}
