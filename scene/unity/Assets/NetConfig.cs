using System;
using UnityEngine;

public sealed class NetConfig : MonoBehaviour
{
    [Header("Base URL")]
    [Tooltip("If empty, host is resolved from Application.absoluteURL (WebGL) with localhost fallback.")]
    [SerializeField] private string originOverride = "http://localhost:8000";

    [Header("Paths")]
    [SerializeField] private string apiPrefix = "/api";
    [SerializeField] private string statePath = "/state";

    public string Origin => ResolveOrigin();
    public string ApiBaseUrl => CombineUrl(Origin, NormalizePath(apiPrefix, trimTrailingSlash: true));
    public string StateUrl => CombineUrl(ApiBaseUrl, NormalizePath(statePath, trimTrailingSlash: false));

    private string resolvedOrigin = "";

    private void Awake()
    {
        resolvedOrigin = ResolveOrigin();
    }

    private void OnValidate()
    {
        resolvedOrigin = "";
    }

    private string ResolveOrigin()
    {
        if (!string.IsNullOrWhiteSpace(originOverride))
        {
            return originOverride.Trim().TrimEnd('/');
        }

        if (!string.IsNullOrWhiteSpace(resolvedOrigin))
        {
            return resolvedOrigin;
        }

        string absoluteUrl = Application.absoluteURL;
        if (Uri.TryCreate(absoluteUrl, UriKind.Absolute, out Uri uri))
        {
            return $"{uri.Scheme}://{uri.Authority}";
        }

        return "http://localhost";
    }

    private static string NormalizePath(string path, bool trimTrailingSlash)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        string normalized = path.Trim();
        if (!normalized.StartsWith("/"))
        {
            normalized = "/" + normalized;
        }

        if (trimTrailingSlash && normalized.Length > 1)
        {
            normalized = normalized.TrimEnd('/');
        }

        return normalized;
    }

    private static string CombineUrl(string root, string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return root.TrimEnd('/');
        }

        return root.TrimEnd('/') + path;
    }
}
