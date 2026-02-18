const CACHE_PREFIX = "t34M4 n3 R4k0V-Skebobia Scene-";
const CACHE_NAME = `${CACHE_PREFIX}disabled-v2`;

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((key) => key.startsWith(CACHE_PREFIX) || key.startsWith("UnityCache_"))
          .map((key) => caches.delete(key)),
      );
      await self.clients.claim();
      console.log("[Service Worker] Cache disabled and old caches removed");
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  // Network-only strategy to avoid stale wasm/data/framework mismatches.
  event.respondWith(fetch(event.request));
});
