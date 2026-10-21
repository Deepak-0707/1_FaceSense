const CACHE_NAME = "facesense-v4";

self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      cache.addAll(["/", "/dashboard"])
    )
  );
});
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== CACHE_NAME)
          .map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});
self.addEventListener("push", (event) => {
  let data = { title: "FaceSense Alert", body: "Check your wellness status." };

  if (event.data) {
    try {
      data = event.data.json();
    } catch {
      data.body = event.data.text();
    }
  }

  event.waitUntil(
    self.registration.showNotification(data.title ?? "FaceSense", {
      body: data.body ?? "",
      icon: "/favicon.ico",
      badge: "/favicon.ico",
      tag: data.tag ?? "facesense-alert",
      requireInteraction: false,
      data: { url: data.url ?? "/dashboard" },
    })
  );
});
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = event.notification.data?.url ?? "/dashboard";
  event.waitUntil(
    clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then((windowClients) => {
        const existing = windowClients.find(
          (c) => c.url.includes(url) && "focus" in c
        );
        if (existing) return existing.focus();
        return clients.openWindow(url);
      })
  );
});
