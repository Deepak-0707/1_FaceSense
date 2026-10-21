import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FaceSense — AI Wellness Monitor",
  description:
    " Production-ready stress, fatigue & emotion detection. Real-time notifications, analytics dashboard, MongoDB Atlas.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (

    <html lang="en">
      <head>
        {/* Register service worker for optional background push notifications */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                window.addEventListener('load', function() {
                  navigator.serviceWorker.register('/sw.js')
                    .then(function(reg) {
                      console.log('[FaceSense] SW registered, scope:', reg.scope);
                    })
                    .catch(function(err) {
                      console.log('[FaceSense] SW registration failed:', err);
                    });
                });
              }
            `,
          }}
        />
      </head>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
