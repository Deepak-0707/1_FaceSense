require("dotenv").config();
const express = require("express");
const cors = require("cors");
const connectDB = require("./config/db");
const sessionRoutes = require("./routes/sessionRoutes");
const baselineRoutes = require("./routes/baselineRoutes");
const userModelRoutes = require("./routes/userModelRoutes");
const analyticsRoutes = require("./routes/analyticsRoutes");
const alertRoutes = require("./routes/alertRoutes");

const app = express();
const PORT = process.env.PORT || 5000;

connectDB();

app.use(
  cors({
    origin: [
      "http://localhost:3000",
      "http://127.0.0.1:3000",
      /^http:\/\/localhost:\d+$/,
      /^https:\/\/.*\.vercel\.app$/,
      ...(process.env.FRONTEND_URL ? [process.env.FRONTEND_URL] : []),
    ],
    methods: ["GET", "POST", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
  })
);

app.use(express.json({ limit: "2mb" }));

app.use("/api/session", sessionRoutes);
app.use("/api/baseline", baselineRoutes);
app.use("/api/usermodel", userModelRoutes);
app.use("/api/analytics", analyticsRoutes);
app.use("/api/alerts", alertRoutes);

app.get("/health", (_req, res) => {
  res.json({ status: "ok", version: "4.0", timestamp: new Date().toISOString() });
});

app.use((_req, res) => {
  res.status(404).json({ error: "Route not found" });
});

app.use((err, _req, res, _next) => {
  console.error("Unhandled error:", err);
  res.status(500).json({ error: "Internal server error" });
});

app.listen(PORT, () => {
  console.log(`\n FaceSense  backend running on http://localhost:${PORT}`);
  console.log(`    Health:      http://localhost:${PORT}/health`);
  console.log(`    Sessions:    http://localhost:${PORT}/api/session`);
  console.log(`    Baseline:    http://localhost:${PORT}/api/baseline/:userId`);
  console.log(`    UserModel:   http://localhost:${PORT}/api/usermodel/:userId`);
  console.log(`    Analytics:   http://localhost:${PORT}/api/analytics/summary?userId=<id>`);
  console.log(`    Alerts:      http://localhost:${PORT}/api/alerts?userId=<id>\n`);
});
