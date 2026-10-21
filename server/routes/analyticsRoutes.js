const express = require("express");
const router = express.Router();
const Session = require("../models/Session");

function buildDateFilter(query) {
  const filter = {};
  if (query.userId) filter.userId = query.userId;
  if (query.from || query.to) {
    filter.timestamp = {};
    if (query.from) filter.timestamp.$gte = new Date(query.from);
    if (query.to) filter.timestamp.$lte = new Date(query.to);
  }
  return filter;
}

router.get("/stress", async (req, res) => {
  try {
    if (!req.query.userId) {
      return res.status(400).json({ error: "userId is required" });
    }

    const filter = buildDateFilter(req.query);
    const intervalMin = parseInt(req.query.intervalMin) || 5;
    const intervalMs = intervalMin * 60 * 1000;

    const docs = await Session.find(filter)
      .sort({ timestamp: 1 })
      .select("timestamp stressScore")
      .lean();

    if (docs.length === 0) return res.json({ data: [] });

    const buckets = {};
    for (const d of docs) {
      const bucket = Math.floor(new Date(d.timestamp).getTime() / intervalMs) * intervalMs;
      if (!buckets[bucket]) buckets[bucket] = [];
      buckets[bucket].push(d.stressScore);
    }

    const data = Object.entries(buckets)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([ts, scores]) => ({
        timestamp: new Date(Number(ts)).toISOString(),
        avgStress: Math.round((scores.reduce((s, v) => s + v, 0) / scores.length) * 100) / 100,
        maxStress: Math.round(Math.max(...scores) * 100) / 100,
        samples: scores.length,
      }));

    return res.json({ data, count: data.length });
  } catch (err) {
    console.error("GET /api/analytics/stress error:", err.message);
    return res.status(500).json({ error: "Failed to fetch stress analytics" });
  }
});

router.get("/blink", async (req, res) => {
  try {
    if (!req.query.userId) {
      return res.status(400).json({ error: "userId is required" });
    }

    const filter = buildDateFilter(req.query);
    const intervalMin = parseInt(req.query.intervalMin) || 5;
    const intervalMs = intervalMin * 60 * 1000;

    const docs = await Session.find(filter)
      .sort({ timestamp: 1 })
      .select("timestamp blinkRate")
      .lean();

    if (docs.length === 0) return res.json({ data: [] });

    const buckets = {};
    for (const d of docs) {
      const bucket = Math.floor(new Date(d.timestamp).getTime() / intervalMs) * intervalMs;
      if (!buckets[bucket]) buckets[bucket] = [];
      buckets[bucket].push(d.blinkRate);
    }

    const data = Object.entries(buckets)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([ts, rates]) => ({
        timestamp: new Date(Number(ts)).toISOString(),
        avgBlinkRate: Math.round(rates.reduce((s, v) => s + v, 0) / rates.length),
        samples: rates.length,
      }));

    return res.json({ data, count: data.length });
  } catch (err) {
    console.error("GET /api/analytics/blink error:", err.message);
    return res.status(500).json({ error: "Failed to fetch blink analytics" });
  }
});

router.get("/emotion", async (req, res) => {
  try {
    if (!req.query.userId) {
      return res.status(400).json({ error: "userId is required" });
    }

    const filter = buildDateFilter(req.query);

    const pipeline = [
      { $match: filter },
      { $group: { _id: "$emotion", count: { $sum: 1 } } },
      { $sort: { count: -1 } },
    ];

    const results = await Session.aggregate(pipeline);
    const total = results.reduce((s, r) => s + r.count, 0);

    const data = results.map((r) => ({
      emotion: r._id,
      count: r.count,
      pct: total > 0 ? Math.round((r.count / total) * 1000) / 10 : 0,
    }));

    return res.json({ data, total });
  } catch (err) {
    console.error("GET /api/analytics/emotion error:", err.message);
    return res.status(500).json({ error: "Failed to fetch emotion analytics" });
  }
});

router.get("/summary", async (req, res) => {
  try {
    if (!req.query.userId) {
      return res.status(400).json({ error: "userId is required" });
    }

    const filter = buildDateFilter(req.query);

    const agg = await Session.aggregate([
      { $match: filter },
      {
        $group: {
          _id: null,
          avgStress: { $avg: "$stressScore" },
          maxStress: { $max: "$stressScore" },
          avgBlinkRate: { $avg: "$blinkRate" },
          totalSamples: { $sum: 1 },
          firstSeen: { $min: "$timestamp" },
          lastSeen: { $max: "$timestamp" },
        },
      },
    ]);

    const sessions = await Session.distinct("sessionId", filter);

    const summary = agg[0] ?? {
      avgStress: 0, maxStress: 0, avgBlinkRate: 0,
      totalSamples: 0, firstSeen: null, lastSeen: null,
    };

    return res.json({
      avgStress: Math.round((summary.avgStress ?? 0) * 100) / 100,
      maxStress: Math.round((summary.maxStress ?? 0) * 100) / 100,
      avgBlinkRate: Math.round(summary.avgBlinkRate ?? 0),
      totalSamples: summary.totalSamples,
      totalSessions: sessions.length,
      firstSeen: summary.firstSeen,
      lastSeen: summary.lastSeen,
    });
  } catch (err) {
    console.error("GET /api/analytics/summary error:", err.message);
    return res.status(500).json({ error: "Failed to fetch summary" });
  }
});

module.exports = router;
