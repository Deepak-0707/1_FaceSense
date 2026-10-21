const express = require("express");
const router = express.Router();
const Session = require("../models/Session");

const VALID_EMOTIONS = new Set([
  "happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral",
]);

function validateSnapshot(snap) {
  const { userId, sessionId, stressScore, blinkRate, emotion } = snap;
  if (!userId || typeof userId !== "string" || userId.trim() === "") {
    return "userId is required";
  }
  if (!sessionId || typeof sessionId !== "string" || sessionId.trim() === "") {
    return "sessionId is required";
  }
  if (stressScore === undefined || stressScore === null) return "stressScore is required";
  const ss = Number(stressScore);
  if (isNaN(ss) || ss < 0 || ss > 1) return "stressScore must be 0–1";
  if (blinkRate === undefined || blinkRate === null) return "blinkRate is required";
  const br = Number(blinkRate);
  if (isNaN(br) || br < 0) return "blinkRate must be >= 0";
  if (emotion && !VALID_EMOTIONS.has(emotion)) return `emotion must be one of: ${[...VALID_EMOTIONS].join(", ")}`;
  return null;
}

router.post("/", async (req, res) => {
  try {
    const body = req.body;
    const items = Array.isArray(body) ? body : [body];

    if (items.length === 0) {
      return res.status(400).json({ error: "Empty payload" });
    }
    if (items.length > 50) {
      return res.status(400).json({ error: "Batch too large (max 50)" });
    }

    const errors = [];
    const docs = [];

    for (let i = 0; i < items.length; i++) {
      const snap = items[i];
      const err = validateSnapshot(snap);
      if (err) {
        errors.push({ index: i, error: err });
        continue;
      }
      docs.push({
        userId: snap.userId.trim(),
        sessionId: snap.sessionId.trim(),
        timestamp: snap.timestamp ? new Date(snap.timestamp) : new Date(),
        emotion: snap.emotion || "neutral",
        stressScore: Number(snap.stressScore),
        blinkRate: Number(snap.blinkRate),
      });
    }

    if (docs.length === 0) {
      return res.status(400).json({ errors });
    }

    const saved = await Session.insertMany(docs);
    return res.status(201).json({
      success: true,
      inserted: saved.length,
      ...(errors.length > 0 && { errors }),
    });
  } catch (err) {
    console.error("POST /api/session error:", err.message);
    return res.status(500).json({ error: "Failed to save session data" });
  }
});

router.get("/", async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 100, 500);
    const filter = {};
    if (req.query.userId) filter.userId = req.query.userId;
    if (req.query.sessionId) filter.sessionId = req.query.sessionId;
    if (req.query.since) filter.timestamp = { $gte: new Date(req.query.since) };

    const sessions = await Session.find(filter)
      .sort({ timestamp: -1 })
      .limit(limit)
      .lean();

    return res.json({ count: sessions.length, data: sessions });
  } catch (err) {
    console.error("GET /api/session error:", err.message);
    return res.status(500).json({ error: "Failed to fetch session data" });
  }
});

router.get("/stats", async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 100, 500);
    const matchStage = {};
    if (req.query.userId) matchStage.userId = req.query.userId;

    const pipeline = [
      ...(Object.keys(matchStage).length ? [{ $match: matchStage }] : []),
      { $sort: { timestamp: -1 } },
      { $limit: limit },
      {
        $group: {
          _id: null,
          avgStress: { $avg: "$stressScore" },
          maxStress: { $max: "$stressScore" },
          avgBlinkRate: { $avg: "$blinkRate" },
          count: { $sum: 1 },
        },
      },
    ];

    const agg = await Session.aggregate(pipeline);

    const emotionPipeline = [
      ...(Object.keys(matchStage).length ? [{ $match: matchStage }] : []),
      { $sort: { timestamp: -1 } },
      { $limit: limit },
      { $group: { _id: "$emotion", count: { $sum: 1 } } },
      { $sort: { count: -1 } },
    ];

    const emotionCounts = await Session.aggregate(emotionPipeline);

    return res.json({
      summary: agg[0] ?? { avgStress: 0, maxStress: 0, avgBlinkRate: 0, count: 0 },
      emotionBreakdown: emotionCounts,
    });
  } catch (err) {
    console.error("GET /api/session/stats error:", err.message);
    return res.status(500).json({ error: "Failed to compute stats" });
  }
});

module.exports = router;
