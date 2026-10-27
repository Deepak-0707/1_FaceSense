const express = require("express");
const router = express.Router();
const Alert = require("../models/Alert");

router.post("/", async (req, res) => {
  try {
    const { userId, sessionId, type, stressScore, blinkRate, durationSec, message } = req.body;

    if (!userId || typeof userId !== "string" || userId.trim() === "") {
      return res.status(400).json({ error: "userId is required" });
    }
    if (!sessionId || typeof sessionId !== "string") {
      return res.status(400).json({ error: "sessionId is required" });
    }
    if (!["stress", "fatigue"].includes(type)) {
      return res.status(400).json({ error: "type must be 'stress' or 'fatigue'" });
    }

    const alert = await Alert.create({
      userId: userId.trim(),
      sessionId: sessionId.trim(),
      type,
      stressScore: Number(stressScore) || 0,
      blinkRate: Number(blinkRate) || 0,
      durationSec: Number(durationSec) || 0,
      message: typeof message === "string" ? message.trim() : "",
      timestamp: new Date(),
    });

    return res.status(201).json({ success: true, alert });
  } catch (err) {
    console.error("POST /api/alerts error:", err.message);
    return res.status(500).json({ error: "Failed to save alert" });
  }
});

router.get("/", async (req, res) => {
  try {
    if (!req.query.userId) {
      return res.status(400).json({ error: "userId is required" });
    }

    const filter = { userId: req.query.userId };
    if (req.query.from || req.query.to) {
      filter.timestamp = {};
      if (req.query.from) filter.timestamp.$gte = new Date(req.query.from);
      if (req.query.to) filter.timestamp.$lte = new Date(req.query.to);
    }
    if (req.query.type) filter.type = req.query.type;

    const limit = Math.min(parseInt(req.query.limit) || 50, 200);

    const alerts = await Alert.find(filter)
      .sort({ timestamp: -1 })
      .limit(limit)
      .lean();

    return res.json({ count: alerts.length, data: alerts });
  } catch (err) {
    console.error("GET /api/alerts error:", err.message);
    return res.status(500).json({ error: "Failed to fetch alerts" });
  }
});

module.exports = router;
