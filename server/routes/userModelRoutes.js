const express = require("express");
const router = express.Router();
const UserModel = require("../models/UserModel");

router.post("/", async (req, res) => {
  try {
    const { userId, samples } = req.body;

    if (!userId || typeof userId !== "string") {
      return res.status(400).json({ error: "userId is required" });
    }
    if (!Array.isArray(samples) || samples.length === 0) {
      return res.status(400).json({ error: "samples must be a non-empty array" });
    }


    for (const s of samples) {
      if (!Array.isArray(s.descriptor) || s.descriptor.length < 128) {
        return res.status(400).json({ error: "Each sample needs a descriptor array (>=128 values)" });
      }
      if (!["relaxed", "stressed"].includes(s.label)) {
        return res.status(400).json({ error: "Each sample label must be 'relaxed' or 'stressed'" });
      }
    }

    const doc = await UserModel.findOneAndUpdate(
      { userId: userId.trim() },
      {
        $push: {
          samples: {
            $each: samples.map((s) => ({
              descriptor: s.descriptor,
              label: s.label,
              capturedAt: new Date(),
            })),
          },
        },
      },
      { upsert: true, new: true }
    );

    return res.status(200).json({ success: true, totalSamples: doc.samples.length });
  } catch (err) {
    console.error("POST /api/usermodel error:", err.message);
    return res.status(500).json({ error: "Failed to save model" });
  }
});

router.get("/:userId", async (req, res) => {
  try {
    const { userId } = req.params;
    const doc = await UserModel.findOne({ userId }).lean();
    if (!doc || doc.samples.length < 2) {
      return res.status(404).json({ error: "No model found", fallback: true });
    }
    return res.json({ userId, samples: doc.samples });
  } catch (err) {
    console.error("GET /api/usermodel error:", err.message);
    return res.status(500).json({ error: "Failed to fetch model" });
  }
});
router.delete("/:userId", async (req, res) => {
  try {
    await UserModel.deleteOne({ userId: req.params.userId });
    return res.json({ success: true });
  } catch (err) {
    return res.status(500).json({ error: "Failed to delete model" });
  }
});

module.exports = router;
