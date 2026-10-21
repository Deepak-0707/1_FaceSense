const mongoose = require("mongoose");


const SessionSchema = new mongoose.Schema(
  {
    userId: {
      type: String,
      required: true,
      index: true,
    },
    sessionId: {
      type: String,
      required: true,
      index: true,
    },
    timestamp: {
      type: Date,
      default: Date.now,
      index: true,
    },
    emotion: {
      type: String,
      enum: ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"],
      default: "neutral",
    },
    stressScore: {
      type: Number,
      min: 0,
      max: 1,
      default: 0,
    },
    blinkRate: {
      type: Number,
      min: 0,
      default: 0,
    },
  },
  {
    timestamps: true,
    collection: "sessions",
  }
);

SessionSchema.index({ userId: 1, timestamp: -1 });

module.exports = mongoose.model("Session", SessionSchema);
