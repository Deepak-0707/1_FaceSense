const mongoose = require("mongoose");

const AlertSchema = new mongoose.Schema(
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
    type: {
      type: String,
      enum: ["stress", "fatigue"],
      required: true,
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
    timestamp: {
      type: Date,
      default: Date.now,
      index: true,
    },
  },
  {
    timestamps: true,
    collection: "alerts",
  }
);

AlertSchema.index({ userId: 1, timestamp: -1 });

module.exports = mongoose.model("Alert", AlertSchema);
