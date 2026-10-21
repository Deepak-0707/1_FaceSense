const mongoose = require("mongoose");

const UserModelSchema = new mongoose.Schema(
  {
    userId: {
      type: String,
      required: true,
      unique: true,
      index: true,
    },
    samples: [
      {
        descriptor: {
          type: [Number],
          required: true,
        },
        label: {
          type: String,
          enum: ["relaxed", "stressed"],
          required: true,
        },
        capturedAt: {
          type: Date,
          default: Date.now,
        },
      },
    ],
  },
  {
    timestamps: true,
    collection: "usermodels",
  }
);

module.exports = mongoose.model("UserModel", UserModelSchema);
