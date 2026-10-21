const mongoose = require("mongoose");

async function connectDB() {
  try {
    const conn = await mongoose.connect(process.env.MONGO_URI, {

    });
    console.log(`✅  MongoDB connected: ${conn.connection.host}`);
  } catch (err) {
    console.error("❌  MongoDB connection failed:", err.message);
    console.error(
      "    Check your MONGO_URI in server/.env and ensure your IP is whitelisted in Atlas."
    );
    process.exit(1);
  }
}

module.exports = connectDB;
