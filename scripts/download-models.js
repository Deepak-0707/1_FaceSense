

const https = require("https");
const fs = require("fs");
const path = require("path");

const FORCE = process.argv.includes("--force");

const BASE_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";

const OUT_DIR = path.join(__dirname, "..", "public", "models");

const FILES = [
  // Face detection
  "ssd_mobilenetv1_model-weights_manifest.json",
  "ssd_mobilenetv1_model-shard1",
  "ssd_mobilenetv1_model-shard2",

  // Landmarks
  "face_landmark_68_model-weights_manifest.json",
  "face_landmark_68_model-shard1",

  // Emotion 
  "face_expression_model-weights_manifest.json",
  "face_expression_model-shard1",

  // Face recognition / descriptors 
  "face_recognition_model-weights_manifest.json",
  "face_recognition_model-shard1",
];

if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  console.log("Created: " + OUT_DIR);
}

function downloadFile(filename) {
  return new Promise((resolve, reject) => {
    const dest = path.join(OUT_DIR, filename);

    if (!FORCE && fs.existsSync(dest) && fs.statSync(dest).size > 0) {
      console.log("↩  Skipping (exists): " + filename);
      return resolve();
    }

    const tmpDest = dest + ".tmp";
    const file = fs.createWriteStream(tmpDest);

    https.get(BASE_URL + filename, (res) => {
      if (res.statusCode !== 200) {
        fs.unlink(tmpDest, () => { });
        return reject(new Error("HTTP " + res.statusCode + " for " + filename));
      }

      res.pipe(file);

      file.on("finish", () => {
        file.close(() => {
          fs.renameSync(tmpDest, dest);
          const kb = (fs.statSync(dest).size / 1024).toFixed(1);
          console.log("✓  Downloaded: " + filename + " (" + kb + " KB)");
          resolve();
        });
      });
    }).on("error", (err) => {
      fs.unlink(tmpDest, () => { });
      reject(err);
    });
  });
}

(async () => {
  console.log("\n  FaceSense ");
  console.log("    Source: " + BASE_URL);
  if (FORCE) console.log("    Mode: --force (re-downloading all)\n");
  else console.log("");

  let failed = 0;

  for (const file of FILES) {
    try {
      await downloadFile(file);
    } catch (err) {
      console.error("✗  Failed: " + file + " — " + err.message);
      failed++;
    }
  }

  console.log("");
  if (failed > 0) {
    console.error("❌  " + failed + " file(s) failed. Retry or check your connection.\n");
    process.exit(1);
  } else {
    console.log("✅  All models ready! Run: npm run dev\n");
  }
})();
