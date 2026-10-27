"use client";


import { useEffect, useRef, useCallback, useState } from "react";
import Link from "next/link";
import type * as FaceAPI from "@vladmandic/face-api";

let faceapi: any = null;
async function getFaceApi() {
  if (!faceapi) {
    faceapi = await import("@vladmandic/face-api");
  }
  return faceapi;
}

type Status =
  | "idle"
  | "loading-models"
  | "models-ready"
  | "requesting-camera"
  | "calibrating"
  | "detecting"
  | "no-face"
  | "error";

const STATUS_LABELS: Record<Status, string> = {
  idle: "Camera Off",
  "loading-models": "Loading Models…",
  "models-ready": "Models Ready",
  "requesting-camera": "Requesting Camera…",
  calibrating: "Calibrating (30s)…",
  detecting: "Detecting Face…",
  "no-face": "No Face Detected",
  error: "Error — See Below",
};

const STATUS_COLORS: Record<Status, string> = {
  idle: "text-zinc-500",
  "loading-models": "text-amber-400",
  "models-ready": "text-sky-400",
  "requesting-camera": "text-amber-400",
  calibrating: "text-purple-400",
  detecting: "text-emerald-400",
  "no-face": "text-orange-400",
  error: "text-red-400",
};

type FaceResult = {
  detection: FaceAPI.FaceDetection;
  landmarks: FaceAPI.FaceLandmarks68;
  expressions: FaceAPI.FaceExpressions;
  descriptor?: Float32Array;
};

type TrackedFace = {
  id: number;
  box: { x: number; y: number; width: number; height: number };
  score: number;
  detection: FaceAPI.FaceDetection;
  landmarks: FaceAPI.FaceLandmarks68;
  expressions: FaceAPI.FaceExpressions;
  dominantEmotion: string;
  stressScore: number;
  stressPct: number;
  stressColor: string;
  ear: number;
  openEarBaseline: number;
  isEyeClosed: boolean;
  eyeClosedStartTime: number;
  blinkCount: number;
  recentBlinks: number[];
  blinkRate: number;
  lastSeen: number;
  // Prolonged stress tracking
  stressHighSince: number | null;
  stressDurationSec: number;
  isCalmAlertActive: boolean;
};

type CalmAlertDetails = {
  faceId: number;
  stressPct: number;
  durationSec: number;
  thresholdPct: number;
  thresholdSec: number;
  timestamp: number;
};

type StressProgress = {
  active: boolean;
  durationSec: number;
  thresholdSec: number;
  stressPct: number;
  faceId: number;
};

type TrainLabel = "relaxed" | "stressed";
type TrainPhase = "idle" | "relaxed" | "stressed" | "done";

type KNNSample = {
  descriptor: number[];
  label: TrainLabel;
};

type Baseline = {
  avgStress: number;
  avgBlinkRate: number;
};

// Stress weights per emotion (0 = calm, 1 = max stress)
const STRESS_WEIGHTS: Record<string, number> = {
  angry: 0.95,
  disgusted: 0.75,
  fearful: 0.90,
  sad: 0.65,
  surprised: 0.45,
  neutral: 0.10,
  happy: 0.05,
};

const EMOTION_EMOJI: Record<string, string> = {
  angry: "😠", disgusted: "🤢", fearful: "😨",
  sad: "😢", surprised: "😲", neutral: "😐", happy: "😊",
};

const EMOTION_COLORS: Record<string, string> = {
  angry: "#ef4444", disgusted: "#a855f7", fearful: "#f97316",
  sad: "#3b82f6", surprised: "#eab308", neutral: "#6b7280", happy: "#22c55e",
};

const EAR_THRESHOLD = 0.21;
const BLINK_CONSEC_FRAMES = 2;
const CALIBRATION_DURATION_MS = 30000;
const ALERT_COOLDOWN_MS = 30000;
const BATCH_INTERVAL_MS = 5000;
const TRAIN_SAMPLES_NEEDED = 25;
const KNN_K = 3;

const FALLBACK_STRESS_THRESHOLD = 0.7;
const FALLBACK_BLINK_THRESHOLD = 8;

//  Utility functions
function euclidean(a: FaceAPI.Point, b: FaceAPI.Point) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function euclideanDesc(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

function eyeAspectRatio(pts: FaceAPI.Point[]): number {
  const A = euclidean(pts[1], pts[5]);
  const B = euclidean(pts[2], pts[4]);
  const C = euclidean(pts[0], pts[3]);
  return (A + B) / (2.0 * C);
}

function mapEmotionToStress(expressions: FaceAPI.FaceExpressions): number {
  let score = 0;
  const exprObj = expressions as unknown as Record<string, number>;
  for (const [emotion, weight] of Object.entries(STRESS_WEIGHTS)) {
    score += (exprObj[emotion] ?? 0) * weight;
  }
  return Math.min(1, score);
}

function dominantEmotion(expressions: FaceAPI.FaceExpressions): string {
  const exprObj = expressions as unknown as Record<string, number>;
  return Object.entries(exprObj).reduce((a, b) => (b[1] > a[1] ? b : a))[0];
}

// KNN Classifier
function knnPredict(
  samples: KNNSample[],
  query: number[],
  k: number = KNN_K
): TrainLabel | null {
  if (samples.length < k) return null;
  const distances = samples.map((s) => ({
    label: s.label,
    dist: euclideanDesc(s.descriptor, query),
  }));
  distances.sort((a, b) => a.dist - b.dist);
  const nearest = distances.slice(0, k);
  const votes: Record<string, number> = {};
  for (const n of nearest) votes[n.label] = (votes[n.label] || 0) + 1;
  return (Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0]) as TrainLabel;
}

//  User identity helpers ──
function getOrCreateUserId(): string {
  if (typeof window === "undefined") return "server";
  let id = localStorage.getItem("facesense_userId");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("facesense_userId", id);
  }
  return id;
}

function generateSessionId(): string {
  return crypto.randomUUID();
}

//  Backend URL 
const BACKEND_URL =
  typeof process !== "undefined" && process.env.NEXT_PUBLIC_BACKEND_URL
    ? process.env.NEXT_PUBLIC_BACKEND_URL
    : "http://localhost:5000";

//  Component ──
export default function FaceSensePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const rafRef = useRef<number | null>(null);
  const inferIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const apiIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const latestResultsRef = useRef<FaceResult[]>([]);
  const trackedFacesRef = useRef<Map<number, TrackedFace>>(new Map());
  const nextFaceIdRef = useRef<number>(1);
  const latestTrackedFacesRef = useRef<TrackedFace[]>([]);
  const [multiFaceSummaries, setMultiFaceSummaries] = useState<
    { id: number; emotion: string; stressPct: number; blinkRate: number }[]
  >([]);

  const modelsLoadedRef = useRef(false);
  const isRunningRef = useRef(false);

  // Identity / session
  const userIdRef = useRef<string>("");
  const sessionIdRef = useRef<string>("");

  // Blink tracking
  const blinkCountRef = useRef(0);
  const blinkConsecRef = useRef(0);
  const blinkStartTimeRef = useRef<number>(Date.now());
  const eyeClosedRef = useRef(false);

  // Alert tracking with cooldown
  const stressHighSinceRef = useRef<number | null>(null);
  const alertFiredRef = useRef(false);
  const lastAlertTimeRef = useRef<number>(0);

  // Latest analytics refs
  const latestEmotionRef = useRef("neutral");
  const latestStressRef = useRef(0);
  const latestBlinkRateRef = useRef(0);
  const latestDescriptorRef = useRef<number[] | null>(null);

  // Calibration
  const calibrationSamplesRef = useRef<{ stress: number; blinkRate: number }[]>([]);
  const calibrationStartRef = useRef<number>(0);
  const isCalibrationDoneRef = useRef(false);

  // Adaptive thresholds
  const stressThresholdRef = useRef(FALLBACK_STRESS_THRESHOLD);
  const blinkThresholdRef = useRef(FALLBACK_BLINK_THRESHOLD);

  // API batch buffer
  const batchBufferRef = useRef<object[]>([]);

  // KNN training state
  const knnSamplesRef = useRef<KNNSample[]>([]);
  const trainingCountRef = useRef(0);

  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [faceCount, setFaceCount] = useState(0);
  const [fps, setFps] = useState(0);

  // Phase 2 UI state
  const [emotion, setEmotion] = useState("neutral");
  const [stressScore, setStressScore] = useState(0);
  const [blinkRate, setBlinkRate] = useState(0);
  const [blinkCount, setBlinkCount] = useState(0);
  const [alertStress, setAlertStress] = useState(false);
  const [alertBlink, setAlertBlink] = useState(false);
  const alertBlinkRef = useRef(false);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);

  // Phase 3 UI state
  const [calibProgress, setCalibProgress] = useState(0);
  const [baseline, setBaseline] = useState<Baseline | null>(null);
  const [knnPrediction, setKnnPrediction] = useState<TrainLabel | null>(null);
  const [trainPhase, setTrainPhase] = useState<TrainPhase>("idle");
  const [trainCount, setTrainCount] = useState(0);
  const [modelReady, setModelReady] = useState(false);
  const [showTrainingPanel, setShowTrainingPanel] = useState(false);
  const [userId, setUserId] = useState<string>("");
  const [notifPermission, setNotifPermission] = useState<NotificationPermission>("default");

  // Prolonged Stress & Calm Reminder State & Controls
  const [stressThresholdPct, setStressThresholdPct] = useState<number>(65);
  const [stressDurationThresholdSec, setStressDurationThresholdSec] = useState<number>(8);
  const [soundAlertMode, setSoundAlertMode] = useState<"chime" | "beep" | "mute">("chime");
  const [voiceAlertEnabled, setVoiceAlertEnabled] = useState<boolean>(true);
  const [showSettingsPanel, setShowSettingsPanel] = useState<boolean>(false);
  const [showBreathingGuide, setShowBreathingGuide] = useState<boolean>(false);
  const [calmAlertDetails, setCalmAlertDetails] = useState<CalmAlertDetails | null>(null);
  const [stressProgress, setStressProgress] = useState<StressProgress | null>(null);
  const [calmRestoredMsg, setCalmRestoredMsg] = useState<string | null>(null);

  // Sync refs for uninterrupted access in inference/draw loops
  const stressThresholdPctRef = useRef(65);
  const stressDurationThresholdSecRef = useRef(8);
  const soundAlertModeRef = useRef<"chime" | "beep" | "mute">("chime");
  const voiceAlertEnabledRef = useRef(true);

  const fpsRef = useRef({ frames: 0, last: performance.now() });

  // ── Initialise user identity & settings ─
  useEffect(() => {
    const id = getOrCreateUserId();
    userIdRef.current = id;
    setUserId(id);

    // Request browser notification permission
    if ("Notification" in window) {
      setNotifPermission(Notification.permission);
      if (Notification.permission === "default") {
        Notification.requestPermission().then((perm) => {
          setNotifPermission(perm);
        });
      }
    }

    // Load persisted threshold & alert preferences
    try {
      const savedThresh = localStorage.getItem("facesense_stress_threshold");
      if (savedThresh) {
        const val = Number(savedThresh);
        if (!isNaN(val) && val >= 30 && val <= 95) {
          setStressThresholdPct(val);
          stressThresholdPctRef.current = val;
          stressThresholdRef.current = val / 100;
        }
      }
      const savedDuration = localStorage.getItem("facesense_duration_threshold");
      if (savedDuration) {
        const val = Number(savedDuration);
        if (!isNaN(val) && val >= 2 && val <= 60) {
          setStressDurationThresholdSec(val);
          stressDurationThresholdSecRef.current = val;
        }
      }
      const savedSound = localStorage.getItem("facesense_sound_mode");
      if (savedSound === "chime" || savedSound === "beep" || savedSound === "mute") {
        setSoundAlertMode(savedSound);
        soundAlertModeRef.current = savedSound;
      }
      const savedVoice = localStorage.getItem("facesense_voice_alert");
      if (savedVoice !== null) {
        const val = savedVoice === "true";
        setVoiceAlertEnabled(val);
        voiceAlertEnabledRef.current = val;
      }
    } catch {
      // offline / private mode
    }

    // Try to load user's KNN model from backend
    fetch(`${BACKEND_URL}/api/usermodel/${id}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.samples && data.samples.length >= KNN_K * 2) {
          knnSamplesRef.current = data.samples;
          setModelReady(true);
        }
      })
      .catch(() => {/* offline — ok */ });

    // Try to load baseline
    fetch(`${BACKEND_URL}/api/baseline/${id}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.baseline) {
          setBaseline({ avgStress: data.baseline.avgStress, avgBlinkRate: data.baseline.avgBlinkRate });
          stressThresholdRef.current = data.stressThreshold ?? FALLBACK_STRESS_THRESHOLD;
          blinkThresholdRef.current = data.blinkThreshold ?? FALLBACK_BLINK_THRESHOLD;
        }
      })
      .catch(() => {/* offline — use fallbacks */ });
  }, []);

  // Setting update handlers with localStorage sync
  const updateStressThreshold = useCallback((val: number) => {
    setStressThresholdPct(val);
    stressThresholdPctRef.current = val;
    stressThresholdRef.current = val / 100;
    try {
      localStorage.setItem("facesense_stress_threshold", String(val));
    } catch { }
  }, []);

  const updateDurationThreshold = useCallback((val: number) => {
    setStressDurationThresholdSec(val);
    stressDurationThresholdSecRef.current = val;
    try {
      localStorage.setItem("facesense_duration_threshold", String(val));
    } catch { }
  }, []);

  const updateSoundMode = useCallback((val: "chime" | "beep" | "mute") => {
    setSoundAlertMode(val);
    soundAlertModeRef.current = val;
    try {
      localStorage.setItem("facesense_sound_mode", val);
    } catch { }
  }, []);

  const updateVoiceAlert = useCallback((val: boolean) => {
    setVoiceAlertEnabled(val);
    voiceAlertEnabledRef.current = val;
    try {
      localStorage.setItem("facesense_voice_alert", String(val));
    } catch { }
  }, []);

  const triggerTestCalmAlert = useCallback(() => {
    const testDetails: CalmAlertDetails = {
      faceId: 1,
      stressPct: 82,
      durationSec: stressDurationThresholdSecRef.current,
      thresholdPct: stressThresholdPctRef.current,
      thresholdSec: stressDurationThresholdSecRef.current,
      timestamp: Date.now(),
    };
    setAlertStress(true);
    setCalmAlertDetails(testDetails);
    setShowBreathingGuide(true);

    if (soundAlertModeRef.current === "chime") {
      playCalmChime();
    } else if (soundAlertModeRef.current === "beep") {
      playBeep(880, 0.3);
    }

    if (voiceAlertEnabledRef.current) {
      speakCalmReminder("Please take a deep breath and stay calm.");
    }

    sendBrowserNotification(
      "🌿 FaceSense: Reminder to Be Calm (Test)",
      "This is a demonstration alert reminding you to take a deep breath and stay calm."
    );
  }, []);

  // ── Load models ──
  const loadModels = useCallback(async () => {
    if (modelsLoadedRef.current) return;
    setStatus("loading-models");
    setErrorMsg("");
    try {
      faceapi = await getFaceApi();
      await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
        faceapi.nets.faceExpressionNet.loadFromUri("/models"),
        faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
      ]);
      modelsLoadedRef.current = true;
      setStatus("models-ready");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setErrorMsg(`Model load failed: ${msg}. Run: node scripts/download-models.js`);
      setStatus("error");
    }
  }, []);

  // ── Start camera ─
  const startCamera = useCallback(async () => {
    if (!modelsLoadedRef.current) {
      await loadModels();
      if (!modelsLoadedRef.current) return;
    }
    setStatus("requesting-camera");
    setErrorMsg("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;

      const video = videoRef.current!;
      video.srcObject = stream;
      await new Promise<void>((resolve) => {
        video.onloadedmetadata = () => { video.play().then(resolve); };
      });

      const canvas = canvasRef.current!;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // New session ID
      sessionIdRef.current = generateSessionId();

      // Reset blink/stress/calibration state
      blinkCountRef.current = 0;
      blinkConsecRef.current = 0;
      blinkStartTimeRef.current = Date.now();
      eyeClosedRef.current = false;
      stressHighSinceRef.current = null;
      alertFiredRef.current = false;
      lastAlertTimeRef.current = 0;
      batchBufferRef.current = [];
      calibrationSamplesRef.current = [];
      calibrationStartRef.current = Date.now();
      isCalibrationDoneRef.current = false;
      trackedFacesRef.current.clear();
      nextFaceIdRef.current = 1;
      latestTrackedFacesRef.current = [];
      setMultiFaceSummaries([]);
      setCalibProgress(0);
      setStressProgress(null);
      setCalmAlertDetails(null);

      isRunningRef.current = true;
      setStatus("calibrating");
      startInferenceLoop();
      startDrawLoop();
      startApiLoop();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      if (/permission|denied/i.test(msg)) {
        setErrorMsg("Camera access denied. Allow camera in browser settings.");
      } else if (/notfound|devices/i.test(msg)) {
        setErrorMsg("No camera found on this device.");
      } else {
        setErrorMsg(`Camera error: ${msg}`);
      }
      setStatus("error");
    }
  }, [loadModels]);

  // ── Stop camera ──
  const stopCamera = useCallback(() => {
    isRunningRef.current = false;

    if (rafRef.current !== null) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (inferIntervalRef.current !== null) { clearInterval(inferIntervalRef.current); inferIntervalRef.current = null; }
    if (apiIntervalRef.current !== null) { clearInterval(apiIntervalRef.current); apiIntervalRef.current = null; }
    latestResultsRef.current = [];
    latestTrackedFacesRef.current = [];
    trackedFacesRef.current.clear();
    nextFaceIdRef.current = 1;

    // Flush remaining buffer
    if (batchBufferRef.current.length > 0) {
      flushBatch(batchBufferRef.current.splice(0));
    }

    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;

    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }

    setFaceCount(0); setFps(0);
    setEmotion("neutral"); setStressScore(0); setBlinkRate(0); setBlinkCount(0);
    setMultiFaceSummaries([]);
    alertBlinkRef.current = false;
    setAlertStress(false); setAlertBlink(false);
    setCalibProgress(0);
    setStressProgress(null);
    setCalmAlertDetails(null);
    setShowBreathingGuide(false);
    setStatus("idle"); setErrorMsg("");
  }, []);

  // ── Calibration finalization ──
  const finalizeCalibration = useCallback((samples: { stress: number; blinkRate: number }[]) => {
    if (samples.length < 5) return;
    const avgStress = samples.reduce((s, x) => s + x.stress, 0) / samples.length;
    const avgBlinkRate = samples.reduce((s, x) => s + x.blinkRate, 0) / samples.length;

    const newBaseline: Baseline = { avgStress, avgBlinkRate };
    setBaseline(newBaseline);
    stressThresholdRef.current = Math.min(0.95, avgStress + 0.2);
    blinkThresholdRef.current = avgBlinkRate * 0.7;

    // Save to backend (non-blocking)
    fetch(`${BACKEND_URL}/api/baseline`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        userId: userIdRef.current,
        avgStress,
        avgBlinkRate,
        sampleCount: samples.length,
      }),
    }).catch(() => {/* offline */ });
  }, []);

  // ── API batch flush ──
  const flushBatch = useCallback(async (items: object[]) => {
    if (items.length === 0) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(items),
      });
      setBackendOk(res.ok);
    } catch {
      setBackendOk(false);
    }
  }, []);

  // ── Inference loop 
  const startInferenceLoop = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    let isProcessing = false;

    inferIntervalRef.current = setInterval(async () => {
      if (!isRunningRef.current || !video || video.paused || video.ended) return;
      if (isProcessing) return;
      isProcessing = true;

      try {
        const needsDescriptor =
          knnSamplesRef.current.length >= KNN_K * 2 ||
          (trainPhaseRef.current !== "idle" && trainPhaseRef.current !== "done");

        let detectQuery = faceapi
          .detectAllFaces(video, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
          .withFaceLandmarks()
          .withFaceExpressions();

        if (needsDescriptor) {
          detectQuery = detectQuery.withFaceDescriptors();
        }

        const detections = await detectQuery;

        const displaySize = { width: video.videoWidth, height: video.videoHeight };
        const resized = faceapi.resizeResults(detections, displaySize) as FaceResult[];
        latestResultsRef.current = resized;
        setFaceCount(resized.length);

        const nowMs = Date.now();
        const elapsed = nowMs - calibrationStartRef.current;

        // Calibration phase
        if (!isCalibrationDoneRef.current) {
          const pct = Math.min(100, Math.round((elapsed / CALIBRATION_DURATION_MS) * 100));
          setCalibProgress(pct);

          if (elapsed >= CALIBRATION_DURATION_MS) {
            isCalibrationDoneRef.current = true;
            finalizeCalibration(calibrationSamplesRef.current);
            setStatus(resized.length === 0 ? "no-face" : "detecting");
          }
        } else {
          setStatus(resized.length === 0 ? "no-face" : "detecting");
        }

        // Multi-face spatial tracking and per-face metrics
        const updatedTrackedFaces: TrackedFace[] = [];
        const existingMap = trackedFacesRef.current;
        const matchedExistingIds = new Set<number>();

        for (const face of resized) {
          const box = face.detection.box;
          const cx = box.x + box.width / 2;
          const cy = box.y + box.height / 2;

          // Find closest unmatched existing tracked face
          let bestId: number | null = null;
          let bestDist = Infinity;
          const matchThreshold = Math.max(box.width, box.height) * 1.2;

          for (const [id, tracked] of existingMap.entries()) {
            if (matchedExistingIds.has(id)) continue;
            const tcx = tracked.box.x + tracked.box.width / 2;
            const tcy = tracked.box.y + tracked.box.height / 2;
            const dist = Math.hypot(cx - tcx, cy - tcy);
            if (dist < matchThreshold && dist < bestDist) {
              bestDist = dist;
              bestId = id;
            }
          }

          let tracker: TrackedFace;
          if (bestId !== null) {
            matchedExistingIds.add(bestId);
            tracker = existingMap.get(bestId)!;
          } else {
            const newId = nextFaceIdRef.current++;
            tracker = {
              id: newId,
              box: { x: box.x, y: box.y, width: box.width, height: box.height },
              score: face.detection.score,
              detection: face.detection,
              landmarks: face.landmarks,
              expressions: face.expressions,
              dominantEmotion: "neutral",
              stressScore: 0,
              stressPct: 0,
              stressColor: "#22c55e",
              ear: 0.3,
              openEarBaseline: 0.30,
              isEyeClosed: false,
              eyeClosedStartTime: 0,
              blinkCount: 0,
              recentBlinks: [],
              blinkRate: 0,
              lastSeen: nowMs,
              stressHighSince: null,
              stressDurationSec: 0,
              isCalmAlertActive: false,
            };
            existingMap.set(newId, tracker);
          }

          // Update face position and basic properties
          tracker.box = { x: box.x, y: box.y, width: box.width, height: box.height };
          tracker.score = face.detection.score;
          tracker.detection = face.detection;
          tracker.landmarks = face.landmarks;
          tracker.expressions = face.expressions;
          tracker.lastSeen = nowMs;

          // Per-face Emotion & Stress detection
          const dom = dominantEmotion(face.expressions);
          const stress = mapEmotionToStress(face.expressions);
          tracker.dominantEmotion = dom;
          tracker.stressScore = stress;
          tracker.stressPct = Math.round(stress * 100);

          const currentThreshPct = stressThresholdPctRef.current;
          tracker.stressColor =
            tracker.isCalmAlertActive
              ? "#00f5d4"
              : tracker.stressPct >= currentThreshPct
              ? "#ef4444"
              : tracker.stressPct >= 40
              ? "#f97316"
              : "#22c55e";

          // Prolonged stress tracking per face
          if (tracker.stressPct >= currentThreshPct) {
            if (tracker.stressHighSince === null) {
              tracker.stressHighSince = nowMs;
            }
            tracker.stressDurationSec = Math.round(((nowMs - tracker.stressHighSince) / 1000) * 10) / 10;
          } else {
            if (tracker.stressHighSince !== null && tracker.isCalmAlertActive) {
              setCalmRestoredMsg(
                `Face #${tracker.id} returned to calm state (${tracker.stressPct}% < ${currentThreshPct}%). Great job!`
              );
              setTimeout(() => setCalmRestoredMsg(null), 6000);
            }
            tracker.stressHighSince = null;
            tracker.stressDurationSec = 0;
            tracker.isCalmAlertActive = false;
          }

          // Per-face EAR & Blink detection
          if (face.landmarks) {
            const pts = face.landmarks.positions;
            const leftEye = pts.slice(36, 42);
            const rightEye = pts.slice(42, 48);
            const ear = (eyeAspectRatio(leftEye) + eyeAspectRatio(rightEye)) / 2;
            tracker.ear = ear;

            // Adaptive open EAR baseline
            if (ear >= 0.23) {
              tracker.openEarBaseline = tracker.openEarBaseline * 0.92 + ear * 0.08;
            }

            // Dynamic EAR threshold: adapts to user eyes and lighting
            const earThresh = Math.max(0.22, Math.min(0.26, tracker.openEarBaseline * 0.80));
            const isClosed = ear < earThresh;

            if (isClosed) {
              if (!tracker.isEyeClosed) {
                tracker.isEyeClosed = true;
                tracker.eyeClosedStartTime = nowMs;
              }
            } else {
              if (tracker.isEyeClosed) {
                const duration = nowMs - tracker.eyeClosedStartTime;
                tracker.isEyeClosed = false;
                // Valid blink closure duration: 50ms - 1000ms
                if (duration >= 50 && duration <= 1000) {
                  tracker.blinkCount++;
                  tracker.recentBlinks.push(nowMs);
                }
              }
            }

            // Filter recent blinks to last 60 seconds
            tracker.recentBlinks = tracker.recentBlinks.filter((t) => nowMs - t <= 60000);

            // Compute rolling blink rate per minute
            const sessionElapsedSec = (nowMs - blinkStartTimeRef.current) / 1000;
            if (sessionElapsedSec >= 60) {
              tracker.blinkRate = tracker.recentBlinks.length;
            } else if (sessionElapsedSec >= 8) {
              tracker.blinkRate = Math.round((tracker.recentBlinks.length / sessionElapsedSec) * 60);
            } else {
              tracker.blinkRate =
                tracker.recentBlinks.length > 0
                  ? Math.round((tracker.recentBlinks.length / sessionElapsedSec) * 60)
                  : 0;
            }
          }

          updatedTrackedFaces.push(tracker);
        }

        // Clean up faces lost for > 2 seconds
        for (const [id, t] of existingMap.entries()) {
          if (nowMs - t.lastSeen > 2000) {
            existingMap.delete(id);
          }
        }

        // Sort by bounding box area (largest face = primary user)
        updatedTrackedFaces.sort(
          (a, b) => b.box.width * b.box.height - a.box.width * a.box.height
        );
        latestTrackedFacesRef.current = updatedTrackedFaces;

        // Multi-face summaries for UI state
        setMultiFaceSummaries(
          updatedTrackedFaces.map((f) => ({
            id: f.id,
            emotion: f.dominantEmotion,
            stressPct: f.stressPct,
            blinkRate: f.blinkRate,
          }))
        );

        if (updatedTrackedFaces.length > 0) {
          const primary = updatedTrackedFaces[0];

          latestEmotionRef.current = primary.dominantEmotion;
          latestStressRef.current = primary.stressScore;
          latestBlinkRateRef.current = primary.blinkRate;
          blinkCountRef.current = primary.blinkCount;
          eyeClosedRef.current = primary.isEyeClosed;

          setEmotion(primary.dominantEmotion);
          setStressScore(Math.round(primary.stressScore * 100) / 100);
          setBlinkRate(primary.blinkRate);
          setBlinkCount(primary.blinkCount);

          // KNN descriptor for primary face
          const firstResized = resized[0];
          if (firstResized?.descriptor) {
            const desc = Array.from(firstResized.descriptor);
            latestDescriptorRef.current = desc;

            if (knnSamplesRef.current.length >= KNN_K * 2) {
              const pred = knnPredict(knnSamplesRef.current, desc);
              setKnnPrediction(pred);
            }

            if (trainPhaseRef.current !== "idle" && trainPhaseRef.current !== "done") {
              const label = trainPhaseRef.current as TrainLabel;
              if (trainingCountRef.current < TRAIN_SAMPLES_NEEDED) {
                trainingCountRef.current++;
                knnSamplesRef.current.push({ descriptor: desc, label });
                setTrainCount(trainingCountRef.current);
                if (trainingCountRef.current >= TRAIN_SAMPLES_NEEDED) {
                  trainPhaseRef.current = "done";
                  setTrainPhase("done");
                }
              }
            }
          }

          // Calibration sample collection
          if (!isCalibrationDoneRef.current) {
            calibrationSamplesRef.current.push({
              stress: primary.stressScore,
              blinkRate: primary.blinkRate,
            });
          }

          // Alerts with threshold and duration check
          const targetDurationSec = stressDurationThresholdSecRef.current;
          const targetThreshPct = stressThresholdPctRef.current;
          const blinkThresh = blinkThresholdRef.current;
          const cooldownOk = nowMs - lastAlertTimeRef.current >= ALERT_COOLDOWN_MS;

          // Find if any face has sustained stress >= targetDurationSec
          const prolongedFace = updatedTrackedFaces.find(
            (f) => f.stressDurationSec >= targetDurationSec
          );

          if (prolongedFace) {
            prolongedFace.isCalmAlertActive = true;
            setAlertStress(true);

            if (!alertFiredRef.current && cooldownOk) {
              alertFiredRef.current = true;
              lastAlertTimeRef.current = nowMs;

              const alertInfo: CalmAlertDetails = {
                faceId: prolongedFace.id,
                stressPct: prolongedFace.stressPct,
                durationSec: prolongedFace.stressDurationSec,
                thresholdPct: targetThreshPct,
                thresholdSec: targetDurationSec,
                timestamp: nowMs,
              };
              setCalmAlertDetails(alertInfo);
              setShowBreathingGuide(true);
              setStressProgress(null);

              // Soothing chime sound
              if (soundAlertModeRef.current === "chime") {
                playCalmChime();
              } else if (soundAlertModeRef.current === "beep") {
                playBeep(880, 0.3);
              }

              // Spoken voice reminder
              if (voiceAlertEnabledRef.current) {
                const label = updatedTrackedFaces.length > 1 ? `Face number ${prolongedFace.id}` : "You";
                speakCalmReminder(`${label}, please take a deep breath and stay calm.`);
              }

              // Browser system push notification
              sendBrowserNotification(
                "🌿 FaceSense: Reminder to Be Calm",
                `High stress (${prolongedFace.stressPct}%) sustained for ${prolongedFace.stressDurationSec}s. Take a slow deep breath and relax.`
              );

              // Persist alert to backend
              persistAlert(
                BACKEND_URL,
                userIdRef.current,
                sessionIdRef.current,
                "stress",
                prolongedFace.stressScore,
                prolongedFace.blinkRate,
                prolongedFace.stressDurationSec,
                `High stress for ${prolongedFace.stressDurationSec}s (Threshold: ${targetThreshPct}%) — Reminder to be calm`
              );
            }
          } else {
            // Check if any face has elevated stress accumulating towards duration threshold
            const maxElevated = updatedTrackedFaces.reduce<TrackedFace | null>((best, f) => {
              if (f.stressDurationSec > 0 && (!best || f.stressDurationSec > best.stressDurationSec)) {
                return f;
              }
              return best;
            }, null);

            if (maxElevated && maxElevated.stressDurationSec >= 1.5) {
              setStressProgress({
                active: true,
                faceId: maxElevated.id,
                durationSec: maxElevated.stressDurationSec,
                thresholdSec: targetDurationSec,
                stressPct: maxElevated.stressPct,
              });
            } else {
              setStressProgress(null);
            }

            // Check if all faces have calmed down
            const anyStressHigh = updatedTrackedFaces.some((f) => f.stressHighSince !== null);
            if (!anyStressHigh) {
              setAlertStress(false);
              alertFiredRef.current = false;
            }
          }

          const elapsed30s = nowMs - blinkStartTimeRef.current > 30000;
          if (elapsed30s && primary.blinkRate < blinkThresh && primary.blinkRate > 0) {
            if (!alertBlinkRef.current) {
              alertBlinkRef.current = true;
              sendBrowserNotification(
                "👁 Low Blink Rate Detected",
                `Your blink rate is ${primary.blinkRate}/min. Look away and blink more often.`
              );
              persistAlert(
                BACKEND_URL,
                userIdRef.current,
                sessionIdRef.current,
                "fatigue",
                primary.stressScore,
                primary.blinkRate
              );
            }
            setAlertBlink(true);
          } else if (primary.blinkRate >= blinkThresh) {
            alertBlinkRef.current = false;
            setAlertBlink(false);
          }
        } else {
          latestEmotionRef.current = "neutral";
          latestStressRef.current = 0;
          latestDescriptorRef.current = null;
          eyeClosedRef.current = false;
          setEmotion("neutral");
          setStressScore(0);
          stressHighSinceRef.current = null;
          alertFiredRef.current = false;
          setAlertStress(false);
          setKnnPrediction(null);
        }
      } catch {
        // Skip failed frames silently
      } finally {
        isProcessing = false;
      }
    }, 100);
  }, [finalizeCalibration]);

  // ── API batch loop — flush every 5s 
  const startApiLoop = useCallback(() => {
    apiIntervalRef.current = setInterval(() => {
      if (!isRunningRef.current) return;

      // Add current snapshot to buffer (even if no face — rate = 0)
      if (userIdRef.current && sessionIdRef.current) {
        batchBufferRef.current.push({
          userId: userIdRef.current,
          sessionId: sessionIdRef.current,
          emotion: latestEmotionRef.current,
          stressScore: latestStressRef.current,
          blinkRate: latestBlinkRateRef.current,
          timestamp: new Date().toISOString(),
        });
      }

      // Flush buffer
      if (batchBufferRef.current.length > 0) {
        flushBatch(batchBufferRef.current.splice(0));
      }
    }, BATCH_INTERVAL_MS);
  }, [flushBatch]);

  // ── Training phase ref (needs to be readable inside interval) ─
  const trainPhaseRef = useRef<TrainPhase>("idle");

  // ── Start training for a label 
  const startTraining = useCallback((label: TrainLabel) => {
    trainingCountRef.current = 0;
    trainPhaseRef.current = label;
    setTrainPhase(label);
    setTrainCount(0);
  }, []);

  // ── Save trained model to backend ──
  const saveModel = useCallback(async () => {
    if (knnSamplesRef.current.length < KNN_K * 2) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/usermodel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: userIdRef.current,
          samples: knnSamplesRef.current,
        }),
      });
      if (res.ok) {
        setModelReady(true);
        setTrainPhase("idle");
        trainPhaseRef.current = "idle";
      }
    } catch {
      // offline — model stays in-memory only
      setModelReady(knnSamplesRef.current.length >= KNN_K * 2);
    }
  }, []);

  // ── Clear model 
  const clearModel = useCallback(() => {
    knnSamplesRef.current = [];
    setModelReady(false);
    setKnnPrediction(null);
    setTrainPhase("idle");
    trainPhaseRef.current = "idle";
    fetch(`${BACKEND_URL}/api/usermodel/${userIdRef.current}`, { method: "DELETE" }).catch(() => { });
  }, []);

  // ── Draw loop ──
  const startDrawLoop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

    function draw() {
      if (!isRunningRef.current || !canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const trackedFaces = latestTrackedFacesRef.current;
      trackedFaces.forEach((face) => {
        const box = face.box;
        const score = face.score;

        // Bounding box border and glow
        if (face.isCalmAlertActive) {
          ctx.strokeStyle = "#00f5d4";
          ctx.lineWidth = 3;
          ctx.shadowColor = "#00f5d4";
          ctx.shadowBlur = 18;
          ctx.strokeRect(box.x, box.y, box.width, box.height);
          ctx.shadowBlur = 0;
          drawCorners(ctx, box.x, box.y, box.width, box.height, 16, "#00f5d4");
        } else {
          ctx.strokeStyle = face.stressColor;
          ctx.lineWidth = 2;
          ctx.shadowColor = face.stressColor;
          ctx.shadowBlur = 10;
          ctx.strokeRect(box.x, box.y, box.width, box.height);
          ctx.shadowBlur = 0;
          drawCorners(ctx, box.x, box.y, box.width, box.height, 16, face.stressColor);
        }

        // Facial landmarks
        const pts = face.landmarks.positions;
        drawLandmarkGroup(ctx, pts.slice(0, 17), "#4cc9f0", false);
        drawLandmarkGroup(ctx, pts.slice(17, 22), "#f72585", false);
        drawLandmarkGroup(ctx, pts.slice(22, 27), "#f72585", false);
        drawLandmarkGroup(ctx, pts.slice(27, 31), "#4cc9f0", false);
        drawLandmarkGroup(ctx, pts.slice(31, 36), "#4cc9f0", false);
        const eyeColor = face.isEyeClosed ? "#facc15" : "#7209b7";
        drawLandmarkGroup(ctx, pts.slice(36, 42), eyeColor, true);
        drawLandmarkGroup(ctx, pts.slice(42, 48), eyeColor, true);
        drawLandmarkGroup(ctx, pts.slice(48, 60), "#f72585", true);
        drawLandmarkGroup(ctx, pts.slice(60, 68), "#f72585", true);

        // Respective Bounding Box HUD:
        // We invert horizontal coordinate to counter CSS scaleX(-1) on canvas
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-canvas.width, 0);

        const screenBoxLeft = canvas.width - box.x - box.width;
        const screenBoxWidth = box.width;
        const cardWidth = Math.max(screenBoxWidth, 240);
        const cardHeight = 48;
        const cardX = Math.max(
          8,
          Math.min(canvas.width - cardWidth - 8, screenBoxLeft + (screenBoxWidth - cardWidth) / 2)
        );
        const cardY = box.y >= 56 ? box.y - 52 : box.y + 6;

        // If face is in active Calm Alert or elevated, draw a top reminder pill
        if (face.isCalmAlertActive) {
          drawRoundedRect(ctx, cardX, cardY - 20, cardWidth, 18, 4);
          ctx.fillStyle = "#00f5d4";
          ctx.fill();
          ctx.font = "bold 10px monospace";
          ctx.fillStyle = "#0a0a0f";
          ctx.fillText("🌿 REMINDER: BE CALM · BREATHE", cardX + 8, cardY - 7);
        } else if (face.stressDurationSec >= 1.5) {
          drawRoundedRect(ctx, cardX, cardY - 18, cardWidth, 16, 4);
          ctx.fillStyle = "rgba(249, 115, 22, 0.95)";
          ctx.fill();
          ctx.font = "bold 9px monospace";
          ctx.fillStyle = "#0a0a0f";
          ctx.fillText(
            `⏱ ELEVATED: ${face.stressDurationSec}s / ${stressDurationThresholdSecRef.current}s`,
            cardX + 8,
            cardY - 6
          );
        }

        // HUD Background Card
        drawRoundedRect(ctx, cardX, cardY, cardWidth, cardHeight, 6);
        ctx.fillStyle = face.isCalmAlertActive ? "rgba(10, 28, 24, 0.94)" : "rgba(10, 12, 20, 0.90)";
        ctx.fill();
        ctx.strokeStyle = face.isCalmAlertActive ? "#00f5d4" : face.stressColor + "aa";
        ctx.lineWidth = face.isCalmAlertActive ? 2 : 1.5;
        ctx.stroke();

        // Row 1: Face #, Dominant Emotion & Emoji, Confidence
        ctx.font = "bold 11px monospace";
        ctx.fillStyle = face.isCalmAlertActive ? "#00f5d4" : "#00f5d4";
        ctx.fillText(`FACE #${face.id}`, cardX + 8, cardY + 17);

        const emEmoji = EMOTION_EMOJI[face.dominantEmotion] ?? "😐";
        const emName = face.dominantEmotion.toUpperCase();
        ctx.fillStyle = EMOTION_COLORS[face.dominantEmotion] ?? "#ffffff";
        ctx.fillText(`${emEmoji} ${emName}`, cardX + 66, cardY + 17);

        ctx.font = "10px monospace";
        ctx.fillStyle = "#94a3b8";
        const confText = `${(score * 100).toFixed(0)}%`;
        const confW = ctx.measureText(confText).width;
        ctx.fillText(confText, cardX + cardWidth - confW - 8, cardY + 17);

        // Row 2: Stress level & Blink rate
        ctx.font = "bold 11px monospace";
        ctx.fillStyle = face.isCalmAlertActive ? "#00f5d4" : face.stressColor;
        const stressText = `STRESS: ${face.stressPct}%`;
        ctx.fillText(stressText, cardX + 8, cardY + 36);

        const stressTag = face.isCalmAlertActive
          ? "BE CALM"
          : face.stressPct >= stressThresholdPctRef.current
          ? "HIGH"
          : face.stressPct >= 40
          ? "MOD"
          : "CALM";
        ctx.font = "9px monospace";
        ctx.fillStyle = face.isCalmAlertActive ? "#00f5d4" : face.stressColor + "aa";
        ctx.fillText(`[${stressTag}]`, cardX + 8 + ctx.measureText(stressText).width + 4, cardY + 36);

        // Blink rate
        ctx.font = "11px monospace";
        ctx.fillStyle = "#38bdf8";
        const blinkRateText = `👁 ${face.blinkRate}/m`;
        const blinkW = ctx.measureText(blinkRateText).width;
        const blinkRightOffset = face.isEyeClosed ? 56 : 8;
        ctx.fillText(blinkRateText, cardX + cardWidth - blinkW - blinkRightOffset, cardY + 36);

        // If eyes closed in a blink, show active BLINK badge
        if (face.isEyeClosed) {
          const blinkBadgeX = cardX + cardWidth - 48;
          drawRoundedRect(ctx, blinkBadgeX, cardY + 24, 40, 16, 4);
          ctx.fillStyle = "#facc15";
          ctx.fill();
          ctx.fillStyle = "#000000";
          ctx.font = "bold 9px monospace";
          ctx.fillText("BLINK", blinkBadgeX + 6, cardY + 36);
        }

        ctx.restore();
      });

      const now = performance.now();
      fpsRef.current.frames++;
      if (now - fpsRef.current.last >= 1000) {
        setFps(fpsRef.current.frames);
        fpsRef.current.frames = 0;
        fpsRef.current.last = now;
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);
  }, []);

  useEffect(() => { return () => { stopCamera(); }; }, [stopCamera]);

  const isRunning = status === "detecting" || status === "no-face" || status === "calibrating";
  const isLoading = status === "loading-models" || status === "requesting-camera";
  const isCalibrating = status === "calibrating";

  const stressPct = Math.round(stressScore * 100);
  const stressColor =
    stressPct >= Math.round(stressThresholdRef.current * 100)
      ? "#ef4444"
      : stressPct >= 40 ? "#f97316" : "#22c55e";

  // ── Render ──
  return (
    <main className="min-h-screen bg-[#0a0a0f] text-white flex flex-col items-center justify-start py-10 px-4">

      {/* Header */}
      <header className="mb-8 text-center select-none w-full max-w-5xl">
        <div className="flex items-center justify-between mb-2">
          <div className="flex-1" />
          <div className="flex items-center justify-center gap-3">
            <div className="w-2 h-2 rounded-full bg-[#00f5d4] animate-pulse shadow-[0_0_8px_#00f5d4]" />
            <h1 className="text-3xl font-black tracking-[0.15em] uppercase font-mono">
              FACE<span className="text-[#00f5d4]">SENSE</span>
            </h1>
            <div className="w-2 h-2 rounded-full bg-[#00f5d4] animate-pulse shadow-[0_0_8px_#00f5d4]" />
          </div>
          <div className="flex-1 flex justify-end">
            <Link
              href="/dashboard"
              className="px-3 py-1.5 text-xs font-mono font-bold rounded-lg bg-[#00f5d4]/10 border border-[#00f5d4]/30 text-[#00f5d4] hover:bg-[#00f5d4]/20 transition-all"
            >
              📊 Dashboard
            </Link>
          </div>
        </div>
        <p className="text-xs text-zinc-600 tracking-widest uppercase font-mono">
          Phase 4 — Notifications · Analytics Dashboard · Production Ready
        </p>
        {userId && (
          <p className="text-[10px] text-zinc-700 font-mono mt-1">
            User: {userId.slice(0, 8)}…
            {notifPermission === "granted" && (
              <span className="ml-2 text-emerald-700">🔔 notifications on</span>
            )}
            {notifPermission === "denied" && (
              <span className="ml-2 text-zinc-700">🔕 notifications blocked</span>
            )}
          </p>
        )}
      </header>

      {/* Calibration bar */}
      {isCalibrating && (
        <div className="mb-4 w-full max-w-5xl">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-mono text-purple-400">Calibrating baseline…</span>
            <span className="text-xs font-mono text-purple-300">{calibProgress}%</span>
          </div>
          <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{ width: `${calibProgress}%`, backgroundColor: "#a855f7" }}
            />
          </div>
          <p className="text-[10px] font-mono text-zinc-600 mt-1">
            Sit relaxed — we are learning your baseline stress and blink patterns
          </p>
        </div>
      )}

      {/* Calm Restored Toast */}
      {calmRestoredMsg && (
        <div className="mb-4 w-full max-w-5xl flex items-center justify-between gap-3 bg-emerald-950/80 border border-emerald-500/50 rounded-xl px-4 py-3 text-emerald-300 text-sm font-mono shadow-[0_0_20px_rgba(16,185,129,0.2)] animate-pulse">
          <div className="flex items-center gap-3">
            <span className="text-xl">🌿</span>
            <span className="font-semibold">{calmRestoredMsg}</span>
          </div>
          <button
            onClick={() => setCalmRestoredMsg(null)}
            className="text-emerald-400 hover:text-white text-xs px-2 py-1 rounded"
          >
            ✕
          </button>
        </div>
      )}

      {/* Elevated Stress Accumulation Progress */}
      {stressProgress && !alertStress && (
        <div className="mb-4 w-full max-w-5xl bg-amber-950/40 border border-amber-500/50 rounded-xl p-3 flex flex-col gap-1.5 backdrop-blur-sm shadow-[0_0_25px_rgba(245,158,11,0.15)]">
          <div className="flex items-center justify-between text-xs font-mono">
            <div className="flex items-center gap-2 text-amber-300">
              <span className="animate-spin text-sm">⏳</span>
              <span>
                <strong>Face #{stressProgress.faceId}</strong> elevated stress ({stressProgress.stressPct}% ≥ {stressThresholdPct}%)
              </span>
            </div>
            <span className="text-amber-400 font-bold tabular-nums">
              {stressProgress.durationSec.toFixed(1)}s / {stressProgress.thresholdSec}s until Calm Alert
            </span>
          </div>
          <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-amber-500 to-red-500 transition-all duration-150"
              style={{
                width: `${Math.min(100, (stressProgress.durationSec / stressProgress.thresholdSec) * 100)}%`,
              }}
            />
          </div>
          <p className="text-[10px] font-mono text-zinc-400">
            Slow down your breathing — if stress drops below {stressThresholdPct}%, this timer resets.
          </p>
        </div>
      )}

      {/* Prolonged Stress & Calm Reminder Alert */}
      {alertStress && (
        <div className="mb-4 w-full max-w-5xl bg-gradient-to-r from-emerald-950/90 via-zinc-900/95 to-cyan-950/90 border-2 border-[#00f5d4] rounded-2xl p-5 shadow-[0_0_50px_rgba(0,245,212,0.25)] relative overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-400 via-[#00f5d4] to-cyan-400 animate-pulse" />

          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-2xl bg-[#00f5d4]/10 border border-[#00f5d4]/40 flex items-center justify-center text-2xl flex-shrink-0 animate-bounce">
                🌿
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#00f5d4]/20 text-[#00f5d4] border border-[#00f5d4]/40 uppercase tracking-wider">
                    Reminder to Be Calm
                  </span>
                  {calmAlertDetails && (
                    <span className="text-xs font-mono text-zinc-400">
                      Face #{calmAlertDetails.faceId} · Elevated for {calmAlertDetails.durationSec}s
                    </span>
                  )}
                </div>
                <h2 className="text-xl font-bold font-mono text-white mt-1">
                  Please Take a Deep Breath and Stay Calm
                </h2>
                <p className="text-xs font-mono text-zinc-300 mt-1 max-w-2xl leading-relaxed">
                  Your stress level ({stressPct}%) has exceeded your threshold ({stressThresholdPct}%) for longer than {stressDurationThresholdSec} seconds. Lower your shoulders, relax your facial muscles, and take slow, calm breaths.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 flex-shrink-0">
              <button
                onClick={() => setShowBreathingGuide((prev) => !prev)}
                className="px-4 py-2 rounded-xl text-xs font-mono font-bold bg-[#00f5d4] text-[#0a0a0f] hover:bg-[#00f5d4]/80 transition-all shadow-[0_0_20px_rgba(0,245,212,0.3)] flex items-center gap-1.5"
              >
                <span>🧘</span>
                <span>{showBreathingGuide ? "Hide Breathing Guide" : "Guided Breathing"}</span>
              </button>
              <button
                onClick={() => {
                  playCalmChime();
                  if (voiceAlertEnabled) speakCalmReminder();
                }}
                title="Play soothing chime & voice reminder"
                className="px-3 py-2 rounded-xl text-xs font-mono border border-zinc-700 bg-zinc-800/80 text-zinc-300 hover:bg-zinc-700 hover:text-white transition-all flex items-center gap-1.5"
              >
                <span>🔔</span>
                <span>Chime</span>
              </button>
              <button
                onClick={() => {
                  setAlertStress(false);
                  setCalmAlertDetails(null);
                  setShowBreathingGuide(false);
                }}
                className="px-3 py-2 rounded-xl text-xs font-mono border border-zinc-700 bg-zinc-800/80 text-zinc-400 hover:text-white transition-all"
              >
                Dismiss
              </button>
            </div>
          </div>

          {/* Interactive Guided Box Breathing */}
          {showBreathingGuide && (
            <GuidedBreathingWidget currentStress={stressPct} targetThreshold={stressThresholdPct} />
          )}
        </div>
      )}

      {/* Blink Alert */}
      {alertBlink && (
        <div className="mb-4 w-full max-w-5xl flex items-center gap-3 bg-amber-950/60 border border-amber-500/50 rounded-lg px-4 py-3 text-amber-400 text-sm font-mono">
          <span className="text-lg">👁</span>
          <span>LOW BLINK RATE ({blinkRate}/min) — blink more often to reduce eye strain.</span>
        </div>
      )}

      {/* Main layout */}
      <div className="w-full max-w-5xl flex flex-col xl:flex-row gap-4">

        {/* Video + Canvas */}
        <div className="relative flex-1">
          <div className={`rounded-xl overflow-hidden border transition-all duration-500 ${isRunning
            ? alertStress
              ? "border-red-500/50 shadow-[0_0_40px_rgba(239,68,68,0.2)]"
              : isCalibrating
                ? "border-purple-500/40 shadow-[0_0_40px_rgba(168,85,247,0.15)]"
                : "border-[#00f5d4]/30 shadow-[0_0_40px_rgba(0,245,212,0.1)]"
            : "border-zinc-800"
            }`}>
            <video
              ref={videoRef}
              autoPlay muted playsInline
              className="w-full block bg-zinc-950"
              style={{ transform: "scaleX(-1)", minHeight: "360px" }}
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
              style={{ transform: "scaleX(-1)" }}
            />

            {!isRunning && !isLoading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/95 min-h-[360px]">
                <CameraOffIcon />
                <p className="mt-4 text-zinc-600 text-sm font-mono tracking-widest uppercase">
                  Press &ldquo;Start Camera&rdquo; to begin
                </p>
              </div>
            )}

            {isLoading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/85 min-h-[360px]">
                <Spinner />
                <p className="mt-4 text-amber-400 text-sm font-mono tracking-widest animate-pulse">
                  {STATUS_LABELS[status]}
                </p>
              </div>
            )}
          </div>

          {isRunning && (
            <div className="absolute top-3 right-3 flex flex-col gap-1.5 items-end pointer-events-none">
              <HudBadge color="#00f5d4" label="FACES" value={String(faceCount)} />
              <HudBadge color="#7209b7" label="FPS" value={String(fps)} />
              {modelReady && knnPrediction && (
                <HudBadge
                  color={knnPrediction === "stressed" ? "#ef4444" : "#22c55e"}
                  label="KNN"
                  value={knnPrediction.toUpperCase()}
                />
              )}
            </div>
          )}

          {isRunning && (
            <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-black/60 border border-red-500/30 rounded px-2 py-1 pointer-events-none backdrop-blur-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
              <span className="text-red-400 text-xs font-mono font-bold tracking-widest">LIVE</span>
            </div>
          )}
        </div>

        {/* ── Analytics Panel  */}
        {isRunning && (
          <div className="xl:w-64 flex flex-col gap-3">

            {/* Emotion card */}
            <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-4">
              <p className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest mb-2">Emotion</p>
              <div className="flex items-center gap-3">
                <span className="text-3xl">{EMOTION_EMOJI[emotion] ?? "😐"}</span>
                <div>
                  <p className="text-lg font-bold font-mono capitalize" style={{ color: EMOTION_COLORS[emotion] ?? "#fff" }}>
                    {emotion}
                  </p>
                  <p className="text-xs text-zinc-600 font-mono">dominant</p>
                </div>
              </div>
            </div>

            {/* Stress card */}
            <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <p className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest">Stress Level</p>
                <p className="text-sm font-bold font-mono" style={{ color: stressColor }}>{stressPct}%</p>
              </div>
              <div className="w-full h-2.5 bg-zinc-800 rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all duration-500" style={{ width: `${stressPct}%`, backgroundColor: stressColor }} />
              </div>
              <div className="flex justify-between mt-1">
                <p className="text-[10px] font-mono text-zinc-600">
                  {stressPct >= Math.round(stressThresholdRef.current * 100) ? "HIGH — take a break" : stressPct >= 40 ? "MODERATE" : "LOW — calm"}
                </p>
                {baseline && (
                  <p className="text-[10px] font-mono text-purple-500">
                    thresh: {Math.round(stressThresholdRef.current * 100)}%
                  </p>
                )}
              </div>
            </div>

            {/* Blink card */}
            <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-4">
              <p className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest mb-3">Blink Stats</p>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-zinc-950/60 rounded-lg p-2 text-center">
                  <p className="text-lg font-bold font-mono text-[#00f5d4]">{blinkRate}</p>
                  <p className="text-[10px] font-mono text-zinc-600">/min</p>
                </div>
                <div className="bg-zinc-950/60 rounded-lg p-2 text-center">
                  <p className="text-lg font-bold font-mono text-[#7209b7]">{blinkCount}</p>
                  <p className="text-[10px] font-mono text-zinc-600">total</p>
                </div>
              </div>
              {baseline && (
                <p className="text-[10px] font-mono text-purple-500 mt-1">
                  threshold: {Math.round(blinkThresholdRef.current)}/min
                </p>
              )}
              {alertBlink && <p className="text-[10px] font-mono text-amber-400 mt-1">⚠ Rate too low</p>}
            </div>

            {/* Multi-face overview when > 1 face */}
            {multiFaceSummaries.length > 1 && (
              <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-3">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest">
                    Multi-Face ({multiFaceSummaries.length})
                  </p>
                  <span className="text-[10px] font-mono text-[#00f5d4] bg-[#00f5d4]/10 px-1.5 py-0.5 rounded border border-[#00f5d4]/30">
                    Live
                  </span>
                </div>
                <div className="flex flex-col gap-1.5 max-h-48 overflow-y-auto pr-1">
                  {multiFaceSummaries.map((f) => (
                    <div
                      key={f.id}
                      className="flex items-center justify-between p-2 rounded-lg bg-zinc-950/60 border border-zinc-800 text-xs font-mono"
                    >
                      <div className="flex items-center gap-1.5">
                        <span className="text-zinc-500 text-[10px] font-bold">#{f.id}</span>
                        <span>{EMOTION_EMOJI[f.emotion] ?? "😐"}</span>
                        <span
                          className="capitalize font-semibold text-[11px]"
                          style={{ color: EMOTION_COLORS[f.emotion] ?? "#fff" }}
                        >
                          {f.emotion}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span
                          className="font-bold text-[11px]"
                          style={{
                            color:
                              f.stressPct >= 70
                                ? "#ef4444"
                                : f.stressPct >= 40
                                ? "#f97316"
                                : "#22c55e",
                          }}
                        >
                          {f.stressPct}%
                        </span>
                        <span className="text-[#38bdf8] text-[10px]">
                          {f.blinkRate}/m
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* KNN prediction card */}
            {modelReady && (
              <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-3">
                <p className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest mb-1">KNN Prediction</p>
                {knnPrediction ? (
                  <p className="text-sm font-bold font-mono" style={{ color: knnPrediction === "stressed" ? "#ef4444" : "#22c55e" }}>
                    {knnPrediction === "stressed" ? "😰 STRESSED" : "😌 RELAXED"}
                  </p>
                ) : (
                  <p className="text-xs font-mono text-zinc-600">waiting for face…</p>
                )}
              </div>
            )}

            {/* Backend status */}
            <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-3 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full flex-shrink-0" style={{
                backgroundColor: backendOk === null ? "#6b7280" : backendOk ? "#22c55e" : "#ef4444"
              }} />
              <div>
                <p className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest">Backend</p>
                <p className="text-xs font-mono text-zinc-400">
                  {backendOk === null ? "waiting…" : backendOk ? "connected ✓" : "offline (data not saved)"}
                </p>
              </div>
            </div>

          </div>
        )}
      </div>

      {/* Training panel toggle */}
      {isRunning && !isCalibrating && (
        <div className="mt-4 w-full max-w-5xl">
          <button
            onClick={() => setShowTrainingPanel(!showTrainingPanel)}
            className="text-xs font-mono text-purple-400 border border-purple-500/30 rounded px-3 py-1.5 hover:bg-purple-500/10 transition-colors"
          >
            {showTrainingPanel ? "▲ Hide" : "▼ Show"} Training Panel
          </button>

          {showTrainingPanel && (
            <div className="mt-3 bg-zinc-900/70 border border-zinc-800 rounded-xl p-4">
              <p className="text-xs font-mono text-zinc-400 font-bold mb-3 uppercase tracking-widest">
                KNN Training Mode
              </p>

              {trainPhase === "idle" && (
                <div className="flex flex-col gap-3">
                  <p className="text-xs font-mono text-zinc-500">
                    Train a personalized model. Make a relaxed face, then a stressed face.
                    The system captures {TRAIN_SAMPLES_NEEDED} samples per label.
                  </p>
                  <div className="flex gap-3">
                    <button
                      onClick={() => startTraining("relaxed")}
                      className="px-4 py-2 text-xs font-mono font-bold rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20 transition-colors"
                    >
                      😌 Start Relaxed
                    </button>
                    <button
                      onClick={() => startTraining("stressed")}
                      className="px-4 py-2 text-xs font-mono font-bold rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 transition-colors"
                    >
                      😰 Start Stressed
                    </button>
                    {modelReady && (
                      <button
                        onClick={clearModel}
                        className="px-4 py-2 text-xs font-mono font-bold rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-400 hover:bg-zinc-700 transition-colors"
                      >
                        🗑 Clear Model
                      </button>
                    )}
                  </div>
                  {modelReady && (
                    <p className="text-xs font-mono text-emerald-400">
                      ✓ Model ready with {knnSamplesRef.current.length} samples
                    </p>
                  )}
                </div>
              )}

              {(trainPhase === "relaxed" || trainPhase === "stressed") && (
                <div>
                  <p className="text-xs font-mono text-zinc-400 mb-2">
                    Capturing <span className="font-bold" style={{ color: trainPhase === "relaxed" ? "#22c55e" : "#ef4444" }}>
                      {trainPhase.toUpperCase()}
                    </span> samples — hold expression naturally
                  </p>
                  <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden mb-2">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${Math.round((trainCount / TRAIN_SAMPLES_NEEDED) * 100)}%`,
                        backgroundColor: trainPhase === "relaxed" ? "#22c55e" : "#ef4444",
                      }}
                    />
                  </div>
                  <p className="text-xs font-mono text-zinc-500">{trainCount} / {TRAIN_SAMPLES_NEEDED}</p>
                </div>
              )}

              {trainPhase === "done" && (
                <div className="flex flex-col gap-3">
                  <p className="text-xs font-mono text-emerald-400">
                    ✓ Capture complete! Total samples: {knnSamplesRef.current.length}
                  </p>
                  <div className="flex gap-3">
                    <button
                      onClick={() => { setTrainPhase("idle"); trainPhaseRef.current = "idle"; }}
                      className="px-4 py-2 text-xs font-mono font-bold rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 transition-colors"
                    >
                      + More Training
                    </button>
                    <button
                      onClick={saveModel}
                      className="px-4 py-2 text-xs font-mono font-bold rounded-lg bg-[#00f5d4]/10 border border-[#00f5d4]/30 text-[#00f5d4] hover:bg-[#00f5d4]/20 transition-colors"
                    >
                      💾 Save Model
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Status bar */}
      <div className="mt-4 flex items-center gap-2 font-mono text-xs tracking-widest uppercase">
        <span className={`w-1.5 h-1.5 rounded-full ${isRunning ? "bg-emerald-400 animate-pulse" : "bg-zinc-600"}`} />
        <span className={STATUS_COLORS[status]}>{STATUS_LABELS[status]}</span>
      </div>

      {/* Error */}
      {errorMsg && (
        <div className="mt-3 max-w-xl w-full bg-red-950/40 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm font-mono leading-relaxed">
          ⚠ {errorMsg}
        </div>
      )}

      {/* Buttons */}
      <div className="mt-6 flex flex-wrap gap-4 items-center justify-center">
        {!isRunning && (
          <button
            onClick={startCamera}
            disabled={isLoading}
            className="px-8 py-3 rounded-lg font-mono font-bold text-sm tracking-widest uppercase bg-[#00f5d4] text-[#0a0a0f] hover:bg-[#00f5d4]/80 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-150 shadow-[0_0_24px_rgba(0,245,212,0.35)]"
          >
            ▶ Start Camera
          </button>
        )}
        {isRunning && (
          <button
            onClick={stopCamera}
            className="px-8 py-3 rounded-lg font-mono font-bold text-sm tracking-widest uppercase bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 active:scale-95 transition-all duration-150"
          >
            ■ Stop Camera
          </button>
        )}

        {/* Calm Alert Settings Toggle Button */}
        <button
          onClick={() => setShowSettingsPanel((prev) => !prev)}
          className={`px-5 py-3 rounded-lg font-mono text-sm tracking-wider uppercase border transition-all duration-150 flex items-center gap-2 ${
            showSettingsPanel
              ? "bg-[#00f5d4]/20 border-[#00f5d4] text-[#00f5d4] shadow-[0_0_18px_rgba(0,245,212,0.25)]"
              : "bg-zinc-900 border-zinc-700 text-zinc-300 hover:border-zinc-500 hover:text-white"
          }`}
        >
          <span>⚙</span>
          <span>Calm Alert Settings</span>
          <span className="text-xs text-zinc-500 font-normal">({stressThresholdPct}% / {stressDurationThresholdSec}s)</span>
        </button>

        {/* Quick Test Alert Button */}
        <button
          onClick={triggerTestCalmAlert}
          title="Simulate a prolonged stress alert to test chime, speech, and popup"
          className="px-4 py-3 rounded-lg font-mono text-sm tracking-wider uppercase border border-zinc-700 bg-zinc-900/80 text-zinc-400 hover:text-[#00f5d4] hover:border-[#00f5d4]/40 transition-all flex items-center gap-1.5"
        >
          <span>🔔</span>
          <span>Test Alert</span>
        </button>
      </div>

      {/* Calm Alert Settings Panel */}
      {showSettingsPanel && (
        <div className="mt-4 w-full max-w-2xl bg-zinc-900/90 border border-zinc-700/80 rounded-2xl p-6 shadow-2xl backdrop-blur-md">
          <div className="flex items-center justify-between pb-3 border-b border-zinc-800 mb-4">
            <div className="flex items-center gap-2.5">
              <span className="text-lg">⚙</span>
              <h3 className="text-sm font-bold font-mono uppercase tracking-wider text-white">
                Prolonged Stress &amp; Calm Alert Thresholds
              </h3>
            </div>
            <button
              onClick={() => setShowSettingsPanel(false)}
              className="text-zinc-500 hover:text-zinc-300 text-xs font-mono"
            >
              ✕ Close
            </button>
          </div>

          <div className="flex flex-col gap-5">
            {/* Stress Threshold Slider & Presets */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-mono text-zinc-300 font-semibold">
                  Stress Level Threshold: <span className="text-[#00f5d4] font-bold text-sm">{stressThresholdPct}%</span>
                </label>
                <span className="text-[10px] font-mono text-zinc-500">Alert triggers when stress &ge; this value</span>
              </div>
              <input
                type="range"
                min="40"
                max="90"
                step="5"
                value={stressThresholdPct}
                onChange={(e) => updateStressThreshold(Number(e.target.value))}
                className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-[#00f5d4]"
              />
              <div className="flex flex-wrap gap-2 mt-2.5">
                {[
                  { label: "Sensitive (50%)", val: 50 },
                  { label: "Standard (65%)", val: 65 },
                  { label: "High Only (80%)", val: 80 },
                ].map(({ label, val }) => (
                  <button
                    key={val}
                    onClick={() => updateStressThreshold(val)}
                    className={`px-2.5 py-1 rounded text-xs font-mono border transition-all ${
                      stressThresholdPct === val
                        ? "bg-[#00f5d4]/20 border-[#00f5d4] text-[#00f5d4] font-bold"
                        : "bg-zinc-800/80 border-zinc-700 text-zinc-400 hover:text-white"
                    }`}
                  >
                    {label}
                  </button>
                ))}
                {baseline && (
                  <button
                    onClick={() => updateStressThreshold(Math.min(90, Math.round(baseline.avgStress * 100) + 20))}
                    className="px-2.5 py-1 rounded text-xs font-mono border border-purple-500/40 bg-purple-950/30 text-purple-300 hover:bg-purple-900/40 transition-all"
                  >
                    ⚡ Use Calibrated Baseline ({Math.min(90, Math.round(baseline.avgStress * 100) + 20)}%)
                  </button>
                )}
              </div>
            </div>

            {/* Duration Threshold */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-mono text-zinc-300 font-semibold">
                  Sustained Duration Threshold: <span className="text-amber-400 font-bold text-sm">{stressDurationThresholdSec} seconds</span>
                </label>
                <span className="text-[10px] font-mono text-zinc-500">Time stress must continuously stay elevated</span>
              </div>
              <div className="grid grid-cols-6 gap-2">
                {[5, 8, 10, 15, 20, 30].map((sec) => (
                  <button
                    key={sec}
                    onClick={() => updateDurationThreshold(sec)}
                    className={`py-1.5 rounded-lg text-xs font-mono font-bold border transition-all text-center ${
                      stressDurationThresholdSec === sec
                        ? "bg-amber-500/20 border-amber-400 text-amber-300 shadow-[0_0_12px_rgba(251,191,36,0.2)]"
                        : "bg-zinc-800/80 border-zinc-700 text-zinc-400 hover:text-white"
                    }`}
                  >
                    {sec}s
                  </button>
                ))}
              </div>
            </div>

            {/* Audio & Voice options */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-3 border-t border-zinc-800">
              <div>
                <label className="text-xs font-mono text-zinc-300 font-semibold block mb-2">
                  Alert Sound Mode
                </label>
                <div className="flex gap-2">
                  {[
                    { label: "🌿 Calm Chime", mode: "chime" as const },
                    { label: "⚠ Beep", mode: "beep" as const },
                    { label: "🔇 Mute", mode: "mute" as const },
                  ].map(({ label, mode }) => (
                    <button
                      key={mode}
                      onClick={() => updateSoundMode(mode)}
                      className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-mono border transition-all text-center ${
                        soundAlertMode === mode
                          ? "bg-[#00f5d4]/20 border-[#00f5d4] text-[#00f5d4] font-bold"
                          : "bg-zinc-800/80 border-zinc-700 text-zinc-400 hover:text-white"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-xs font-mono text-zinc-300 font-semibold block mb-2">
                  Voice Calm Guidance
                </label>
                <label className="flex items-center gap-2.5 cursor-pointer bg-zinc-800/60 border border-zinc-700 rounded-lg px-3 py-1.5">
                  <input
                    type="checkbox"
                    checked={voiceAlertEnabled}
                    onChange={(e) => updateVoiceAlert(e.target.checked)}
                    className="w-4 h-4 rounded border-zinc-600 accent-[#00f5d4] cursor-pointer"
                  />
                  <span className="text-xs font-mono text-zinc-300">
                    Spoken voice: &quot;Please take a deep breath and stay calm&quot;
                  </span>
                </label>
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-zinc-800/60 text-[11px] font-mono text-zinc-500">
              <span>Settings automatically saved to browser storage.</span>
              <button
                onClick={triggerTestCalmAlert}
                className="text-[#00f5d4] hover:underline font-semibold"
              >
                ▶ Test Alert Now
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Info cards */}
      <div className="mt-10 grid grid-cols-2 sm:grid-cols-4 gap-3 w-full max-w-2xl text-center">
        {[
          { label: "Detection Model", value: "SSD MobileNet v1" },
          { label: "Emotion Model", value: "Expression Net" },
          { label: "Landmark Points", value: "68 Points" },
          { label: "AI Model", value: modelReady ? "KNN Ready ✓" : "KNN (untrained)" },
        ].map(({ label, value }) => (
          <div key={label} className="bg-zinc-900/60 border border-zinc-800 rounded-lg py-3 px-2">
            <p className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest mb-1">{label}</p>
            <p className="text-sm font-bold font-mono text-zinc-300">{value}</p>
          </div>
        ))}
      </div>

      {/* Legend */}
      {isRunning && (
        <div className="mt-6 flex flex-wrap gap-5 justify-center text-xs font-mono text-zinc-500">
          <LegendDot color="#00f5d4" label="Face Box" />
          <LegendDot color="#4cc9f0" label="Jaw / Nose" />
          <LegendDot color="#7209b7" label="Eyes" />
          <LegendDot color="#f72585" label="Brows / Mouth" />
        </div>
      )}

      <footer className="mt-12 text-zinc-800 text-xs font-mono text-center">
        FaceSense Phase 4 · face-api.js · KNN · Adaptive Thresholds · Express + MongoDB · Notifications
      </footer>
    </main>
  );
}

//  Browser Notification helper ─
function sendBrowserNotification(title: string, body: string, icon = "🧠") {
  if (typeof window === "undefined") return;
  if (!("Notification" in window)) return;
  if (Notification.permission !== "granted") return;
  try {
    new Notification(title, {
      body,
      icon: "/favicon.ico",
      tag: icon, // deduplicate same-type notifications
      silent: false,
    });
  } catch {
    // Safari / older browsers may throw — fail silently
  }
}

//  Persist alert event to backend 
async function persistAlert(
  backendUrl: string,
  userId: string,
  sessionId: string,
  type: "stress" | "fatigue",
  stressScore: number,
  blinkRate: number,
  durationSec: number = 0,
  message: string = ""
) {
  try {
    await fetch(`${backendUrl}/api/alerts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userId, sessionId, type, stressScore, blinkRate, durationSec, message }),
    });
  } catch {
    // offline — alerts not persisted this time
  }
}

//  Soothing harmonic chime for relaxation (Web Audio API)
function playCalmChime() {
  try {
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextClass) return;
    const ctx = new AudioContextClass();
    const now = ctx.currentTime;

    // Harmonic Solfeggio 528Hz calming meditation chord:
    const freqs = [528, 792, 1056];
    const gains = [0.35, 0.18, 0.09];

    freqs.forEach((freq, idx) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, now);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.exponentialRampToValueAtTime(gains[idx], now + 0.06);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 2.2);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 2.3);
    });
  } catch {
    // AudioContext blocked
  }
}

//  Spoken voice reminder using Web Speech API
function speakCalmReminder(message = "Please take a deep breath and stay calm.") {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  try {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 0.88;
    utterance.pitch = 1.0;
    utterance.volume = 0.85;
    window.speechSynthesis.speak(utterance);
  } catch {
    // Speech synthesis error
  }
}

//  Audio beep helper 
function playBeep(freq = 880, duration = 0.2) {
  try {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + duration);
  } catch {
    // AudioContext not available
  }
}

//  Interactive Guided Box Breathing Widget (4-4-4-4)
function GuidedBreathingWidget({
  currentStress,
  targetThreshold,
}: {
  currentStress: number;
  targetThreshold: number;
}) {
  const [phaseIndex, setPhaseIndex] = useState(0); // 0: Inhale, 1: Hold, 2: Exhale, 3: Rest
  const [secondsLeft, setSecondsLeft] = useState(4);

  const PHASES = [
    { name: "Inhale Slowly", duration: 4, instruction: "Breathe in deeply through your nose", color: "#00f5d4" },
    { name: "Hold Gently", duration: 4, instruction: "Gently hold your breath, relax shoulders", color: "#38bdf8" },
    { name: "Exhale Slowly", duration: 4, instruction: "Release breath smoothly through mouth", color: "#a855f7" },
    { name: "Rest & Be Calm", duration: 4, instruction: "Stay peaceful and centered", color: "#22c55e" },
  ];

  useEffect(() => {
    const timer = setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          setPhaseIndex((p) => (p + 1) % PHASES.length);
          return 4;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [PHASES.length]);

  const currentPhase = PHASES[phaseIndex];
  const isRecovered = currentStress < targetThreshold;

  return (
    <div className="mt-4 pt-4 border-t border-zinc-800/80 flex flex-col md:flex-row items-center justify-between gap-6">
      <div className="flex items-center gap-5">
        {/* Animated breathing circle */}
        <div className="relative w-24 h-24 flex items-center justify-center flex-shrink-0">
          <div
            className={`absolute inset-0 rounded-full transition-all duration-1000 ${
              phaseIndex === 0
                ? "scale-100 bg-[#00f5d4]/20 border-2 border-[#00f5d4] shadow-[0_0_24px_rgba(0,245,212,0.4)]"
                : phaseIndex === 1
                ? "scale-100 bg-sky-500/20 border-2 border-sky-400 shadow-[0_0_24px_rgba(56,189,248,0.4)] animate-pulse"
                : phaseIndex === 2
                ? "scale-75 bg-purple-500/20 border-2 border-purple-400 shadow-[0_0_20px_rgba(168,85,247,0.3)]"
                : "scale-70 bg-emerald-500/10 border-2 border-emerald-400 shadow-[0_0_15px_rgba(34,197,94,0.3)]"
            }`}
          />
          <span className="relative z-10 text-2xl font-black font-mono" style={{ color: currentPhase.color }}>
            {secondsLeft}s
          </span>
        </div>

        <div>
          <span
            className="text-[10px] font-mono font-bold uppercase tracking-widest px-2 py-0.5 rounded border"
            style={{
              color: currentPhase.color,
              borderColor: currentPhase.color + "50",
              backgroundColor: currentPhase.color + "15",
            }}
          >
            Phase {phaseIndex + 1}/4 · {currentPhase.name}
          </span>
          <p className="text-sm font-mono font-bold text-white mt-1">{currentPhase.instruction}</p>
          <p className="text-xs font-mono text-zinc-400 mt-0.5">
            Box Breathing (4-4-4-4): Resets nervous system and reduces heart rate during prolonged stress.
          </p>
        </div>
      </div>

      {/* Real-time stress monitor */}
      <div className="bg-zinc-950/80 border border-zinc-800 rounded-xl p-3 flex flex-col gap-1 min-w-[210px] text-right">
        <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest">
          Live Stress Level
        </span>
        <div className="flex items-center justify-end gap-2">
          <span
            className="text-2xl font-mono font-black"
            style={{ color: isRecovered ? "#22c55e" : "#ef4444" }}
          >
            {currentStress}%
          </span>
          <span className="text-xs font-mono text-zinc-500">/ threshold &lt;{targetThreshold}%</span>
        </div>
        {isRecovered ? (
          <span className="text-[11px] font-mono text-emerald-400 font-semibold">
            ✓ Calm restored! Keep breathing.
          </span>
        ) : (
          <span className="text-[11px] font-mono text-amber-400">
            Elevated — breathe slowly...
          </span>
        )}
      </div>
    </div>
  );
}

//  Canvas helpers 
function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawCorners(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number,
  size: number, color: string
) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;

  const corners: Array<[[number, number], [number, number], [number, number]]> = [
    [[x, y + size], [x, y], [x + size, y]],
    [[x + w - size, y], [x + w, y], [x + w, y + size]],
    [[x + w, y + h - size], [x + w, y + h], [x + w - size, y + h]],
    [[x + size, y + h], [x, y + h], [x, y + h - size]],
  ];

  corners.forEach(([[ax, ay], [bx, by], [cx, cy]]) => {
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.lineTo(cx, cy);
    ctx.stroke();
  });
  ctx.shadowBlur = 0;
}

function drawLandmarkGroup(
  ctx: CanvasRenderingContext2D,
  points: FaceAPI.Point[],
  color: string,
  close: boolean
) {
  if (!points.length) return;

  ctx.beginPath();
  ctx.strokeStyle = color + "70";
  ctx.lineWidth = 1;
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  if (close) ctx.closePath();
  ctx.stroke();

  points.forEach((p) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 5;
    ctx.fill();
    ctx.shadowBlur = 0;
  });
}

//  UI sub-components 
function CameraOffIcon() {
  return (
    <svg width="52" height="52" viewBox="0 0 52 52" fill="none" className="text-zinc-800">
      <rect x="4" y="14" width="44" height="30" rx="5" stroke="currentColor" strokeWidth="2" />
      <circle cx="26" cy="29" r="8" stroke="currentColor" strokeWidth="2" />
      <path d="M18 14l3-6h10l3 6" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

function Spinner() {
  return <div className="w-9 h-9 rounded-full border-2 border-zinc-800 border-t-amber-400 animate-spin" />;
}

function HudBadge({ color, label, value }: { color: string; label: string; value: string }) {
  return (
    <div
      className="flex items-center gap-2 bg-black/60 border rounded px-2.5 py-1 font-mono text-xs backdrop-blur-sm"
      style={{ borderColor: color + "50" }}
    >
      <span style={{ color: color + "90" }} className="tracking-widest">{label}</span>
      <span style={{ color }} className="font-bold tabular-nums min-w-[1.5ch] text-right">{value}</span>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}
