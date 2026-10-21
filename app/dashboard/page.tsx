"use client";



import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

//  Config 
const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:5000";

const DATE_RANGES = [
  { label: "Last 1h", hours: 1 },
  { label: "Last 6h", hours: 6 },
  { label: "Last 24h", hours: 24 },
  { label: "Last 7d", hours: 168 },
  { label: "All time", hours: 0 },
];

const EMOTION_COLORS: Record<string, string> = {
  angry: "#ef4444",
  disgusted: "#a855f7",
  fearful: "#f97316",
  sad: "#3b82f6",
  surprised: "#eab308",
  neutral: "#6b7280",
  happy: "#22c55e",
};

const EMOTION_EMOJI: Record<string, string> = {
  angry: "😠", disgusted: "🤢", fearful: "😨",
  sad: "😢", surprised: "😲", neutral: "😐", happy: "😊",
};

//  Types 
type Summary = {
  avgStress: number;
  maxStress: number;
  avgBlinkRate: number;
  totalSamples: number;
  totalSessions: number;
  firstSeen: string | null;
  lastSeen: string | null;
};

type StressPoint = {
  timestamp: string;
  avgStress: number;
  maxStress: number;
  samples: number;
};

type BlinkPoint = {
  timestamp: string;
  avgBlinkRate: number;
  samples: number;
};

type EmotionPoint = {
  emotion: string;
  count: number;
  pct: number;
};

type AlertItem = {
  _id: string;
  type: "stress" | "fatigue";
  stressScore: number;
  blinkRate: number;
  timestamp: string;
  sessionId: string;
};

//  Helper: format ISO → short time label ─
function fmtTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fmtDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString([], { month: "short", day: "numeric" }) +
    " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

//  Loading skeleton ─
function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-pulse bg-zinc-800 rounded ${className}`}
    />
  );
}

//  Stat card 
function StatCard({
  label, value, sub, color = "#00f5d4",
}: {
  label: string; value: string | number; sub?: string; color?: string;
}) {
  return (
    <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-4">
      <p className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest mb-1">{label}</p>
      <p className="text-2xl font-black font-mono" style={{ color }}>{value}</p>
      {sub && <p className="text-[10px] font-mono text-zinc-600 mt-0.5">{sub}</p>}
    </div>
  );
}

//  Custom tooltip for recharts 
function ChartTooltip({ active, payload, label }: {
  active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-xs font-mono shadow-xl">
      <p className="text-zinc-400 mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: <span className="font-bold">{p.value}</span>
        </p>
      ))}
    </div>
  );
}

//  Main Dashboard Page ─
export default function DashboardPage() {
  const [userId, setUserId] = useState<string>("");
  const [rangeIdx, setRangeIdx] = useState(2); // default: last 24h
  const [summary, setSummary] = useState<Summary | null>(null);
  const [stressData, setStressData] = useState<StressPoint[]>([]);
  const [blinkData, setBlinkData] = useState<BlinkPoint[]>([]);
  const [emotionData, setEmotionData] = useState<EmotionPoint[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");
  const [backendOk, setBackendOk] = useState<boolean | null>(null);

  //  Resolve userId from localStorage 
  useEffect(() => {
    const id = localStorage.getItem("facesense_userId") ?? "";
    setUserId(id);
  }, []);

  //  Build query params for current date range 
  const buildParams = useCallback(
    (extra: Record<string, string> = {}) => {
      if (!userId) return null;
      const params = new URLSearchParams({ userId, ...extra });
      const hours = DATE_RANGES[rangeIdx].hours;
      if (hours > 0) {
        const from = new Date(Date.now() - hours * 3600 * 1000).toISOString();
        params.set("from", from);
      }
      return params.toString();
    },
    [userId, rangeIdx]
  );

  //  Fetch all dashboard data 
  const fetchAll = useCallback(async () => {
    if (!userId) return;
    const params = buildParams();
    if (!params) return;

    setLoading(true);
    setError("");

    try {
      const [sumRes, stressRes, blinkRes, emotionRes, alertRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/analytics/summary?${params}`),
        fetch(`${BACKEND_URL}/api/analytics/stress?${params}&intervalMin=5`),
        fetch(`${BACKEND_URL}/api/analytics/blink?${params}&intervalMin=5`),
        fetch(`${BACKEND_URL}/api/analytics/emotion?${params}`),
        fetch(`${BACKEND_URL}/api/alerts?${params}&limit=20`),
      ]);

      setBackendOk(sumRes.ok);

      if (!sumRes.ok) throw new Error("Backend returned an error. Is the server running?");

      const [sumJson, stressJson, blinkJson, emotionJson, alertJson] = await Promise.all([
        sumRes.json(), stressRes.json(), blinkRes.json(),
        emotionRes.json(), alertRes.json(),
      ]);

      setSummary(sumJson);
      setStressData(stressJson.data ?? []);
      setBlinkData(blinkJson.data ?? []);
      setEmotionData(emotionJson.data ?? []);
      setAlerts(alertJson.data ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dashboard data.");
      setBackendOk(false);
    } finally {
      setLoading(false);
    }
  }, [userId, buildParams]);

  useEffect(() => {
    if (userId) fetchAll();
  }, [userId, fetchAll]);

  const stressPct = summary ? Math.round(summary.avgStress * 100) : 0;
  const stressColor = stressPct >= 70 ? "#ef4444" : stressPct >= 40 ? "#f97316" : "#22c55e";

  //  Prepare stress chart data with formatted labels ─
  const stressChartData = stressData.map((d) => ({
    ...d,
    time: fmtTime(d.timestamp),
    avgStressPct: Math.round(d.avgStress * 100),
    maxStressPct: Math.round(d.maxStress * 100),
  }));

  const blinkChartData = blinkData.map((d) => ({
    ...d,
    time: fmtTime(d.timestamp),
  }));

  //  No userId state 
  if (!userId) {
    return (
      <main className="min-h-screen bg-[#0a0a0f] text-white flex flex-col items-center justify-center px-4">
        <div className="text-center max-w-md">
          <div className="text-5xl mb-4">📊</div>
          <h1 className="text-2xl font-black font-mono mb-3 text-[#00f5d4]">No User Data Found</h1>
          <p className="text-zinc-500 font-mono text-sm mb-6">
            You haven&apos;t started a FaceSense session yet. Open the main page, start the camera, and your analytics will appear here.
          </p>
          <Link
            href="/"
            className="px-6 py-3 rounded-lg font-mono font-bold text-sm tracking-widest uppercase bg-[#00f5d4] text-[#0a0a0f] hover:bg-[#00f5d4]/80 transition-all"
          >
            ← Go to FaceSense
          </Link>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-[#0a0a0f] text-white px-4 py-8">
      <div className="max-w-6xl mx-auto">

        {/*  Header ─ */}
        <header className="flex items-center justify-between mb-8 flex-wrap gap-4">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <div className="w-2 h-2 rounded-full bg-[#00f5d4] animate-pulse shadow-[0_0_8px_#00f5d4]" />
              <h1 className="text-2xl font-black tracking-[0.15em] uppercase font-mono">
                FACE<span className="text-[#00f5d4]">SENSE</span>
                <span className="text-zinc-600 ml-2">/ Dashboard</span>
              </h1>
            </div>
            <p className="text-[10px] text-zinc-700 font-mono">
              User: {userId.slice(0, 8)}… &nbsp;·&nbsp;
              <span style={{ color: backendOk === false ? "#ef4444" : backendOk ? "#22c55e" : "#6b7280" }}>
                {backendOk === null ? "connecting…" : backendOk ? "backend connected ✓" : "backend offline ✗"}
              </span>
            </p>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            {/* Date range selector */}
            <div className="flex bg-zinc-900 border border-zinc-800 rounded-lg p-0.5 gap-0.5">
              {DATE_RANGES.map((r, i) => (
                <button
                  key={r.label}
                  onClick={() => setRangeIdx(i)}
                  className={`px-3 py-1.5 text-xs font-mono rounded transition-all ${rangeIdx === i
                    ? "bg-[#00f5d4]/10 text-[#00f5d4] border border-[#00f5d4]/30"
                    : "text-zinc-500 hover:text-zinc-300"
                    }`}
                >
                  {r.label}
                </button>
              ))}
            </div>

            <button
              onClick={fetchAll}
              disabled={loading}
              className="px-4 py-2 text-xs font-mono font-bold rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 disabled:opacity-40 transition-all"
            >
              {loading ? "⟳ Loading…" : "↺ Refresh"}
            </button>

            <Link
              href="/"
              className="px-4 py-2 text-xs font-mono font-bold rounded-lg bg-[#00f5d4]/10 border border-[#00f5d4]/30 text-[#00f5d4] hover:bg-[#00f5d4]/20 transition-all"
            >
              ← Camera
            </Link>
          </div>
        </header>

        {/*  Error banner ─ */}
        {error && (
          <div className="mb-6 bg-red-950/40 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm font-mono">
            ⚠ {error}
          </div>
        )}

        {/*  Summary cards  */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
          {loading ? (
            Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-20" />)
          ) : summary ? (
            <>
              <StatCard
                label="Avg Stress"
                value={`${stressPct}%`}
                sub={stressPct >= 70 ? "High stress" : stressPct >= 40 ? "Moderate" : "Low — calm"}
                color={stressColor}
              />
              <StatCard
                label="Peak Stress"
                value={`${Math.round(summary.maxStress * 100)}%`}
                color="#f97316"
              />
              <StatCard
                label="Avg Blink Rate"
                value={`${summary.avgBlinkRate}/min`}
                sub={summary.avgBlinkRate < 10 ? "⚠ Below normal" : "Normal range"}
                color="#7209b7"
              />
              <StatCard
                label="Sessions"
                value={summary.totalSessions}
                sub="camera runs"
                color="#4cc9f0"
              />
              <StatCard
                label="Data Points"
                value={summary.totalSamples.toLocaleString()}
                sub={summary.lastSeen ? `Last: ${fmtDate(summary.lastSeen)}` : "No data yet"}
                color="#00f5d4"
              />
            </>
          ) : (
            <div className="col-span-5 text-center text-zinc-600 font-mono text-sm py-6">
              No data for this time range. Start a session on the camera page.
            </div>
          )}
        </div>

        {/*  Charts grid  */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">

          {/* Stress over time */}
          <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-4">
            <p className="text-xs font-mono text-zinc-400 font-bold uppercase tracking-widest mb-4">
              Stress Score Over Time
            </p>
            {loading ? (
              <Skeleton className="h-52" />
            ) : stressChartData.length === 0 ? (
              <EmptyChart message="No stress data in this range" />
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={stressChartData} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis
                    dataKey="time"
                    tick={{ fill: "#71717a", fontSize: 10, fontFamily: "monospace" }}
                    tickLine={false}
                    axisLine={{ stroke: "#27272a" }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fill: "#71717a", fontSize: 10, fontFamily: "monospace" }}
                    tickLine={false}
                    axisLine={false}
                    domain={[0, 100]}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend
                    wrapperStyle={{ fontSize: "10px", fontFamily: "monospace", color: "#71717a" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="avgStressPct"
                    name="Avg Stress %"
                    stroke="#00f5d4"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: "#00f5d4" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="maxStressPct"
                    name="Peak Stress %"
                    stroke="#ef4444"
                    strokeWidth={1}
                    strokeDasharray="4 2"
                    dot={false}
                    activeDot={{ r: 4, fill: "#ef4444" }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Blink rate over time */}
          <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-4">
            <p className="text-xs font-mono text-zinc-400 font-bold uppercase tracking-widest mb-4">
              Blink Rate Over Time
            </p>
            {loading ? (
              <Skeleton className="h-52" />
            ) : blinkChartData.length === 0 ? (
              <EmptyChart message="No blink data in this range" />
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={blinkChartData} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis
                    dataKey="time"
                    tick={{ fill: "#71717a", fontSize: 10, fontFamily: "monospace" }}
                    tickLine={false}
                    axisLine={{ stroke: "#27272a" }}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fill: "#71717a", fontSize: 10, fontFamily: "monospace" }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `${v}/m`}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend
                    wrapperStyle={{ fontSize: "10px", fontFamily: "monospace", color: "#71717a" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="avgBlinkRate"
                    name="Blinks/min"
                    stroke="#7209b7"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: "#7209b7" }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
            {!loading && blinkChartData.length > 0 && (
              <p className="text-[10px] font-mono text-zinc-600 mt-2">
                Normal range: 15–20 blinks/min · Below 10/min indicates eye strain
              </p>
            )}
          </div>
        </div>

        {/* Emotion bar chart + alert history */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">

          {/* Emotion distribution */}
          <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-4">
            <p className="text-xs font-mono text-zinc-400 font-bold uppercase tracking-widest mb-4">
              Emotion Distribution
            </p>
            {loading ? (
              <Skeleton className="h-52" />
            ) : emotionData.length === 0 ? (
              <EmptyChart message="No emotion data in this range" />
            ) : (
              <>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart
                    data={emotionData}
                    margin={{ top: 4, right: 8, bottom: 4, left: -20 }}
                    barCategoryGap="30%"
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis
                      dataKey="emotion"
                      tick={{ fill: "#71717a", fontSize: 10, fontFamily: "monospace" }}
                      tickLine={false}
                      axisLine={{ stroke: "#27272a" }}
                      tickFormatter={(e) => (EMOTION_EMOJI[e] ?? e).slice(0, 2)}
                    />
                    <YAxis
                      tick={{ fill: "#71717a", fontSize: 10, fontFamily: "monospace" }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => `${v}%`}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar
                      dataKey="pct"
                      name="Frequency %"
                      radius={[4, 4, 0, 0]}
                      // Use per-bar coloring via Cell — fallback to solid color
                      fill="#00f5d4"
                      isAnimationActive={true}
                    />
                  </BarChart>
                </ResponsiveContainer>

                {/* Legend pills */}
                <div className="flex flex-wrap gap-2 mt-3">
                  {emotionData.map((d) => (
                    <span
                      key={d.emotion}
                      className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full border"
                      style={{
                        color: EMOTION_COLORS[d.emotion] ?? "#71717a",
                        borderColor: (EMOTION_COLORS[d.emotion] ?? "#71717a") + "40",
                        backgroundColor: (EMOTION_COLORS[d.emotion] ?? "#71717a") + "10",
                      }}
                    >
                      {EMOTION_EMOJI[d.emotion] ?? "?"} {d.emotion} {d.pct}%
                    </span>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Alert history */}
          <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-4">
            <p className="text-xs font-mono text-zinc-400 font-bold uppercase tracking-widest mb-4">
              Alert History
            </p>
            {loading ? (
              <div className="flex flex-col gap-2">
                {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-10" />)}
              </div>
            ) : alerts.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-40 text-zinc-600 font-mono text-sm gap-2">
                <span className="text-3xl">✓</span>
                <span>No alerts in this range</span>
              </div>
            ) : (
              <div className="flex flex-col gap-2 max-h-64 overflow-y-auto pr-1 scrollbar-thin">
                {alerts.map((a) => (
                  <div
                    key={a._id}
                    className={`flex items-start gap-3 rounded-lg px-3 py-2.5 border text-xs font-mono ${a.type === "stress"
                      ? "bg-red-950/30 border-red-500/20 text-red-400"
                      : "bg-amber-950/30 border-amber-500/20 text-amber-400"
                      }`}
                  >
                    <span className="text-base mt-0.5">{a.type === "stress" ? "⚠" : "👁"}</span>
                    <div className="flex-1 min-w-0">
                      <p className="font-bold uppercase tracking-widest text-[10px]">
                        {a.type === "stress" ? "High Stress" : "Low Blink Rate"}
                      </p>
                      <p className="text-zinc-500 text-[10px]">
                        {a.type === "stress"
                          ? `Stress: ${Math.round(a.stressScore * 100)}%`
                          : `Blink rate: ${a.blinkRate}/min`}
                      </p>
                    </div>
                    <span className="text-zinc-600 text-[10px] flex-shrink-0">{fmtDate(a.timestamp)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-zinc-800 text-xs font-mono text-center">
          FaceSense Phase 4 · Analytics Dashboard · recharts · MongoDB Atlas
        </footer>
      </div>
    </main>
  );
}

//  Empty chart placeholder 
function EmptyChart({ message }: { message: string }) {
  return (
    <div className="h-52 flex flex-col items-center justify-center text-zinc-700 font-mono text-sm gap-2">
      <span className="text-3xl opacity-30">📈</span>
      <span>{message}</span>
    </div>
  );
}
