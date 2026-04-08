import { useState, useMemo } from "react";

// ============================================================================
// Paper 4 Level 2 Output: BIPV Suitability Lookup Tool
// 基于 FDSI 五维适宜性评价框架的交互式查询工具
// ============================================================================

// 五城市完整数据（来自 Step 1-5 计算结果）
const CITY_DATA = {
  shenzhen: {
    name_en: "Shenzhen", name_cn: "深圳",
    zone: "HSWW", zone_cn: "夏热冬暖", zone_full: "Hot Summer Warm Winter",
    lat: 22.54, lon: 114.06,
    fdsi: 0.647, rank: 1, suitability: "High", uncertainty: "Low",
    dominant_morphology: "high_rise",
    scores: { D1: 0.671, D2: 0.117, D3: 0.761, D4: 0.838, D5: 0.913 },
    raw: {
      ghi: 1561, temp: 23.1, sunshine_h: 3842, specific_yield: 1299,
      height_m: 40.7, density: 0.667, roof_area: 329.6,
      lcoe: 0.214, pbt: 3.52, deployable_mw: 329.6,
      pbt_ci_width: 1.21, dominant_factor: "pv_cost"
    }
  },
  beijing: {
    name_en: "Beijing", name_cn: "北京",
    zone: "Cold", zone_cn: "寒冷", zone_full: "Cold",
    lat: 39.90, lon: 116.40,
    fdsi: 0.575, rank: 2, suitability: "High", uncertainty: "Low",
    dominant_morphology: "mid_rise",
    scores: { D1: 0.735, D2: 0.712, D3: 0.648, D4: 0.573, D5: 0.567 },
    raw: {
      ghi: 1578, temp: 13.2, sunshine_h: 4156, specific_yield: 1358,
      height_m: 18.5, density: 0.052, roof_area: 475.9,
      lcoe: 0.205, pbt: 4.74, deployable_mw: 475.9,
      pbt_ci_width: 1.85, dominant_factor: "pv_cost"
    }
  },
  kunming: {
    name_en: "Kunming", name_cn: "昆明",
    zone: "Mild", zone_cn: "温和", zone_full: "Mild",
    lat: 25.04, lon: 102.68,
    fdsi: 0.552, rank: 3, suitability: "Medium", uncertainty: "Medium",
    dominant_morphology: "mid_high",
    scores: { D1: 1.0, D2: 0.578, D3: 1.0, D4: 0.340, D5: 0.421 },
    raw: {
      ghi: 1650, temp: 15.8, sunshine_h: 4520, specific_yield: 1405,
      height_m: 27.4, density: 0.074, roof_area: 262.3,
      lcoe: 0.198, pbt: 4.98, deployable_mw: 262.3,
      pbt_ci_width: 2.31, dominant_factor: "pv_cost"
    }
  },
  changsha: {
    name_en: "Changsha", name_cn: "长沙",
    zone: "HSCW", zone_cn: "夏热冬冷", zone_full: "Hot Summer Cold Winter",
    lat: 28.23, lon: 112.94,
    fdsi: 0.332, rank: 4, suitability: "Low", uncertainty: "High",
    dominant_morphology: "mid_high",
    scores: { D1: 0.0, D2: 0.458, D3: 0.0, D4: 0.280, D5: 0.187 },
    raw: {
      ghi: 1378, temp: 17.5, sunshine_h: 3280, specific_yield: 1162,
      height_m: 22.0, density: 0.046, roof_area: 361.8,
      lcoe: 0.239, pbt: 4.59, deployable_mw: 361.8,
      pbt_ci_width: 2.65, dominant_factor: "pv_cost"
    }
  },
  harbin: {
    name_en: "Harbin", name_cn: "哈尔滨",
    zone: "Severe Cold", zone_cn: "严寒", zone_full: "Severe Cold",
    lat: 45.75, lon: 126.65,
    fdsi: 0.273, rank: 5, suitability: "Low", uncertainty: "High",
    dominant_morphology: "mid_high",
    scores: { D1: 0.169, D2: 0.432, D3: 0.240, D4: 0.0, D5: 0.0 },
    raw: {
      ghi: 1424, temp: 5.2, sunshine_h: 3650, specific_yield: 1259,
      height_m: 21.3, density: 0.374, roof_area: 1362.0,
      lcoe: 0.221, pbt: 4.90, deployable_mw: 1362.0,
      pbt_ci_width: 2.88, dominant_factor: "pv_cost"
    }
  },
};

const DIMENSIONS = [
  { key: "D1", label: "Climate Resource", label_cn: "气候资源", icon: "☀️" },
  { key: "D2", label: "Urban Morphology", label_cn: "城市形态", icon: "🏗️" },
  { key: "D3", label: "Technical Deployment", label_cn: "技术部署", icon: "⚡" },
  { key: "D4", label: "Economic Feasibility", label_cn: "经济可行", icon: "💰" },
  { key: "D5", label: "Uncertainty", label_cn: "确定性", icon: "🎯" },
];

const SUIT_CONFIG = {
  High:   { bg: "rgba(46,125,50,0.12)", border: "#2E7D32", text: "#2E7D32", tag: "bg-emerald-100 text-emerald-800 border-emerald-300" },
  Medium: { bg: "rgba(245,127,23,0.12)", border: "#F57F17", text: "#F57F17", tag: "bg-amber-100 text-amber-800 border-amber-300" },
  Low:    { bg: "rgba(198,40,40,0.12)", border: "#C62828", text: "#C62828", tag: "bg-red-100 text-red-800 border-red-300" },
};

const UNCERT_CONFIG = {
  Low:    { label: "Low Uncertainty", tag: "bg-sky-100 text-sky-800 border-sky-300" },
  Medium: { label: "Medium Uncertainty", tag: "bg-orange-100 text-orange-800 border-orange-300" },
  High:   { label: "High Uncertainty", tag: "bg-rose-100 text-rose-800 border-rose-300" },
};

// ── Radar Chart (SVG) ──
function RadarChart({ scores, color = "#2E7D32", size = 220 }) {
  const cx = size / 2, cy = size / 2, r = size * 0.38;
  const n = DIMENSIONS.length;
  const angleStep = (2 * Math.PI) / n;
  const startAngle = -Math.PI / 2;

  const getPoint = (i, val) => {
    const angle = startAngle + i * angleStep;
    return { x: cx + r * val * Math.cos(angle), y: cy + r * val * Math.sin(angle) };
  };

  const gridLevels = [0.25, 0.5, 0.75, 1.0];
  const axes = DIMENSIONS.map((_, i) => getPoint(i, 1));
  const dataPoints = DIMENSIONS.map((d, i) => getPoint(i, scores[d.key] || 0));
  const pathD = dataPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ") + " Z";

  return (
    <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size}>
      {gridLevels.map(lv => {
        const pts = DIMENSIONS.map((_, i) => getPoint(i, lv));
        const d = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ") + " Z";
        return <path key={lv} d={d} fill="none" stroke="#e0e0e0" strokeWidth={0.8} />;
      })}
      {axes.map((p, i) => (
        <g key={i}>
          <line x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="#d0d0d0" strokeWidth={0.6} />
          <text
            x={cx + (r + 18) * Math.cos(startAngle + i * angleStep)}
            y={cy + (r + 18) * Math.sin(startAngle + i * angleStep)}
            textAnchor="middle" dominantBaseline="central"
            fontSize={9} fill="#555" fontWeight={600}
          >
            {DIMENSIONS[i].key}
          </text>
        </g>
      ))}
      <path d={pathD} fill={color} fillOpacity={0.18} stroke={color} strokeWidth={2.2} />
      {dataPoints.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={3.5} fill={color} stroke="white" strokeWidth={1.5} />
      ))}
    </svg>
  );
}

// ── Dimension Bar ──
function DimensionBar({ dim, score, maxScore = 1 }) {
  const pct = Math.round(score * 100);
  const hue = score > 0.66 ? 142 : score > 0.33 ? 36 : 0;
  const sat = 65, light = 42;
  return (
    <div className="flex items-center gap-3 py-1.5">
      <span className="text-lg w-6 text-center">{dim.icon}</span>
      <span className="text-xs font-semibold w-10 text-right" style={{ color: `hsl(${hue},${sat}%,${light}%)` }}>
        {score.toFixed(2)}
      </span>
      <div className="flex-1 h-2.5 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, hsl(${hue},${sat}%,${light + 15}%), hsl(${hue},${sat}%,${light}%))`,
          }}
        />
      </div>
      <span className="text-xs text-gray-400 w-16 truncate">{dim.label_cn}</span>
    </div>
  );
}

// ── City Card ──
function CityCard({ cityKey, data, isSelected, onClick }) {
  const suitCfg = SUIT_CONFIG[data.suitability];
  return (
    <button
      onClick={onClick}
      className={`w-full text-left rounded-xl p-4 border-2 transition-all duration-300 ${
        isSelected
          ? "shadow-lg scale-[1.02]"
          : "hover:shadow-md hover:scale-[1.01] opacity-80 hover:opacity-100"
      }`}
      style={{
        borderColor: isSelected ? suitCfg.border : "transparent",
        background: isSelected ? suitCfg.bg : "rgba(255,255,255,0.6)",
      }}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-black tracking-tight" style={{ fontFamily: "'DM Serif Display', Georgia, serif" }}>
              #{data.rank}
            </span>
            <div>
              <div className="font-bold text-base leading-tight">{data.name_en}</div>
              <div className="text-xs text-gray-500">{data.name_cn} · {data.zone_cn}</div>
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-xl font-black tabular-nums" style={{ color: suitCfg.text }}>
            {data.fdsi.toFixed(3)}
          </div>
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${suitCfg.tag}`}>
            {data.suitability}
          </span>
        </div>
      </div>
    </button>
  );
}

// ── Detail Panel ──
function DetailPanel({ data }) {
  if (!data) return null;
  const suitCfg = SUIT_CONFIG[data.suitability];
  const uncertCfg = UNCERT_CONFIG[data.uncertainty];
  const r = data.raw;

  const stats = [
    { label: "Annual GHI", value: `${r.ghi.toLocaleString()} kWh/m²/yr`, sub: "年均水平面总辐射" },
    { label: "Specific Yield", value: `${r.specific_yield.toLocaleString()} kWh/kWp/yr`, sub: "单位容量发电量" },
    { label: "LCOE", value: `¥${r.lcoe.toFixed(3)}/kWh`, sub: "度电成本" },
    { label: "Payback Period", value: `${r.pbt.toFixed(1)} years`, sub: "简单回收期" },
    { label: "Mean Height", value: `${r.height_m.toFixed(1)} m`, sub: "平均建筑高度" },
    { label: "Building Density", value: `${(r.density * 100).toFixed(1)}%`, sub: "建筑密度" },
    { label: "PBT 95% CI Width", value: `${r.pbt_ci_width.toFixed(2)} yr`, sub: "回收期不确定性范围" },
    { label: "Key Sensitivity", value: r.dominant_factor, sub: "Sobol主导因子" },
  ];

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-black tracking-tight" style={{ fontFamily: "'DM Serif Display', Georgia, serif" }}>
            {data.name_en} <span className="text-gray-400 font-normal text-lg">{data.name_cn}</span>
          </h2>
          <p className="text-sm text-gray-500 mt-0.5">
            {data.zone_full} Zone ({data.zone}) · {data.lat}°N, {data.lon}°E
          </p>
        </div>
        <div className="text-right">
          <div className="text-3xl font-black tabular-nums" style={{ color: suitCfg.text }}>
            {data.fdsi.toFixed(3)}
          </div>
          <div className="text-xs text-gray-400 mt-0.5">FDSI Score</div>
        </div>
      </div>

      {/* Tags */}
      <div className="flex gap-2 flex-wrap">
        <span className={`text-xs font-semibold px-3 py-1 rounded-full border ${suitCfg.tag}`}>
          Suitability: {data.suitability}
        </span>
        <span className={`text-xs font-semibold px-3 py-1 rounded-full border ${uncertCfg.tag}`}>
          {uncertCfg.label}
        </span>
        <span className="text-xs font-semibold px-3 py-1 rounded-full border bg-gray-100 text-gray-600 border-gray-300">
          {data.dominant_morphology.replace("_", " ")}
        </span>
      </div>

      {/* Radar */}
      <div className="flex justify-center">
        <RadarChart scores={data.scores} color={suitCfg.border} />
      </div>

      {/* Dimension Bars */}
      <div className="space-y-0.5">
        {DIMENSIONS.map(dim => (
          <DimensionBar key={dim.key} dim={dim} score={data.scores[dim.key]} />
        ))}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-2">
        {stats.map((s, i) => (
          <div key={i} className="bg-gray-50 rounded-lg p-2.5">
            <div className="text-xs text-gray-400">{s.label}</div>
            <div className="font-bold text-sm mt-0.5">{s.value}</div>
            <div className="text-xs text-gray-300 mt-0.5">{s.sub}</div>
          </div>
        ))}
      </div>

      {/* Recommendation */}
      <div className="rounded-lg p-3 border" style={{ background: suitCfg.bg, borderColor: suitCfg.border + "40" }}>
        <div className="text-xs font-bold mb-1" style={{ color: suitCfg.text }}>
          Deployment Recommendation
        </div>
        <p className="text-xs text-gray-600 leading-relaxed">
          {data.suitability === "High" && "Highly recommended for residential BIPV deployment. Strong solar resource, favorable economics, and reliable assessment results support large-scale implementation."}
          {data.suitability === "Medium" && "Moderate suitability for residential BIPV. Consider site-specific assessment and focus on neighborhoods with favorable morphological conditions."}
          {data.suitability === "Low" && "Lower overall suitability under current conditions. BIPV deployment should be selective, targeting buildings with optimal orientation and minimal shading. Policy incentives may improve economic feasibility."}
        </p>
      </div>
    </div>
  );
}

// ── Main App ──
export default function BIPVLookupTool() {
  const [selectedCity, setSelectedCity] = useState("shenzhen");
  const [sortBy, setSortBy] = useState("rank");

  const sortedCities = useMemo(() => {
    const entries = Object.entries(CITY_DATA);
    if (sortBy === "rank") return entries.sort((a, b) => a[1].rank - b[1].rank);
    if (sortBy === "ghi") return entries.sort((a, b) => b[1].raw.ghi - a[1].raw.ghi);
    if (sortBy === "pbt") return entries.sort((a, b) => a[1].raw.pbt - b[1].raw.pbt);
    if (sortBy === "lcoe") return entries.sort((a, b) => a[1].raw.lcoe - b[1].raw.lcoe);
    return entries;
  }, [sortBy]);

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #f8faf8 0%, #f0f4f8 50%, #f5f3f0 100%)",
      fontFamily: "'IBM Plex Sans', -apple-system, sans-serif",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@400;500;600;700;800;900&display=swap" rel="stylesheet" />

      {/* Header */}
      <div className="px-6 pt-8 pb-4 max-w-5xl mx-auto">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-black tracking-tight leading-tight" style={{ fontFamily: "'DM Serif Display', Georgia, serif" }}>
              BIPV Suitability<br/>
              <span className="text-gray-400">Lookup Tool</span>
            </h1>
            <p className="text-xs text-gray-400 mt-2 max-w-md leading-relaxed">
              Multi-factor suitability assessment for residential BIPV deployment
              across China's five climate zones. Based on the Five-Dimension
              Suitability Index (FDSI) framework.
            </p>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-300 mb-1">Sort by</div>
            <div className="flex gap-1">
              {[
                { key: "rank", label: "FDSI" },
                { key: "ghi", label: "GHI" },
                { key: "pbt", label: "PBT" },
                { key: "lcoe", label: "LCOE" },
              ].map(opt => (
                <button
                  key={opt.key}
                  onClick={() => setSortBy(opt.key)}
                  className={`text-xs px-2.5 py-1 rounded-full border transition-all ${
                    sortBy === opt.key
                      ? "bg-gray-800 text-white border-gray-800"
                      : "bg-white text-gray-500 border-gray-200 hover:border-gray-400"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="px-6 pb-12 max-w-5xl mx-auto">
        <div className="flex gap-5" style={{ alignItems: "flex-start" }}>
          {/* City List */}
          <div className="w-72 flex-shrink-0 space-y-2">
            {sortedCities.map(([key, data]) => (
              <CityCard
                key={key}
                cityKey={key}
                data={data}
                isSelected={selectedCity === key}
                onClick={() => setSelectedCity(key)}
              />
            ))}
          </div>

          {/* Detail */}
          <div className="flex-1 bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
            <DetailPanel data={CITY_DATA[selectedCity]} />
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center">
          <p className="text-xs text-gray-300">
            Paper 4 · Five-Dimension Suitability Index (FDSI) Framework · {new Date().getFullYear()}
          </p>
          <p className="text-xs text-gray-300 mt-1">
            Data: OSM + PVGIS ERA5 · Method: Entropy-AHP Combined Weighting + Monte Carlo + Sobol SA
          </p>
        </div>
      </div>
    </div>
  );
}
