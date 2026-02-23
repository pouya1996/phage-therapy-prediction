import { useCallback, useRef, useState } from 'react';
import html2canvas from 'html2canvas';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Cell,
} from 'recharts';

/* ---------------------------------------------------------------
   Tableau 10 — widely considered the best-practice palette
   for data visualisation (colorblind-safe, print-friendly).
   --------------------------------------------------------------- */
const PALETTE = [
  '#4E79A7', // steel blue
  '#F28E2B', // orange
  '#E15759', // coral red
  '#76B7B2', // teal
  '#59A14F', // green
  '#EDC948', // gold
  '#B07AA1', // lavender
];

const METRICS = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'specificity'];

const METRIC_LABELS = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1: 'F1 Score',
  mcc: 'MCC',
  auc: 'AUC',
  specificity: 'Specificity',
};

const MODEL_DISPLAY = {
  knn: 'KNN',
  svm: 'SVM',
  rf: 'Random Forest',
  xgboost: 'XGBoost',
  adaboost: 'AdaBoost',
  lr: 'Logistic Reg.',
  multiview_cnn: 'Multiview CNN',
};

function fmt(v) {
  if (v == null || isNaN(v)) return '—';
  return Number(v).toFixed(4);
}

function fmtTime(v) {
  if (v == null || isNaN(v)) return '—';
  const s = Number(v);
  return s >= 60 ? `${(s / 60).toFixed(1)}m` : `${s.toFixed(1)}s`;
}

function modelColor(idx) {
  return PALETTE[idx % PALETTE.length];
}

/* ---------------------------------------------------------------
   Training Info Panel
   --------------------------------------------------------------- */
function TrainingInfoPanel({ info }) {
  if (!info) return null;

  const items = [
    { label: 'Total Samples', value: info.total_samples?.toLocaleString() },
    { label: 'Training Set', value: info.train_samples?.toLocaleString() },
    { label: 'Test Set', value: info.test_samples?.toLocaleString() },
    { label: 'Positive (class 1)', value: info.positive_samples?.toLocaleString() },
    { label: 'Negative (class 0)', value: info.negative_samples?.toLocaleString() },
    { label: 'Unique Phages', value: info.unique_phages?.toLocaleString() },
    { label: 'Unique Hosts', value: info.unique_hosts?.toLocaleString() },
    { label: 'CV Folds', value: info.cv_folds },
    { label: 'Random Seed', value: info.random_seed },
  ];

  const morphEntries = info.morphologies
    ? Object.entries(info.morphologies)
    : [];

  return (
    <div className="card">
      <h2>Training Dataset</h2>
      <div className="info-grid">
        {items.map((it) => (
          <div key={it.label} className="info-item">
            <span className="info-value">{it.value ?? '—'}</span>
            <span className="info-label">{it.label}</span>
          </div>
        ))}
      </div>
      {morphEntries.length > 0 && (
        <div className="morph-bar-wrapper">
          <h3 className="morph-title">Morphology Distribution</h3>
          <div className="morph-bars">
            {morphEntries.map(([name, count], i) => {
              const total = morphEntries.reduce((s, [, c]) => s + c, 0);
              const pct = ((count / total) * 100).toFixed(1);
              return (
                <div key={name} className="morph-row">
                  <span className="morph-name">{name}</span>
                  <div className="morph-track">
                    <div
                      className="morph-fill"
                      style={{ width: `${pct}%`, background: PALETTE[i] }}
                    />
                  </div>
                  <span className="morph-count">{count.toLocaleString()} ({pct}%)</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------
   Dashboard
   --------------------------------------------------------------- */
export default function Dashboard({ data, trainingInfo }) {
  const [highlightMetric, setHighlightMetric] = useState('auc');
  const [hoveredModel, setHoveredModel] = useState(null);
  const [cvModel, setCvModel] = useState(null);

  /* Refs for downloadable sections */
  const radarRef = useRef(null);
  const barRef = useRef(null);
  const metricsRef = useRef(null);
  const confusionRef = useRef(null);
  const cvRef = useRef(null);

  /* ---- Download helpers ---- */
  const downloadPNG = useCallback(async (ref, filename) => {
    if (!ref.current) return;
    try {
      // Hide download buttons before capture
      const buttons = ref.current.querySelectorAll('.btn-download, .dash-metric-pills');
      buttons.forEach((el) => (el.style.display = 'none'));

      const canvas = await html2canvas(ref.current, {
        backgroundColor: '#ffffff',
        scale: 2,
        useCORS: true,
        logging: false,
      });

      // Restore buttons
      buttons.forEach((el) => (el.style.display = ''));

      const url = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
    } catch (e) {
      console.error('Download failed:', e);
    }
  }, []);

  const downloadTableCSV = useCallback((headers, rows, filename) => {
    const csv = [
      headers.join(','),
      ...rows.map((r) => r.map((c) => JSON.stringify(c ?? '')).join(',')),
    ].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  if (!data || !data.comparison || data.comparison.length === 0) {
    return (
      <div className="card">
        <div className="loading">
          <p>No training results available yet. Train models first.</p>
        </div>
      </div>
    );
  }

  const { comparison, per_model } = data;

  const modelNames = Object.keys(per_model || {});

  // Auto-select first model for CV panel
  if (cvModel === null && modelNames.length > 0) {
    queueMicrotask(() => setCvModel(modelNames[0]));
  }

  const sorted = [...comparison].sort(
    (a, b) => (b[highlightMetric] || 0) - (a[highlightMetric] || 0)
  );
  const bestModelName = sorted[0]?.model;

  const bestPerMetric = {};
  for (const m of METRICS) {
    bestPerMetric[m] = Math.max(...comparison.map((r) => r[m] || 0));
  }

  /* --- data for the grouped bar chart --- */
  const barData = sorted.map((row) => ({
    model: MODEL_DISPLAY[row.model] || row.model,
    ...Object.fromEntries(METRICS.map((m) => [m, row[m] || 0])),
  }));

  /* --- data for the radar chart --- */
  const radarData = METRICS.map((m) => {
    const entry = { metric: METRIC_LABELS[m] };
    comparison.forEach((row) => {
      entry[MODEL_DISPLAY[row.model] || row.model] = row[m] || 0;
    });
    return entry;
  });

  /* --- KPI row --- */
  const totalTime = comparison.reduce(
    (s, r) => s + (r.training_time_sec || 0), 0
  );

  return (
    <>
      {/* ---- Training info ---- */}
      <TrainingInfoPanel info={trainingInfo} />

      {/* ---- KPI cards ---- */}
      <div className="dash-kpi-row">
        <div className="dash-kpi">
          <span className="dash-kpi-value">{comparison.length}</span>
          <span className="dash-kpi-label">Models Trained</span>
        </div>
        <div className="dash-kpi">
          <span className="dash-kpi-value">{fmt(bestPerMetric.auc)}</span>
          <span className="dash-kpi-label">Best AUC</span>
        </div>
        <div className="dash-kpi">
          <span className="dash-kpi-value">{fmt(bestPerMetric.f1)}</span>
          <span className="dash-kpi-label">Best F1</span>
        </div>
        <div className="dash-kpi">
          <span className="dash-kpi-value">{fmtTime(totalTime)}</span>
          <span className="dash-kpi-label">Total Train Time</span>
        </div>
      </div>

      {/* ---- Radar chart ---- */}
      <div className="card" ref={radarRef}>
        <div className="dash-section-header">
          <h2>Model Performance Radar</h2>
          <button
            className="btn btn-download"
            onClick={() => downloadPNG(radarRef, 'radar_chart.png')}
            title="Download as PNG"
          >
            ⬇ PNG
          </button>
        </div>
        <ResponsiveContainer width="100%" height={380}>
          <RadarChart data={radarData} outerRadius="75%">
            <PolarGrid stroke="#e2e8f0" />
            <PolarAngleAxis
              dataKey="metric"
              tick={{ fill: '#64748b', fontSize: 12, fontFamily: 'Inter, sans-serif' }}
            />
            <PolarRadiusAxis
              angle={90}
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tick={{ fill: '#94a3b8', fontSize: 10 }}
            />
            {comparison.map((row, i) => {
              const displayName = MODEL_DISPLAY[row.model] || row.model;
              const isHovered = hoveredModel === displayName;
              const someHovered = hoveredModel !== null;
              return (
                <Radar
                  key={row.model}
                  name={displayName}
                  dataKey={displayName}
                  stroke={modelColor(i)}
                  fill={modelColor(i)}
                  fillOpacity={someHovered ? (isHovered ? 0.25 : 0.0) : 0.08}
                  strokeOpacity={someHovered ? (isHovered ? 1 : 0.1) : 1}
                  strokeWidth={isHovered ? 3 : 2}
                  label={isHovered ? (props) => {
                    const { x, y, value } = props;
                    return (
                      <text
                        x={x} y={y}
                        fill={modelColor(i)}
                        fontSize={11}
                        fontWeight={700}
                        fontFamily="Inter, sans-serif"
                        textAnchor="middle"
                        dy={-8}
                      >
                        {value?.toFixed(3)}
                      </text>
                    );
                  } : false}
                />
              );
            })}
            <Legend
              wrapperStyle={{ fontSize: 12, fontFamily: 'Inter, sans-serif', cursor: 'pointer' }}
              onMouseEnter={(e) => setHoveredModel(e.value)}
              onMouseLeave={() => setHoveredModel(null)}
            />
            <Tooltip
              contentStyle={{
                borderRadius: 8,
                fontSize: 12,
                fontFamily: 'Inter, sans-serif',
                boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
                border: 'none',
              }}
              formatter={(v) => v.toFixed(4)}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* ---- Bar chart with metric selector ---- */}
      <div className="card" ref={barRef}>
        <div className="dash-section-header">
          <h2>Model Comparison</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <div className="dash-metric-pills">
              {METRICS.map((m) => (
                <button
                  key={m}
                  className={`pill ${highlightMetric === m ? 'active' : ''}`}
                  onClick={() => setHighlightMetric(m)}
                >
                  {METRIC_LABELS[m]}
                </button>
              ))}
            </div>
            <button
              className="btn btn-download"
              onClick={() => downloadPNG(barRef, 'model_comparison.png')}
              title="Download as PNG"
            >
              ⬇ PNG
            </button>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={barData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              type="number"
              domain={[0, 1]}
              tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'Inter, sans-serif' }}
            />
            <YAxis
              dataKey="model"
              type="category"
              width={120}
              tick={{ fill: '#334155', fontSize: 12, fontWeight: 600, fontFamily: 'Inter, sans-serif' }}
            />
            <Tooltip
              contentStyle={{
                borderRadius: 8,
                fontSize: 12,
                fontFamily: 'Inter, sans-serif',
                boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
                border: 'none',
              }}
              formatter={(v) => v.toFixed(4)}
            />
            <Bar dataKey={highlightMetric} radius={[0, 6, 6, 0]} barSize={24}>
              {barData.map((_, i) => (
                <Cell key={i} fill={modelColor(i)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ---- Full comparison table ---- */}
      <div className="card" ref={metricsRef}>
        <div className="dash-section-header">
          <h2>Test Set Metrics</h2>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              className="btn btn-download"
              onClick={() => downloadPNG(metricsRef, 'test_set_metrics.png')}
              title="Download as PNG"
            >
              ⬇ PNG
            </button>
            <button
              className="btn btn-download"
              onClick={() => {
                const headers = ['Model', ...METRICS.map(m => METRIC_LABELS[m]), 'Time (s)'];
                const rows = sorted.map(row => [
                  MODEL_DISPLAY[row.model] || row.model,
                  ...METRICS.map(m => fmt(row[m])),
                  row.training_time_sec?.toFixed(1) ?? '',
                ]);
                downloadTableCSV(headers, rows, 'test_set_metrics.csv');
              }}
              title="Download as CSV"
            >
              ⬇ CSV
            </button>
          </div>
        </div>
        <div className="results-table-wrapper">
          <table className="results-table">
            <thead>
              <tr>
                <th>Model</th>
                {METRICS.map((m) => (
                  <th key={m}>{METRIC_LABELS[m]}</th>
                ))}
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((row, i) => (
                <tr
                  key={row.model}
                  className={cvModel === row.model ? 'dash-row-selected' : ''}
                  onClick={() => setCvModel(row.model)}
                  style={{ cursor: 'pointer' }}
                >
                  <td>
                    <span className="model-dot" style={{ background: modelColor(i) }} />
                    <strong>{MODEL_DISPLAY[row.model] || row.model}</strong>
                    {row.model === bestModelName && (
                      <span className="dash-best-badge">BEST</span>
                    )}
                  </td>
                  {METRICS.map((m) => (
                    <td key={m}>
                      <span
                        className={`ci-badge ${
                          row[m] === bestPerMetric[m] ? 'high' : row[m] >= 0.5 ? 'medium' : 'low'
                        }`}
                      >
                        {fmt(row[m])}
                      </span>
                    </td>
                  ))}
                  <td>{fmtTime(row.training_time_sec)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ---- Confusion matrix ---- */}
      <div className="card" ref={confusionRef}>
        <div className="dash-section-header">
          <h2>Confusion Matrix (Test Set)</h2>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              className="btn btn-download"
              onClick={() => downloadPNG(confusionRef, 'confusion_matrix.png')}
              title="Download as PNG"
            >
              ⬇ PNG
            </button>
            <button
              className="btn btn-download"
              onClick={() => {
                const headers = ['Model', 'TP', 'TN', 'FP', 'FN', 'Total'];
                const rows = sorted.map(row => {
                  const tp = Math.round(row.tp || 0);
                  const tn = Math.round(row.tn || 0);
                  const fp = Math.round(row.fp || 0);
                  const fn = Math.round(row.fn || 0);
                  return [MODEL_DISPLAY[row.model] || row.model, tp, tn, fp, fn, tp+tn+fp+fn];
                });
                downloadTableCSV(headers, rows, 'confusion_matrix.csv');
              }}
              title="Download as CSV"
            >
              ⬇ CSV
            </button>
          </div>
        </div>
        <div className="results-table-wrapper">
          <table className="results-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>TP</th>
                <th>TN</th>
                <th>FP</th>
                <th>FN</th>
                <th>Total</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((row) => {
                const tp = Math.round(row.tp || 0);
                const tn = Math.round(row.tn || 0);
                const fp = Math.round(row.fp || 0);
                const fn = Math.round(row.fn || 0);
                return (
                  <tr key={row.model}>
                    <td><strong>{MODEL_DISPLAY[row.model] || row.model}</strong></td>
                    <td className="cm-tp">{tp}</td>
                    <td className="cm-tn">{tn}</td>
                    <td className="cm-fp">{fp}</td>
                    <td className="cm-fn">{fn}</td>
                    <td>{tp + tn + fp + fn}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ---- Cross-Validation Results ---- */}
      {per_model && modelNames.length > 0 && cvModel && per_model[cvModel] && (
        <div className="card" ref={cvRef}>
          <div className="dash-section-header">
            <h2>Cross-Validation Folds</h2>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
              <select
                className="cv-model-select"
                value={cvModel}
                onChange={(e) => setCvModel(e.target.value)}
              >
                {modelNames.map((m) => (
                  <option key={m} value={m}>
                    {MODEL_DISPLAY[m] || m}
                  </option>
                ))}
              </select>
              <button
                className="btn btn-download"
                onClick={() => downloadPNG(cvRef, `cv_folds_${cvModel}.png`)}
                title="Download as PNG"
              >
                ⬇ PNG
              </button>
              <button
                className="btn btn-download"
                onClick={() => {
                  const headers = ['Fold', ...METRICS.map(m => METRIC_LABELS[m])];
                  const cvData = per_model[cvModel].cv;
                  const rows = cvData.map((row, i) => [
                    row.fold ?? i + 1,
                    ...METRICS.map(m => fmt(row[m])),
                  ]);
                  // Add average row
                  const avgRow = ['AVG', ...METRICS.map(m => {
                    const vals = cvData.map(r => r[m]).filter(v => v != null && !isNaN(v));
                    return vals.length > 0 ? fmt(vals.reduce((a, b) => a + b, 0) / vals.length) : '—';
                  })];
                  rows.push(avgRow);
                  downloadTableCSV(headers, rows, `cv_folds_${cvModel}.csv`);
                }}
                title="Download as CSV"
              >
                ⬇ CSV
              </button>
            </div>
          </div>
          <div className="results-table-wrapper">
            <table className="results-table">
              <thead>
                <tr>
                  <th>Fold</th>
                  {METRICS.map((m) => (
                    <th key={m}>{METRIC_LABELS[m]}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {per_model[cvModel].cv.map((row, i) => (
                  <tr key={i}>
                    <td><strong>{row.fold ?? i + 1}</strong></td>
                    {METRICS.map((m) => (
                      <td key={m}>{fmt(row[m])}</td>
                    ))}
                  </tr>
                ))}
                <tr className="cv-avg-row">
                  <td><strong>AVG</strong></td>
                  {METRICS.map((m) => {
                    const vals = per_model[cvModel].cv
                      .map((r) => r[m])
                      .filter((v) => v != null && !isNaN(v));
                    const avg =
                      vals.length > 0
                        ? vals.reduce((a, b) => a + b, 0) / vals.length
                        : NaN;
                    return <td key={m}><strong>{fmt(avg)}</strong></td>;
                  })}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
