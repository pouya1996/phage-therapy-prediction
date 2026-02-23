import { useEffect, useState } from 'react';
import FileUpload from './components/FileUpload';
import Controls from './components/Controls';
import ResultsPanel from './components/ResultsPanel';
import Dashboard from './components/Dashboard';
import { predictPhages, fetchResults, fetchTrainingInfo } from './api';
import './index.css';

const TABS = [
  { key: 'predict', label: 'Predict' },
  { key: 'dashboard', label: 'Dashboard' },
];

export default function App() {
  const [tab, setTab] = useState('predict');
  const [file, setFile] = useState(null);
  const [topK, setTopK] = useState(10);
  const [view, setView] = useState('phage');
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [trainingResults, setTrainingResults] = useState(null);
  const [trainingInfo, setTrainingInfo] = useState(null);

  useEffect(() => {
    fetchResults()
      .then(setTrainingResults)
      .catch(() => {});
    fetchTrainingInfo()
      .then(setTrainingInfo)
      .catch(() => {});
  }, []);

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictPhages(file, { top_k: topK, view, threshold });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      {/* ---- Header ---- */}
      <header className="hero-header">
        <div className="hero-glow" />
        <div className="hero-content">
          <div className="hero-icon">🧬</div>
          <h1>Phage Therapy Predictor</h1>
          <p className="hero-subtitle">
            AI-powered phage–host interaction prediction for personalised therapy
          </p>
        </div>
      </header>

      {/* ---- Tab bar ---- */}
      <nav className="tab-bar">
        {TABS.map((t) => (
          <button
            key={t.key}
            className={`tab-btn ${tab === t.key ? 'active' : ''}`}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {/* ---- Predict tab ---- */}
      {tab === 'predict' && (
        <>
          <div className="card">
            <h2>Upload Clinical Isolate</h2>
            <FileUpload file={file} onFileSelect={setFile} />

            <Controls
              topK={topK} setTopK={setTopK}
              view={view} setView={setView}
              threshold={threshold} setThreshold={setThreshold}
            />

            <div className="controls" style={{ marginTop: '1rem' }}>
              <button
                className="btn btn-primary"
                disabled={!file || loading}
                onClick={handleSubmit}
              >
                {loading ? 'Predicting…' : 'Run Prediction'}
              </button>
            </div>
          </div>

          {loading && (
            <div className="card loading-card">
              <div className="loading">
                <div className="dna-helix">
                  <div className="helix-strand">
                    {[...Array(12)].map((_, i) => (
                      <div key={i} className="helix-dot" style={{ animationDelay: `${i * 0.12}s` }} />
                    ))}
                  </div>
                  <div className="helix-strand helix-strand-b">
                    {[...Array(12)].map((_, i) => (
                      <div key={i} className="helix-dot" style={{ animationDelay: `${i * 0.12 + 0.5}s` }} />
                    ))}
                  </div>
                  <div className="helix-glow" />
                </div>
                <div className="loading-text">
                  <p className="loading-title">Analysing Genome</p>
                  <p className="loading-sub">Extracting ORFs · Translating proteins · Running 7 models</p>
                </div>
                <div className="loading-progress">
                  <div className="loading-progress-bar" />
                </div>
              </div>
            </div>
          )}

          {error && <div className="error-banner">⚠ {error}</div>}

          <ResultsPanel data={result} />
        </>
      )}

      {/* ---- Dashboard tab ---- */}
      {tab === 'dashboard' && (
        <Dashboard data={trainingResults} trainingInfo={trainingInfo} />
      )}
    </div>
  );
}
