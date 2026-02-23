import { useState } from 'react';
import { downloadCSV, downloadReport } from '../api';

const MODEL_DISPLAY = {
  knn: 'KNN',
  svm: 'SVM',
  rf: 'Random Forest',
  xgboost: 'XGBoost',
  adaboost: 'AdaBoost',
  lr: 'Logistic Reg.',
  multiview_cnn: 'Multiview CNN',
};

function ciBadgeClass(score) {
  if (score >= 0.7) return 'high';
  if (score >= 0.4) return 'medium';
  return 'low';
}

export default function ResultsPanel({ data }) {
  const modelNames = data?.metadata?.model_names || (data ? Object.keys(data.models) : []);
  const [activeModel, setActiveModel] = useState(modelNames[0] || null);

  if (!data) return null;

  const { models, metadata } = data;

  // Sync activeModel when data changes
  if (activeModel === null || !modelNames.includes(activeModel)) {
    if (modelNames.length > 0 && activeModel !== modelNames[0]) {
      // Safe: will render with the correct model on next paint
      queueMicrotask(() => setActiveModel(modelNames[0]));
      return null;
    }
  }

  const current = models[activeModel];

  return (
    <>
      {/* Metadata bar */}
      <div className="card">
        <div className="meta-bar">
          <span>
            Interactions: <strong>{metadata.total_interactions.toLocaleString()}</strong>
          </span>
          <span>
            Models: <strong>{modelNames.length}</strong>
          </span>
          <span>
            Time: <strong>{metadata.elapsed_seconds}s</strong>
          </span>
        </div>
      </div>

      {/* Model selector tabs */}
      <div className="model-tab-bar">
        {modelNames.map((m) => (
          <button
            key={m}
            className={`model-tab-btn ${activeModel === m ? 'active' : ''}`}
            onClick={() => setActiveModel(m)}
          >
            {MODEL_DISPLAY[m] || m}
          </button>
        ))}
      </div>

      {/* Active model results */}
      {current && (
        <>
          <div className="card">
            <h2>
              Top {current.rankings.length} Phage Candidates —{' '}
              {MODEL_DISPLAY[activeModel] || activeModel}
            </h2>
            <div className="results-table-wrapper">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Phage ID</th>
                    <th>Morphology</th>
                    <th>Concentration</th>
                    <th>CI Score</th>
                    <th>Feasible</th>
                  </tr>
                </thead>
                <tbody>
                  {current.rankings.map((r) => (
                    <tr key={`${r.phage_id}-${r.concentration}-${r.rank}`}>
                      <td>{r.rank}</td>
                      <td>{r.phage_id}</td>
                      <td>{r.morphology}</td>
                      <td>{r.concentration}</td>
                      <td>
                        <span className={`ci-badge ${ciBadgeClass(r.ci_score)}`}>
                          {r.ci_score.toFixed(4)}
                        </span>
                      </td>
                      <td>
                        <span className={`feasible-tag ${r.feasible ? 'yes' : 'no'}`}>
                          {r.feasible ? '✓ Yes' : '— No'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Download buttons */}
            <div className="download-bar">
              <button
                className="btn btn-secondary"
                onClick={() =>
                  downloadCSV(
                    current.rankings,
                    `phage_rankings_${activeModel}.csv`
                  )
                }
              >
                ⬇ Download CSV
              </button>
              <button
                className="btn btn-secondary"
                onClick={() =>
                  downloadReport(
                    current.report,
                    `recommendation_${activeModel}.txt`
                  )
                }
              >
                ⬇ Download Report
              </button>
            </div>
          </div>

          {/* Full report */}
          <div className="card">
            <h2>Recommendation Report — {MODEL_DISPLAY[activeModel] || activeModel}</h2>
            <div className="report-block">{current.report}</div>
          </div>
        </>
      )}
    </>
  );
}
