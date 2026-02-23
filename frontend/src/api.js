const API_BASE = '/api';

export async function predictPhages(file, options = {}) {
  const {
    top_k = 10,
    view = 'phage',
    threshold = 0.5,
  } = options;

  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams({ top_k, view, threshold });
  const response = await fetch(`${API_BASE}/predict?${params}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server error (${response.status})`);
  }

  return response.json();
}

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('API unreachable');
  return res.json();
}

export async function fetchResults() {
  const res = await fetch(`${API_BASE}/results`);
  if (!res.ok) throw new Error('Failed to fetch results');
  return res.json();
}

export async function fetchTrainingInfo() {
  const res = await fetch(`${API_BASE}/training-info`);
  if (!res.ok) throw new Error('Failed to fetch training info');
  return res.json();
}

export function downloadCSV(rankings, filename = 'phage_rankings.csv') {
  if (!rankings || rankings.length === 0) return;

  const headers = Object.keys(rankings[0]);
  const rows = rankings.map((r) =>
    headers.map((h) => JSON.stringify(r[h] ?? '')).join(',')
  );
  const csv = [headers.join(','), ...rows].join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadReport(reportText, filename = 'recommendation_report.txt') {
  const blob = new Blob([reportText], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
