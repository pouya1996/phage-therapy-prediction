import { useCallback, useState } from 'react';

export default function FileUpload({ file, onFileSelect }) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) onFileSelect(dropped);
    },
    [onFileSelect]
  );

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const handleClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.fasta,.fa,.fna,.fsa';
    input.onchange = (e) => {
      if (e.target.files[0]) onFileSelect(e.target.files[0]);
    };
    input.click();
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  /* ---- File selected: show success state ---- */
  if (file) {
    return (
      <div className="upload-success">
        <div className="upload-success-icon">✅</div>
        <p className="upload-success-msg">File uploaded successfully</p>

        <div className="upload-file-info">
          <span className="upload-file-icon">🧬</span>
          <div className="upload-file-details">
            <span className="upload-file-name">{file.name}</span>
            <span className="upload-file-size">{formatSize(file.size)}</span>
          </div>
        </div>

        <button
          type="button"
          className="btn btn-outline upload-new-btn"
          onClick={handleClick}
        >
          ↻ Upload New Clinical Isolate
        </button>
      </div>
    );
  }

  /* ---- No file: show drop-zone ---- */
  return (
    <div
      className={`upload-area ${dragOver ? 'drag-over' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={handleClick}
    >
      <div className="icon">📂</div>
      <p>
        <strong>Click or drag &amp; drop</strong> a clinical-isolate FASTA file
      </p>
      <p>.fasta, .fa, .fna, .fsa — up to 10 MB</p>
    </div>
  );
}
