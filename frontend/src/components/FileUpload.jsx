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
