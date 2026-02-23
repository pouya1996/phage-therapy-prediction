export default function Controls({
  topK,
  setTopK,
  view,
  setView,
  threshold,
  setThreshold,
}) {
  return (
    <>
      <div className="controls">
        <div className="control-group">
          <label>View</label>
          <select value={view} onChange={(e) => setView(e.target.value)}>
            <option value="phage">Best per Phage (all concentrations)</option>
            <option value="interaction">Best per Interaction (unique phages)</option>
          </select>
        </div>

        <div className="control-group">
          <label>Top K</label>
          <input
            type="number"
            min={1}
            max={500}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          />
        </div>

        <div className="control-group">
          <label>Threshold</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
        </div>
      </div>

      <div className="control-hints">
        <p><strong>View:</strong> <em>Best per Phage</em> ranks all phage × concentration combinations. <em>Best per Interaction</em> shows only unique phages with their highest-scoring concentration.</p>
        <p><strong>Top K:</strong> Number of top-ranked candidates to display per model.</p>
        <p><strong>Threshold:</strong> Minimum CI score to mark a candidate as viable (shown as ✓ Yes / — No).</p>
      </div>
    </>
  );
}
