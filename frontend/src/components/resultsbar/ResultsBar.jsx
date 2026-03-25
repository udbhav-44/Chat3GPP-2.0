import { useContext, useEffect, useRef } from "react";
import { Context } from "../../context/Context";
import ResultsTable from "../sidebar/ResultsTable";
import "./resultsbar.css";

const ResultsBar = ({ isOpen, onToggle, offset = 0, drawerWidth = 420, onResize }) => {
  const { resultsTable, resultsUpdatedAt } = useContext(Context);
  const lastAutoOpenRef = useRef(null);

  useEffect(() => {
    if (!resultsUpdatedAt) {
      return;
    }
    if (resultsUpdatedAt !== lastAutoOpenRef.current) {
      lastAutoOpenRef.current = resultsUpdatedAt;
      if (!isOpen) {
        onToggle(true);
      }
    }
  }, [resultsUpdatedAt, isOpen, onToggle]);

  const handleResizeStart = (event) => {
    if (!isOpen || !onResize) {
      return;
    }
    event.preventDefault();
    const initialCursor = document.body.style.cursor;
    const initialUserSelect = document.body.style.userSelect;
    document.body.style.cursor = "ew-resize";
    document.body.style.userSelect = "none";

    const handleMove = (moveEvent) => {
      const nextWidth = window.innerWidth - offset - moveEvent.clientX;
      onResize(nextWidth);
    };

    const handleUp = () => {
      document.body.style.cursor = initialCursor;
      document.body.style.userSelect = initialUserSelect;
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
    };

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
  };

  return (
    <div
      className="results-main"
      style={{ right: `${offset}px`, "--drawer-width": `${drawerWidth}px` }}
    >
      <button
        className={`results-toggle ${isOpen ? "open" : "closed"}`}
        onClick={() => onToggle(!isOpen)}
      >
        {isOpen ? "Results →" : "Results ←"}
      </button>
      <div className={`results-bar ${isOpen ? "open" : "closed"}`}>
        <div
          className="drawer-resize-handle"
          role="separator"
          aria-label="Resize results panel"
          onPointerDown={handleResizeStart}
        />
        {isOpen && (
          <div className="panel-header">
            <div>
              <p className="panel-title">Results Table</p>
              <p className="panel-subtitle">Latest CSV output</p>
            </div>
            <span className="panel-pill">
              {resultsTable.rows.length} rows
            </span>
          </div>
        )}
        {isOpen && (
          <div className="panel-meta">
            {resultsUpdatedAt
              ? `Updated ${new Date(resultsUpdatedAt).toLocaleTimeString()}`
              : "No updates yet"}
          </div>
        )}
        <div className="panel-body">
          <ResultsTable
            columns={resultsTable.columns}
            rows={resultsTable.rows}
          />
        </div>
      </div>
    </div>
  );
};

export default ResultsBar;
