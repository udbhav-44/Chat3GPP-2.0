import "./resultsTable.css";

const ResultsTable = ({ columns, rows }) => {
  const sourcePathColumnRe = /\bsource\s*[_\.-]?\s*path\b/i;
  const nameColumnRe = /(^|[._\s-])(title|name)(?=$|[._\s-])/i;
  const scoreColumnRe = /score/i;

  const isSourcePathColumn = (column) =>
    sourcePathColumnRe.test(String(column || ""));
  const isNameColumn = (column) => nameColumnRe.test(String(column || ""));
  const isScoreColumn = (column) => scoreColumnRe.test(String(column || ""));

  const sourcePathColumn = columns.find(isSourcePathColumn);
  const visibleColumns = columns.filter(
    (column) => !isSourcePathColumn(column) && !isScoreColumn(column)
  );

  if (!visibleColumns.length || !rows.length) {
    return <p className="empty">No results yet</p>;
  }

  const buildSourceHref = (value) => {
    const raw = String(value || "").trim();
    if (!raw) {
      return "";
    }
    if (/^[a-z][a-z0-9+.-]*:\/\//i.test(raw) || /^file:\/\//i.test(raw)) {
      return raw;
    }
    if (/^[a-zA-Z]:[\\/]/.test(raw)) {
      return `file:///${raw.replace(/\\/g, "/")}`;
    }
    if (raw.startsWith("/")) {
      return `file://${raw}`;
    }
    return raw;
  };

  return (
    <div className="results-table">
      <table>
        <thead>
          <tr>
            {visibleColumns.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              {visibleColumns.map((col) => {
                const value = row[col];
                if (isNameColumn(col)) {
                  const sourceValue = sourcePathColumn ? row[sourcePathColumn] : "";
                  const href = buildSourceHref(sourceValue || value);
                  return (
                    <td key={col}>
                      {href ? (
                        <a href={href} target="_blank" rel="noopener noreferrer">
                          {String(value ?? "")}
                        </a>
                      ) : (
                        String(value ?? "")
                      )}
                    </td>
                  );
                }
                return <td key={col}>{value ?? ""}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ResultsTable;
