import React from "react";

const ResultItem = ({ result }) => {
  return (
    <div style={{ border: "1px solid #ccc", padding: "16px", margin: "8px 0" }}>
      <h2>{result.title}</h2>
      <p>{result.description}</p>
      <a href={result.url} target="_blank" rel="noopener noreferrer">
        {result.url}
      </a>
    </div>
  );
};

export default ResultItem;
