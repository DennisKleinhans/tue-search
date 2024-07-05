import React from "react";

const ResultItem = ({ result }) => {
  return (
    <div
      style={{
        margin: "10px 0",
        padding: "4px",
      }}
    >
      <div style={{ color: "#006621", fontSize: "14px", margin: "5px 0" }}>
        {result.url}
      </div>
      <a
        href={result.url}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          textDecoration: "none",
          color: "#1a0dab",
          fontSize: "20px",
          fontWeight: "bold",
        }}
        onMouseOver={(e) => (e.target.style.textDecoration = "underline")}
        onMouseOut={(e) => (e.target.style.textDecoration = "none")}
      >
        {result.title}
      </a>
      <p style={{ color: "#545454", fontSize: "16px", margin: "5px 0" }}>
        {result.description}
      </p>
    </div>
  );
};

export default ResultItem;
