import React from "react";
import ResultItem from "./ResultItem";

const SearchResults = ({ results }) => {
  return (
    <div>
      {results.map((result) => (
        <ResultItem key={result.id} result={result} />
      ))}
    </div>
  );
};

export default SearchResults;
