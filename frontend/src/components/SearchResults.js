import React, { useState } from "react";
import ResultItem from "./ResultItem";
import Pagination from "./Pagination";

const SearchResults = ({ results }) => {
  const resultsPerPage = 10;
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.ceil(results.length / resultsPerPage);

  const showPagination = results.length > 0 && results.length > resultsPerPage;

  const startIndex = (currentPage - 1) * resultsPerPage;
  const endIndex = startIndex + resultsPerPage;
  const currentResults = results.slice(startIndex, endIndex);

  // Navigationsfunktionen
  const goToPage = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  return (
    <div>
      {currentResults.map((result) => (
        <ResultItem key={result.id} result={result} />
      ))}

      {showPagination && (
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={goToPage}
        />
      )}
    </div>
  );
};

export default SearchResults;
