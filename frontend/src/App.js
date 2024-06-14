import React, { useState } from "react";
import SearchBar from "./components/SearchBar";
import SearchResults from "./components/SearchResults";
import { getExampleResults } from "./utils";

const App = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  const handleSearch = async (searchQuery) => {
    setQuery(searchQuery);

    try {
      // simulate API request
      const data = getExampleResults();
      setResults(data);
    } catch (error) {
      console.error("Error fetching search results:", error);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        height: "100vh",
      }}
    >
      <h1>TüSearch</h1>
      <SearchBar onSearch={handleSearch} />
      <SearchResults results={results} />
    </div>
  );
};

export default App;
