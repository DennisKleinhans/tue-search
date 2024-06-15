import React, { useState } from "react";
import SearchResults from "./components/SearchResults";
import { getExampleResults } from "./utils";
import Title from "./components/Title";
import StartPage from "./components/StartPage";
import "./App.css";
import SearchBar from "./components/SearchBar";

const App = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (searchQuery) => {
    setQuery(searchQuery);
    setHasSearched(true);

    try {
      // simulate API request
      const data = getExampleResults();
      setResults(data);
    } catch (error) {
      console.error("Error fetching search results:", error);
    }
  };

  const handleBatchSearch = async (searchQueries) => {};

  const clearResults = () => {
    setResults([]);
  };

  return (
    <div className="App">
      {!hasSearched ? (
        <StartPage onSearch={handleSearch} onBatchSearch={handleBatchSearch} />
      ) : (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "right",
            height: "100%",
            marginLeft: "10vw",
          }}
        >
          <Title
            titlePrefix={"TÜ"}
            titleSuffix={"Search"}
            subtitle={"Search engine for Tübingen"}
          />
          <SearchBar
            onSearch={handleSearch}
            onBatchSearch={handleBatchSearch}
          />
          <div style={{ width: "50%", marginTop: "20px" }}>
            <SearchResults results={results} />
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
