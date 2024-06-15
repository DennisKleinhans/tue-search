import React, { useState } from "react";
import SearchResults from "./components/SearchResults";
import { getExampleResults } from "./utils";
import StartPage from "./components/StartPage";
import Header from "./components/Header";
import "./App.css";

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

  const handleBatchSearch = async (searchQueries) => {
    console.log("Batch search queries:", searchQueries);
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
            width: "100%",
          }}
        >
          <Header onSearch={handleSearch} onBatchSearch={handleBatchSearch} />
          <div style={{ width: "60%", marginTop: "20px", marginLeft: "250px" }}>
            <SearchResults results={results} />
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
