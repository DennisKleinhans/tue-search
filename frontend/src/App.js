import React, { useState } from "react";
import SearchResults from "./components/SearchResults";
import { getExampleResults } from "./utils";
import Title from "./components/Title";
import SearchContainer from "./components/SearchContainer";
import "./App.css";

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

  const clearResults = () => {
    setResults([]);
  };

  return (
    <div
      className="App"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "right",
        height: "100vh",
        marginLeft: "10vw",
      }}
    >
      <Title
        titlePrefix={"TÜ"}
        titleSuffix={"Search"}
        subtitle={"Search engine for Tübingen"}
      />
      <div style={{ width: "60%" }}>
        <SearchContainer onSearch={handleSearch} onClear={clearResults} />
      </div>
      <div style={{ width: "50%", marginTop: "20px" }}>
        <SearchResults results={results} />
      </div>
    </div>
  );
};

export default App;
