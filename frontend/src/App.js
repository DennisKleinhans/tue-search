import React, { useState } from "react";
import SearchBar from "./components/SearchBar";
import SearchResults from "./components/SearchResults";
import { getExampleResults } from "./utils";
import Title from "./components/Title";

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
        alignItems: "right",
        height: "100vh",
        marginLeft: "10vw",
      }}
    >
      <Title
        titlePrefix={"TÜ"}
        titleSuffix={"Search"}
        subtitle={"Die Suchmaschine für Tübingen"}
      />
      <div style={{ width: "60%" }}>
        <SearchBar onSearch={handleSearch} />
      </div>
      <div style={{ width: "50%", marginTop: "20px" }}>
        <SearchResults results={results} />
      </div>
    </div>
  );
};

export default App;
