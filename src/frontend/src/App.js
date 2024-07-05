import React, { useState } from "react";
import axios from "axios";
import SearchResults from "./components/SearchResults";
import { getExampleResults } from "./util/utils";
import StartPage from "./components/StartPage";
import Header from "./components/Header";
import "./App.css";

const App = () => {
  const [query, setQuery] = useState("");
  const [translatedQuery, setTranslatedQuery] = useState("");
  const [results, setResults] = useState([]);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (searchQuery) => {
    setHasSearched(true);
    setTranslatedQuery(searchQuery);

    try {
      console.log("Searching for:", searchQuery);

      const response = await axios.post("http://localhost:5000/search", {
        query: searchQuery,
      });

      setResults(response.data.results);
    } catch (error) {
      console.error("Error fetching search results:", error);
    }
  };

  const handleBatchSearch = async (searchQueries) => {
    // simulate batch search results
    const results = searchQueries.map((query) => `${query}: result`);

    // automatically download the batch results
    const element = document.createElement("a");
    const file = new Blob([results.join("\n")], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = "batch_results.txt";
    document.body.appendChild(element);
    element.click();

    // clean up the DOM
    document.body.removeChild(element);
  };

  return (
    <div className="App">
      {!hasSearched ? (
        <StartPage
          query={query}
          setQuery={setQuery}
          onSearch={handleSearch}
          onBatchSearch={handleBatchSearch}
        />
      ) : (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            height: "100%",
            width: "100%",
          }}
        >
          <Header
            query={query}
            setQuery={setQuery}
            onSearch={handleSearch}
            onBatchSearch={handleBatchSearch}
          />
          <div style={{ width: "60%", marginTop: "20px" }}>
            <SearchResults results={results} />
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
