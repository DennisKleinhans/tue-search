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
  const endpointSearch = "http://localhost:5000/search";
  const endpointBatchSearch = "http://localhost:5000/batch_search";

  const handleSearch = async (searchQuery) => {
    setHasSearched(true);
    setTranslatedQuery(searchQuery);

    try {
      console.log("Searching for:", searchQuery);

      const response = await axios.post(endpointSearch, {
        query: searchQuery,
      });
      console.log("Search results:", response.data.results);
      setResults(response.data.results);
    } catch (error) {
      console.error("Error fetching search results:", error);
    }
  };

  const handleBatchSearch = async (searchQueries) => {
    try {
      // pack the actual queries and the corresponding query numbers
      const queries = searchQueries.map((query, queryNumber) => ({
        query: query.query,
        queryNumber: queryNumber + 1,
      }));

      const response = await axios.post(endpointBatchSearch, {
        queries: queries,
      });

      if (response.data && response.data.batch_results) {
        // automatically download the batch results
        const batchResults = response.data.batch_results;
        const element = document.createElement("a");

        let fileContent = "";
        batchResults.forEach((result) => {
          result.results.forEach((item) => {
            fileContent += `${result.queryNumber}\t${JSON.stringify(item)}\n`;
          });
        });

        const file = new Blob([fileContent], { type: "text/plain" });
        element.href = URL.createObjectURL(file);
        element.download = "batch_results.txt";
        document.body.appendChild(element);
        element.click();

        // clean up the DOM
        document.body.removeChild(element);
      } else {
        console.error("No batch results found in response:", response);
      }
    } catch (error) {
      console.error("Error fetching batch search results:", error);
    }
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
