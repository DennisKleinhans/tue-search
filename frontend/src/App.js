import React from "react";
import SearchBar from "./components/SearchBar";

const App = () => {
  const handleSearch = (query) => {
    // TODO: implement search logic
    console.log("Suche nach:", query);
  };

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <div>
        <h1>Search Engine</h1>
        <SearchBar onSearch={handleSearch} />
      </div>
    </div>
  );
};

export default App;
