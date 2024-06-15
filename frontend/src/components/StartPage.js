import React from "react";
import Title from "./Title";
import SearchBar from "./SearchBar";

const StartPage = ({ onSearch, onBatchSearch }) => {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "90vh",
        width: "100%",
      }}
    >
      <Title
        titlePrefix={"TÜ"}
        titleSuffix={"Search"}
        subtitle={"Search Engine for Tübingen"}
        size={2}
      />
      <div style={{ height: "20px" }} />
      <SearchBar onSearch={onSearch} onBatchSearch={onBatchSearch} />
    </div>
  );
};

export default StartPage;
