import React from "react";
import SearchBar from "./SearchBar";
import Title from "./Title";
import Box from "@mui/material/Box";

const Header = ({ query, setQuery, onSearch, onBatchSearch }) => {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        width: "100%",
        backgroundColor: "#f8f8f8",
        padding: "10px 0",
        boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "left",
          justifyContent: "left",
          paddingLeft: "20px",
          width: "0%",
        }}
      >
        <Title
          titlePrefix={"TÜ"}
          titleSuffix={"Search"}
          subtitle={""}
          size={0.9}
        />
      </Box>
      <Box
        sx={{
          display: "flex",
          justifyContent: "left",
          alignItems: "left",
          width: "100%",
          paddingLeft: "20px",
        }}
      >
        <SearchBar
          query={query}
          setQuery={setQuery}
          onSearch={onSearch}
          onBatchSearch={onBatchSearch}
          width={"65%"}
        />
      </Box>
    </Box>
  );
};

export default Header;
