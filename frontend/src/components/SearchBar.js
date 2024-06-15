import React, { useState } from "react";
import TextField from "@mui/material/TextField";
import IconButton from "@mui/material/IconButton";
import SearchIcon from "@mui/icons-material/Search";
import ClearIcon from "@mui/icons-material/Clear";
import InputAdornment from "@mui/material/InputAdornment";
import Divider from "@mui/material/Divider";
import Box from "@mui/material/Box";
import BatchUpload from "./BatchUplaod";

const SearchBar = ({ onSearch, onBatchSearch, width = "60%" }) => {
  const [query, setQuery] = useState("");

  const handleSearch = (query) => {
    if (onSearch) {
      onSearch(query);
    }
  };

  const handleBatchSearch = (queries) => {
    if (onBatchSearch) {
      onBatchSearch(queries);
    }
  };

  const handleClear = () => {
    setQuery("");
  };

  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <Box display="flex" justifyContent="center" width={width}>
      <TextField
        variant="outlined"
        size="small"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="Search"
        InputProps={{
          endAdornment: (
            <InputAdornment position="end">
              {query && (
                <IconButton size="small" onClick={handleClear}>
                  <ClearIcon style={{ color: "#ccc" }} />
                </IconButton>
              )}
              <Divider
                orientation="vertical"
                flexItem
                style={{
                  height: 28,
                  alignSelf: "center",
                  marginLeft: 8,
                  marginRight: 8,
                  backgroundColor: "#ccc",
                }}
              />
              <IconButton onClick={handleSearch}>
                <SearchIcon />
              </IconButton>
              <BatchUpload onBatchSearch={handleBatchSearch} />
            </InputAdornment>
          ),
          style: {
            borderRadius: 100,
          },
        }}
        sx={{
          width: { width },
        }}
      />
    </Box>
  );
};

export default SearchBar;
