import React, { useState } from "react";
import TextField from "@mui/material/TextField";
import IconButton from "@mui/material/IconButton";
import SearchIcon from "@mui/icons-material/Search";
import ClearIcon from "@mui/icons-material/Clear";
import InputAdornment from "@mui/material/InputAdornment";
import Divider from "@mui/material/Divider";
import Box from "@mui/material/Box";

const SearchBar = ({ onSearch }) => {
  const [query, setQuery] = useState("");

  const handleSearch = () => {
    if (onSearch) {
      onSearch(query);
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
    <Box display="flex" alignItems="center">
      <TextField
        variant="outlined"
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
            </InputAdornment>
          ),
        }}
        style={{ width: "60%" }}
      />
    </Box>
  );
};

export default SearchBar;
