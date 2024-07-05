import React, { useState } from "react";
import TextField from "@mui/material/TextField";
import IconButton from "@mui/material/IconButton";
import SearchIcon from "@mui/icons-material/Search";
import ClearIcon from "@mui/icons-material/Clear";
import InputAdornment from "@mui/material/InputAdornment";
import Divider from "@mui/material/Divider";
import Box from "@mui/material/Box";
import EmojiPicker from "./EmojiPicker"; // Passe den Import-Pfad entsprechend deiner Projektstruktur an
import { emojiMappings } from "../util/emojiMappings";
import { replaceUnicodeEmojis } from "../util/utils";
import BatchUpload from "./BatchUplaod";

const SearchBar = ({
  query,
  setQuery,
  onSearch,
  onBatchSearch,
  width = "60%",
}) => {
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);

  const handleSearch = () => {
    if (onSearch && query) {
      let translatedQuery = replaceUnicodeEmojis(query, emojiMappings);

      // clean up query
      translatedQuery = translatedQuery.replace(/\s+/g, " ").trim();

      console.log("Translated query:", translatedQuery);
      onSearch(translatedQuery);
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

  const handleSelectEmoji = (emoji) => {
    setQuery((prevQuery) => prevQuery + emoji);
  };

  return (
    <Box
      display="flex"
      justifyContent="center"
      width={width}
      flexDirection="column"
      alignItems="center"
    >
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
              <IconButton
                size="small"
                onClick={() => setShowEmojiPicker((prev) => !prev)}
              >
                😀
              </IconButton>
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
      {showEmojiPicker && (
        <EmojiPicker
          anchorEl={document.querySelector(".MuiInputBase-root")}
          onClose={() => setShowEmojiPicker(false)}
          onSelectEmoji={handleSelectEmoji}
        />
      )}
    </Box>
  );
};

export default SearchBar;
