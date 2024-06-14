import React, { useState } from "react";
import BatchSearch from "./BatchSearch";
import SearchBar from "./SearchBar";
import Switch from "@mui/material/Switch";
import FormControlLabel from "@mui/material/FormControlLabel";
import Box from "@mui/material/Box";

const SearchContainer = ({ onSearch }) => {
  const [batchMode, setBatchMode] = useState(false);

  const handleToggle = () => {
    setBatchMode(!batchMode);
  };

  return (
    <Box>
      <Box mb={2}>
        <FormControlLabel
          control={<Switch checked={batchMode} onChange={handleToggle} />}
          label={batchMode ? "Batch Search" : "Single Search"}
          labelPlacement="start"
        />
      </Box>
      {batchMode ? <BatchSearch /> : <SearchBar onSearch={onSearch} />}
    </Box>
  );
};

export default SearchContainer;
