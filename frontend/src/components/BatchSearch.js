import React, { useState } from "react";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import { saveAs } from "file-saver";

const BatchSearch = () => {
  const [fileContent, setFileContent] = useState("");
  const [results, setResults] = useState("");

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
      setFileContent(e.target.result);
    };
    reader.readAsText(file);
  };

  const handleSearch = () => {
    // TODO: Implement batch search
  };

  return (
    <div style={{ marginBottom: "20px" }}>
      <div style={{ marginBottom: "10px" }}>
        <input
          accept=".txt"
          id="contained-button-file"
          type="file"
          style={{ display: "none" }}
          onChange={handleFileChange}
        />
        <label htmlFor="contained-button-file">
          <Button variant="contained" component="span">
            Upload File
          </Button>
        </label>
        {fileContent && (
          <Typography variant="subtitle1">File uploaded</Typography>
        )}
      </div>
      <div>
        <Button
          variant="contained"
          color="primary"
          onClick={handleSearch}
          disabled={!fileContent}
        >
          Search Queries
        </Button>
      </div>
    </div>
  );
};

export default BatchSearch;
