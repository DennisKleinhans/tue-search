import React from "react";
import IconButton from "@mui/material/IconButton";
import DriveFolderUploadIcon from "@mui/icons-material/DriveFolderUpload";

const BatchUpload = ({ onBatchSearch }) => {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        const queries = text
          .split("\n")
          .filter((query) => query.trim() !== "" && query.trim() !== " ");
        onBatchSearch(queries);
      };
      reader.readAsText(file);
    }
  };

  return (
    <div>
      <input
        accept=".txt"
        id="contained-button-file"
        type="file"
        style={{ display: "none" }}
        onChange={handleFileChange}
      />
      <label htmlFor="contained-button-file">
        <IconButton component="span">
          <DriveFolderUploadIcon />
        </IconButton>
      </label>
    </div>
  );
};

export default BatchUpload;
