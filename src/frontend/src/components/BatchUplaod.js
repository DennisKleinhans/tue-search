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
        const lines = text.split("\n").map((line) => line.trim());
        const queries = lines.map((line) => {
          const parts = line.split("\t");
          const queryNumber = parts[0].trim();
          const query = parts[1].trim();
          return { queryNumber, query };
        });
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
