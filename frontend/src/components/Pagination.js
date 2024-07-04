import React from "react";
import IconButton from "@mui/material/IconButton";
import NavigateBeforeIcon from "@mui/icons-material/NavigateBefore";
import NavigateNextIcon from "@mui/icons-material/NavigateNext";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";

const Pagination = ({ currentPage, totalPages, onPageChange }) => {
  const handlePrevClick = () => {
    onPageChange(currentPage - 1);
  };

  const handleNextClick = () => {
    onPageChange(currentPage + 1);
  };

  return (
    <Box
      display="flex"
      alignItems="center"
      justifyContent="center"
      mt={4}
      mb={6}
    >
      <IconButton
        onClick={handlePrevClick}
        disabled={currentPage === 1}
        aria-label="previous page"
      >
        <NavigateBeforeIcon />
      </IconButton>
      <Typography variant="body1" style={{ margin: "0 16px" }}>
        Page {currentPage} of {totalPages}
      </Typography>
      <IconButton
        onClick={handleNextClick}
        disabled={currentPage === totalPages}
        aria-label="next page"
      >
        <NavigateNextIcon />
      </IconButton>
    </Box>
  );
};

export default Pagination;
