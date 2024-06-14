import React from "react";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";

const Title = ({ titlePrefix, titleSuffix, subtitle }) => {
  return (
    <Box textAlign="left" mb={4} mt={6}>
      <div style={{ display: "flex", alignItems: "center" }}>
        <img
          src={process.env.PUBLIC_URL + "/img/wappen.png"}
          alt="Tübingen Wappen"
          style={{ width: "50px", marginRight: "10px" }}
        />
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
          }}
        >
          <Typography variant="h2" component="h1" gutterBottom>
            <span style={{ color: "#ffcc00" }}>{titlePrefix}</span>
            <span style={{ color: "red", marginLeft: "5px" }}>
              {titleSuffix}
            </span>
          </Typography>
          <div style={{ marginLeft: "25px" }}>
            <Typography variant="subtitle1" color="textSecondary" mt={-4}>
              {subtitle}
            </Typography>
          </div>
        </div>
      </div>
    </Box>
  );
};

export default Title;
