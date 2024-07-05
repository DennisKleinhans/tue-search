import React from "react";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";

const Title = ({ titlePrefix, titleSuffix, subtitle, size = 1 }) => {
  const scaleFactor = size;

  return (
    <Box textAlign="left">
      <div style={{ display: "flex", alignItems: "center" }}>
        <img
          src={process.env.PUBLIC_URL + "/img/wappen.png"}
          alt="Tübingen Wappen"
          style={{
            width: `${50 * scaleFactor}px`,
            marginRight: `${10 * scaleFactor}px`,
          }}
        />
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
          }}
        >
          <Typography
            variant="h5"
            component="h1"
            gutterBottom
            style={{ fontSize: `${2.125 * scaleFactor}rem` }}
          >
            <span style={{ color: "#ffcc00" }}>{titlePrefix}</span>
            <span style={{ color: "red" }}>{titleSuffix}</span>
          </Typography>
          <Typography
            variant="subtitle1"
            color="textSecondary"
            mt={`${-20 * scaleFactor}px`}
            style={{ fontSize: `${1 * scaleFactor}rem` }}
          >
            {subtitle}
          </Typography>
        </div>
      </div>
    </Box>
  );
};

export default Title;
