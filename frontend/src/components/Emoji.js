import React from "react";
import { Tooltip } from "@mui/material";

const Emoji = ({ symbol, label, onSelect }) => (
  <Tooltip title={label} arrow>
    <span
      role="img"
      aria-label={label ? label : ""}
      aria-hidden={label ? "false" : "true"}
      style={{ fontSize: "1.5rem", margin: "0.2rem", cursor: "pointer" }}
      onClick={() => onSelect(symbol)}
    >
      {symbol}
    </span>
  </Tooltip>
);

export default Emoji;
