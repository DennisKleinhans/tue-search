import React from "react";

const Emoji = ({ symbol, label, onSelect }) => (
  <span
    role="img"
    aria-label={label ? label : ""}
    aria-hidden={label ? "false" : "true"}
    style={{ fontSize: "1.5rem", margin: "0.2rem", cursor: "pointer" }}
    onClick={() => onSelect(symbol)} // Hier wird die onSelect-Funktion aufgerufen
  >
    {symbol}
  </span>
);

export default Emoji;
