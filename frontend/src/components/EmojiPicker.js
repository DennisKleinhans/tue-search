// EmojiPicker.js

import React, { useState } from "react";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Box from "@mui/material/Box";
import Popover from "@mui/material/Popover";
import Emoji from "./Emoji"; // Passe den Import-Pfad entsprechend deiner Projektstruktur an
import { emojiMap } from "../util/emojiMappings";

const EmojiPicker = ({ anchorEl, onClose, onSelectEmoji }) => {
  const [currentTab, setCurrentTab] = useState("persons"); // Startkategorie festlegen

  const handleChangeTab = (event, newTab) => {
    setCurrentTab(newTab);
  };

  const emojiCategories = Object.keys(emojiMap);

  return (
    <Popover
      open={Boolean(anchorEl)}
      anchorEl={anchorEl}
      onClose={onClose}
      anchorOrigin={{
        vertical: "bottom",
        horizontal: "center", // Mittig unter dem Anchor
      }}
      transformOrigin={{
        vertical: "top",
        horizontal: "center",
      }}
      sx={{
        maxWidth: 600, // Feste Breite für den EmojiPicker
        maxHeight: 400, // Feste Höhe für den EmojiPicker
      }}
    >
      <Tabs
        value={currentTab}
        onChange={handleChangeTab}
        variant="scrollable"
        scrollButtons="auto"
      >
        {emojiCategories.map((category) => (
          <Tab key={category} label={category} value={category} />
        ))}
      </Tabs>
      <Box maxHeight="calc(400px - 48px)" overflow="auto">
        {/* 
          48px abziehen, um Platz für die Tabs zu lassen. 
          Höhe des Emoji-Bereichs bleibt fest bei 400px, 
          und es wird eine Scrollbar hinzugefügt, falls notwendig.
        */}
        {emojiMap[currentTab].map(([emoji, label]) => (
          <Emoji
            key={emoji}
            symbol={emoji}
            label={label}
            onSelect={onSelectEmoji}
          />
        ))}
      </Box>
    </Popover>
  );
};

export default EmojiPicker;
