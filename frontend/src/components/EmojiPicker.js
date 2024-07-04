import React, { useState } from "react";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Box from "@mui/material/Box";
import Popover from "@mui/material/Popover";
import Emoji from "./Emoji"; // Passe den Import-Pfad entsprechend deiner Projektstruktur an
import { emojiMap } from "../util/emojiMappings";
import { constructEmojiCategoryLabel } from "../util/utils";

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
        horizontal: "center",
      }}
      transformOrigin={{
        vertical: "top",
        horizontal: "center",
      }}
      sx={{
        maxWidth: 600,
        maxHeight: 400,
      }}
    >
      <Tabs
        value={currentTab}
        onChange={handleChangeTab}
        variant="scrollable"
        scrollButtons="auto"
      >
        {emojiCategories.map((category) => (
          <Tab
            key={category}
            label={constructEmojiCategoryLabel(category)}
            value={category}
          />
        ))}
      </Tabs>
      <Box
        sx={{ maxHeight: "calc(400px - 48px)", overflow: "auto", padding: 1 }}
      >
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
