import { sortedEmojiKeys } from "./emojiMappings";

export function replaceUnicodeEmojis(input, emojiReplacements) {
  let translatedInput = "";
  let i = 0;

  while (i < input.length) {
    let matched = false;

    for (let emoji of sortedEmojiKeys) {
      if (input.startsWith(emoji, i)) {
        translatedInput += emojiReplacements.get(emoji) + " ";
        i += emoji.length;
        matched = true;
        break;
      }
    }

    if (!matched) {
      translatedInput += input[i];
      i++;
    }
  }
  return translatedInput;
}

export function constructEmojiCategoryLabel(category) {
  switch (category) {
    case "persons":
      return "Persons & Gesture";
    case "food":
      return "Food & Drinks";
    case "travelAndPlaces":
      return "Travel";
    case "activity":
      return "Aktivities";
    case "objects":
      return "Objects";
    case "animalAndNature":
      return "Animals & Nature";
  }
}
