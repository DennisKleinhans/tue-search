const foodEmojis = [
  ["\u{1F347}", "grape"],
  ["\u{1F348}", "melon"],
  ["\u{1F349}", "watermelon"],
  ["\u{1F34A}", "orange"],
  ["\u{1F34B}", "lemon"],
  ["\u{1F34C}", "banana"],
  ["\u{1F34D}", "pineapple"],
  ["\u{1F96D}", "mango"],
  ["\u{1F34E}", "apple"],
  ["\u{1F34F}", "green apple"],
  ["\u{1F350}", "pear"],
  ["\u{1F351}", "peach"],
  ["\u{1F352}", "cherry"],
  ["\u{1F353}", "strawberry"],
  ["\u{1F95D}", "kiwi"],
  ["\u{1F345}", "tomato"],
  ["\u{1F965}", "coconut"],
  ["\u{1F951}", "avocado"],
  ["\u{1F346}", "eggplant"],
  ["\u{1F954}", "potato"],
  ["\u{1F955}", "carrot"],
  ["\u{1F33D}", "corn"],
  ["\u{1F336}", "hot pepper"],
  ["\u{1F952}", "cucumber"],
  ["\u{1F966}", "broccoli"],
  ["\u{1F344}", "mushroom"],
  ["\u{1F95C}", "peanuts"],
  ["\u{1F330}", "chestnut"],
  ["\u{1F35E}", "bread"],
  ["\u{1F950}", "croissant"],
  ["\u{1F956}", "baguette"],
  ["\u{1F968}", "pretzel"],
  ["\u{1F95E}", "pancakes"],
  ["\u{1F9C0}", "cheese"],
  ["\u{1F969}", "meat"],
  ["\u{1F953}", "bacon"],
  ["\u{1F354}", "hamburger"],
  ["\u{1F35F}", "fries"],
  ["\u{1F355}", "pizza"],
  ["\u{1F32D}", "hot dog"],
  ["\u{1F96A}", "sandwich"],
  ["\u{1F32E}", "taco"],
  ["\u{1F32F}", "burrito"],
  ["\u{1F95A}", "egg"],
  ["\u{1F373}", "cooking"],
  ["\u{1F957}", "salad"],
  ["\u{1F37F}", "popcorn"],
  ["\u{1F9C2}", "salt"],
  ["\u{1F96B}", "canned food"],
  ["\u{1F358}", "rice cracker"],
  ["\u{1F359}", "rice ball"],
  ["\u{1F35A}", "cooked rice"],
  ["\u{1F35B}", "curry rice"],
  ["\u{1F35D}", "spaghetti"],
  ["\u{1F360}", "roasted sweet potato"],
  ["\u{1F362}", "oden"],
  ["\u{1F363}", "sushi"],
  ["\u{1F364}", "fried shrimp"],
  ["\u{1F365}", "fish cake with swirl"],
  ["\u{1F96E}", "moon cake"],
  ["\u{1F361}", "dango"],
  ["\u{1F95F}", "dumpling"],
  ["\u{1F980}", "crab"],
  ["\u{1F99E}", "lobster"],
  ["\u{1F990}", "shrimp"],
  ["\u{1F991}", "squid"],
  ["\u{1F9AA}", "oyster"],
  ["\u{1F368}", "ice cream"],
  ["\u{1F369}", "doughnut"],
  ["\u{1F36A}", "cookie"],
  ["\u{1F382}", "birthday cake"],
  ["\u{1F370}", "shortcake"],
  ["\u{1F9C1}", "cupcake"],
  ["\u{1F967}", "pie"],
  ["\u{1F36B}", "chocolate"],
  ["\u{1F36C}", "candy"],
  ["\u{1F36D}", "lollipop"],
  ["\u{1F36E}", "custard"],
  ["\u{1F36F}", "honey pot"],
  ["\u{1F9DB}", "milk"],
  ["\u{1F376}", "sake"],
  ["\u{1F377}", "wine"],
  ["\u{1F378}", "cocktail"],
  ["\u{1F37A}", "beer"],
  ["\u{1F943}", "whisky"],
  ["\u{1F9C9}", "mate"],
  ["\u{1F9CA}", "ice"],
  ["\u{1F962}", "chopsticks"],
  ["\u{1F944}", "spoon"],
  ["\u{1F52A}", "knife"],
];

const travelAndPlacesEmojis = [
  ["\u{1F697}", "car"],
  ["\u{1F695}", "taxi"],
  ["\u{1F68C}", "bus"],
  ["\u{1F3CE}", "racing car"],
  ["\u{1F693}", "police car"],
  ["\u{1F691}", "ambulance"],
  ["\u{1F692}", "fire engine"],
  ["\u{1F69A}", "delivery truck"],
  ["\u{1F69C}", "tractor"],
  ["\u{1F6F4}", "scooter"],
  ["\u{1F6B2}", "bicycle"],
  ["\u{1F6F5}", "motor scooter"],
  ["\u{1F3CD}", "motorcycle"],
  ["\u{1F682}", "locomotive"],
  ["\u{1F686}", "train"],
  ["\u{1F681}", "helicopter"],
  ["\u{2708}", "airplane"],
  ["\u{1F680}", "rocket"],
  ["\u{1F6F8}", "ufo"],
  ["\u{1F6A2}", "ship"],
  ["\u{1F6A4}", "speedboat"],
  ["\u{1F6E5}", "motor boat"],
  ["\u{1F3E0}", "house"],
];

const activityEmojis = [
  ["\u{1F3A8}", "painting"],
  ["\u{1F3AC}", "movie"],
  ["\u{1F3A4}", "microphone"],
  ["\u{1F3A7}", "headphones"],
  ["\u{1F3B9}", "keyboard"],
  ["\u{1F941}", "drum"],
  ["\u{1F3B7}", "saxophone"],
  ["\u{1F3BA}", "trumpet"],
  ["\u{1F3B8}", "guitar"],
  ["\u{1F3BB}", "violin"],
  ["\u{1F3B2}", "dice"],
  ["\u{1F9E9}", "puzzle"],
  ["\u{1F3AE}", "video game"],
  ["\u{1F579}", "joystick"],
  ["\u{1F3B3}", "bowling"],
  ["\u{1F3C8}", "football"],
  ["\u{1F3C0}", "basketball"],
  ["\u{26BD}", "soccer"],
  ["\u{26BE}", "baseball"],
  ["\u{1F94E}", "softball"],
  ["\u{1F3D0}", "volleyball"],
  ["\u{1F3C9}", "rugby"],
  ["\u{1F3D3}", "ping pong"],
  ["\u{1F3F8}", "badminton"],
  ["\u{1F94A}", "boxing"],
  ["\u{1F3AF}", "dart"],
  ["\u{1F3BF}", "ski"],
  ["\u{1F6F7}", "sled"],
];

const objectEmojis = [
  ["\u{1F4F1}", "phone"],
  ["\u{260E}", "telephone"],
  ["\u{1F4DF}", "pager"],
  ["\u{1F4E0}", "fax"],
  ["\u{1F50B}", "battery"],
  ["\u{1F50C}", "electric plug"],
  ["\u{1F4BB}", "laptop"],
  ["\u{1F4BB}", "computer"],
  ["\u{1F5A8}", "printer"],
  ["\u{2328}", "keyboard"],
  ["\u{1F5B2}", "trackball"],
  ["\u{1F4BD}", "disk"],
  ["\u{1F4C0}", "dvd"],
  ["\u{1F9EE}", "abacus"],
  ["\u{1F4F7}", "camera"],
];

const animalAndNatureEmojis = [
  ["\u{1F436}", "dog"],
  ["\u{1F431}", "cat"],
  ["\u{1F42D}", "mouse"],
  ["\u{1F439}", "hamster"],
  ["\u{1F430}", "rabbit"],
  ["\u{1F98A}", "fox"],
  ["\u{1F43B}", "bear"],
  ["\u{1F43C}", "panda"],
  ["\u{1F428}", "koala"],
  ["\u{1F42F}", "tiger"],
  ["\u{1F981}", "lion"],
  ["\u{1F42E}", "cow"],
  ["\u{1F437}", "pig"],
  ["\u{1F438}", "frog"],
  ["\u{1F412}", "monkey"],
  ["\u{1F414}", "chicken"],
  ["\u{1F427}", "penguin"],
  ["\u{1F426}", "bird"],
  ["\u{1F986}", "duck"],
  ["\u{1F985}", "eagle"],
  ["\u{1F989}", "owl"],
  ["\u{1F987}", "bat"],
  ["\u{1F98B}", "butterfly"],
  ["\u{1F40C}", "snail"],
  ["\u{1F41E}", "ladybug"],
  ["\u{1F41C}", "ant"],
  ["\u{1F997}", "cricket"],
  ["\u{1F577}", "spider"],
  ["\u{1F578}", "spider web"],
  ["\u{1F982}", "scorpion"],
  ["\u{1F99F}", "mosquito"],
  ["\u{1F9A0}", "microbe"],
  ["\u{1F422}", "turtle"],
  ["\u{1F40D}", "snake"],
  ["\u{1F98E}", "lizard"],
  ["\u{1F996}", "t-rex"],
  ["\u{1F995}", "sauropod"],
  ["\u{1F419}", "octopus"],
  ["\u{1F420}", "fish"],
  ["\u{1F42C}", "dolphin"],
  ["\u{1F433}", "whale"],
  ["\u{1F988}", "shark"],
  ["\u{1F40A}", "crocodile"],
  ["\u{1F406}", "leopard"],
  ["\u{1F993}", "zebra"],
  ["\u{1F98D}", "gorilla"],
  ["\u{1F9A7}", "orangutan"],
  ["\u{1F418}", "elephant"],
  ["\u{1F99B}", "hippopotamus"],
  ["\u{1F98F}", "rhinoceros"],
  ["\u{1F42A}", "camel"],
  ["\u{1F99A}", "giraffe"],
  ["\u{1F998}", "kangaroo"],
  ["\u{1F9A6}", "sloth"],
  ["\u{1F9A6}", "otter"],
  ["\u{1F9A8}", "skunk"],
  ["\u{1F9A9}", "flamingo"],
  ["\u{1F415}", "dog"],
  ["\u{1F40E}", "horse"],
  ["\u{1F984}", "unicorn"],
  ["\u{1F98C}", "deer"],
  ["\u{1F410}", "goat"],
  ["\u{1F40F}", "ram"],
  ["\u{1F411}", "sheep"],
  ["\u{1F33F}", "plant"],
  ["\u{1F332}", "tree"],
  ["\u{1F334}", "palm"],
  ["\u{1F335}", "cactus"],
  ["\u{1F33A}", "hibiscus"],
  ["\u{1F33B}", "sunflower"],
  ["\u{1F33C}", "blossom"],
  ["\u{1F337}", "tulip"],
  ["\u{1F339}", "rose"],
  ["\u{1F338}", "cherry blossom"],
  ["\u{1F31E}", "sun"],
  ["\u{1F30E}", "world"],
  ["\u{1F315}", "moon"],
  ["\u{1F320}", "shooting star"],
  ["\u{2B50}", "star"],
  ["\u{1F308}", "rainbow"],
  ["\u{2601}", "cloud"],
  ["\u{1F327}", "rain"],
  ["\u{1F32A}", "tornado"],
  ["\u{1F32B}", "fog"],
  ["\u{1F30A}", "wave"],
];

const personEmojis = [
  ["\u{1F46E}", "police officer"],
  ["\u{1F476}", "baby"],
  ["\u{1F9D2}", "child"],
  ["\u{1F466}", "boy"],
  ["\u{1F467}", "girl"],
  ["\u{1F9D1}", "person"],
  ["\u{1F471}", "woman"],
  ["\u{1F468}", "man"],
  ["\u{1F475}", "grandmother"],
  ["\u{1F474}", "grandfather"],
  ["\u{1F477}", "construction worker"],
  ["\u{1F482}", "guard"],
  ["\u{1F575}", "detective"],
  ["\u{1F468}\u{200D}\u{2695}\u{FE0F}", "doctor"],
  ["\u{1F468}\u{200D}\u{1F33E}", "farmer"],
  ["\u{1F468}\u{200D}\u{1F373}", "cook"],
  ["\u{1F468}\u{200D}\u{1F393}", "student"],
  ["\u{1F468}\u{200D}\u{1F3A4}", "singer"],
  ["\u{1F468}\u{200D}\u{1F3EB}", "teacher"],
  ["\u{1F468}\u{200D}\u{1F3ED}", "factory worker"],
  ["\u{1F468}\u{200D}\u{1F527}", "mechanic"],
  ["\u{1F468}\u{200D}\u{1F52C}", "scientist"],
  ["\u{1F468}\u{200D}\u{1F3A8}", "artist"],
  ["\u{1F468}\u{200D}\u{1F692}", "firefighter"],
  ["\u{1F468}\u{200D}\u{2708}\u{FE0F}", "pilot"],
  ["\u{1F468}\u{200D}\u{1F680}", "astronaut"],
  ["\u{1F468}\u{200D}\u{2696}\u{FE0F}", "judge"],
  ["\u{1F470}", "bride"],
  ["\u{1F935}", "groom"],
  ["\u{1F478}", "princess"],
  ["\u{1F934}", "prince"],
  ["\u{1F385}", "Santa Claus"],
  ["\u{1F9D9}", "mage"],
  ["\u{1F9DB}", "vampire"],
  ["\u{1F9DC}\u{200D}\u{2640}\u{FE0F}", "mermaid"],
  ["\u{1F933}", "selfie"],
  ["\u{1F483}", "dancing"],
  ["\u{1F3CA}\u{200D}\u{2642}\u{FE0F}", "swimming"],
  ["\u{1F48F}", "kiss"],
  ["\u{1F46A}", "family"],
  ["\u{270B}", "hand"],
  ["\u{270C}", "peace"],
  ["\u{1F44D}", "thumb"],
  ["\u{1F44A}", "fist"],
];

export const emojiMap = {
  persons: personEmojis,
  food: foodEmojis,
  travelAndPlaces: travelAndPlacesEmojis,
  activity: activityEmojis,
  objects: objectEmojis,
  animalAndNature: animalAndNatureEmojis,
};

export const emojiMappings = new Map([
  ...personEmojis,
  ...foodEmojis,
  ...travelAndPlacesEmojis,
  ...activityEmojis,
  ...objectEmojis,
  ...animalAndNatureEmojis,
]);

export const sortedEmojiKeys = Array.from(emojiMappings.keys()).sort(
  (a, b) => b.length - a.length
);
