import pandas as pd
import random

DATASET_SIZE = 10000

tops = [
"tshirt","shirt","blouse","crop_top","tank_top","camisole",
"tube_top","halter_top","wrap_top","peplum_top","hoodie",
"sweater","cardigan","turtleneck","bodysuit"
]

bottoms = [
"jeans","skinny_jeans","wide_leg_jeans","mom_jeans",
"baggy_jeans","trousers","cargo_pants","leggings",
"palazzo","culottes","skirt","mini_skirt","midi_skirt",
"maxi_skirt","skort","shorts","denim_shorts","bike_shorts"
]

dresses = [
"mini_dress","midi_dress","maxi_dress","bodycon_dress",
"wrap_dress","slip_dress","shirt_dress","sundress",
"evening_gown","cocktail_dress","sweater_dress"
]

outerwear = [
"denim_jacket","leather_jacket","bomber_jacket",
"trench_coat","overcoat","parka","blazer","cape","puffer_jacket"
]

shoes = [
"sneakers","running_shoes","high_heels","stilettos",
"boots","ankle_boots","knee_boots","loafers",
"sandals","platforms","wedges","flats"
]

accessories = [
"necklace","choker","earrings","bracelet",
"watch","belt","bag","tote_bag",
"crossbody_bag","clutch","scarf",
"sunglasses","hairband","ring"
]

categories = tops + bottoms + dresses + outerwear + shoes + accessories

colors = [
"black","white","blue","grey","brown",
"beige","pink","red","green","navy",
"purple","orange","yellow","olive"
]

patterns = [
"solid","striped","checked","floral",
"printed","denim","polka_dot"
]

styles = [
"casual","streetwear","minimal","formal",
"sporty","party","vintage","elegant"
]

fits = [
"oversized","regular","slim",
"baggy","wide_leg","skinny","loose"
]

seasons = [
"summer","winter","spring","autumn"
]

occasions = [
"daily","office","party","date",
"travel","gym","formal_event"
]

body_types = [
"pear","apple","hourglass",
"rectangle","petite","tall","plus"
]

skin_tones = [
"warm","cool","neutral","deep","fair","olive"
]

comfort_levels = [
"loose","regular","tight"
]

materials = [
"cotton","denim","wool","linen",
"polyester","silk","leather"
]

brands = [
"zara","hm","uniqlo","gucci","prada",
"nike","adidas","levi","puma","mango"
]

price_ranges = [
"low","medium","high","premium"
]


# Compatibility rules

winter_items = ["coat","parka","puffer_jacket","boots","sweater"]
summer_items = ["tank_top","crop_top","shorts","sandals","sundress"]

pear_body_items = ["wide_leg_jeans","palazzo","a_line_skirt"]
apple_body_items = ["wrap_dress","peplum_top"]
hourglass_items = ["bodycon_dress","belt"]


data = []

for i in range(DATASET_SIZE):

    category = random.choice(categories)

    season = random.choice(seasons)

    if season == "winter":
        if random.random() < 0.3:
            category = random.choice(outerwear)

    if season == "summer":
        if random.random() < 0.3:
            category = random.choice(tops + dresses)

    body_type = random.choice(body_types)

    if body_type == "pear" and random.random() < 0.3:
        category = random.choice(["wide_leg_jeans","palazzo","skirt"])

    if body_type == "apple" and random.random() < 0.3:
        category = random.choice(["wrap_dress","peplum_top"])

    row = {
        "id": i,
        "category": category,
        "color": random.choice(colors),
        "pattern": random.choice(patterns),
        "style": random.choice(styles),
        "fit": random.choice(fits),
        "season": season,
        "occasion": random.choice(occasions),
        "body_type": body_type,
        "skin_tone": random.choice(skin_tones),
        "comfort_level": random.choice(comfort_levels),
        "material": random.choice(materials),
        "brand": random.choice(brands),
        "price_range": random.choice(price_ranges)
    }

    data.append(row)

df = pd.DataFrame(data)

df.to_csv("data.csv", index=False)

print("Fashion dataset generated with", len(df), "items")