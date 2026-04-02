"""
X-Fashion: Synthetic Dataset Generator
Generates 50,000–100,000 fashion items with rule-based distributions
"""

import pandas as pd
import numpy as np
import random
import os
import json

# ─── Seed for reproducibility ────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Taxonomy ─────────────────────────────────────────────────────────────────
CATEGORIES = {
    # Western
    "top": ["tshirt", "blouse", "crop_top", "tank_top", "shirt", "polo", "sweater", "hoodie", "turtleneck", "camisole"],
    "bottom": ["jeans", "trousers", "shorts", "skirt", "leggings", "wide_leg_pants", "palazzo", "cargo_pants", "culottes", "mini_skirt"],
    "dress": ["maxi_dress", "midi_dress", "mini_dress", "wrap_dress", "shift_dress", "bodycon_dress"],
    "outerwear": ["blazer", "jacket", "coat", "trench_coat", "denim_jacket", "bomber_jacket", "cardigan", "puffer_jacket", "windbreaker"],
    "shoes": ["sneakers", "heels", "boots", "loafers", "sandals", "flats", "ankle_boots", "oxfords", "mules", "platforms"],
    "accessories": ["belt", "watch", "sunglasses", "handbag", "scarf", "hat", "necklace", "earrings", "bracelet", "backpack"],
    # Indian / Ethnic
    "ethnic_top": ["kurti", "kurta", "blouse_ethnic", "choli", "anarkali_top"],
    "ethnic_bottom": ["salwar", "palazzo_ethnic", "lehenga_skirt", "dhoti_pants", "sharara"],
    "ethnic_full": ["saree", "lehenga", "salwar_kameez", "anarkali", "churidar_suit"],
    "ethnic_outerwear": ["dupatta", "shawl", "waistcoat_ethnic", "cape_ethnic"],
    "ethnic_shoes": ["juttis", "kolhapuri", "wedges_ethnic", "mojari", "heels_ethnic"],
    "ethnic_accessories": ["maang_tikka", "jhumkas", "bangles", "potli_bag", "kamarband", "nath"],
}

COLORS = {
    "warm": ["beige", "camel", "rust", "mustard", "terracotta", "coral", "peach", "gold", "brown", "cream", "ivory", "orange"],
    "cool": ["navy", "cobalt", "steel_grey", "silver", "icy_blue", "lavender", "mint", "teal", "charcoal", "white", "black", "dusty_rose"],
    "neutral": ["white", "black", "grey", "beige", "cream", "nude", "off_white", "charcoal"],
    "bright": ["red", "yellow", "electric_blue", "hot_pink", "lime_green", "magenta", "orange", "purple"],
    "dark": ["black", "midnight_blue", "dark_green", "maroon", "charcoal", "dark_brown", "burgundy", "deep_purple"],
    "ethnic_colors": ["rani_pink", "parrot_green", "saffron", "turmeric", "indigo", "peacock_blue", "maroon", "ivory", "gold", "emerald"],
}

PATTERNS = ["solid", "stripes", "floral", "geometric", "abstract", "checks", "polka_dots", "animal_print",
            "paisley", "block_print", "embroidered", "sequin", "ikat", "batik", "tie_dye"]

STYLES = ["casual", "streetwear", "minimal", "formal", "sporty", "party", "vintage", "elegant", "bohemian", "preppy"]

FITS = {
    "top":        ["loose", "regular", "fitted", "oversized", "cropped"],
    "bottom":     ["slim", "regular", "wide_leg", "skinny", "relaxed", "flared"],
    "dress":      ["fitted", "flowy", "structured", "relaxed"],
    "outerwear":  ["oversized", "regular", "fitted", "structured"],
    "shoes":      ["regular"],
    "accessories":["regular"],
    "ethnic_top": ["kurta_straight", "kurta_flared", "fitted", "regular"],
    "ethnic_bottom":["straight", "flared", "fitted", "gathered"],
    "ethnic_full":["draped", "lehenga_flared", "fitted", "flowy"],
    "ethnic_outerwear":["drape", "structured", "sheer"],
    "ethnic_shoes":["regular"],
    "ethnic_accessories":["regular"],
}

SEASONS = ["summer", "winter", "spring", "autumn", "all"]
OCCASIONS = ["daily", "office", "party", "date", "travel", "gym", "formal_event", "wedding", "festival", "casual"]
BODY_TYPES = ["pear", "apple", "hourglass", "rectangle", "petite", "tall", "plus"]
SKIN_TONES = ["warm", "cool", "neutral", "deep", "fair", "olive"]
COMFORT_LEVELS = ["loose", "regular", "tight"]
MATERIALS = {
    "summer": ["cotton", "linen", "chiffon", "georgette", "jersey"],
    "winter": ["wool", "fleece", "velvet", "corduroy", "tweed", "knit"],
    "all":    ["polyester", "viscose", "modal", "blend", "denim", "silk"],
}
BRANDS = ["Zara", "H&M", "Mango", "Biba", "Libas", "Fabindia", "W", "Indya", "Max", "Westside",
          "Myntra", "Ajio", "Uniqlo", "Levi's", "Nike", "Adidas", "Puma", "Only", "Vero Moda", "Global Desi"]
PRICE_RANGES = ["budget", "mid", "premium", "luxury"]

# ─── Rule-based probability tables ─────────────────────────────────────────────
SEASON_CATEGORY_BIAS = {
    "winter": {"outerwear": 3.0, "ethnic_outerwear": 2.0, "bottom": 1.2},
    "summer": {"top": 2.0, "dress": 2.5, "ethnic_full": 1.5, "shoes": 1.3},
    "spring": {"top": 1.5, "outerwear": 1.2, "dress": 1.5},
    "autumn": {"outerwear": 2.0, "ethnic_outerwear": 1.5, "bottom": 1.2},
}

OCCASION_STYLE_BIAS = {
    "office":        {"formal": 3.0, "minimal": 2.0, "elegant": 1.5},
    "party":         {"party": 3.0, "streetwear": 2.0, "vintage": 1.2},
    "gym":           {"sporty": 4.0},
    "formal_event":  {"formal": 3.0, "elegant": 3.0},
    "daily":         {"casual": 3.0, "minimal": 2.0},
    "date":          {"elegant": 2.0, "party": 1.5, "minimal": 1.5},
    "wedding":       {"elegant": 3.0, "formal": 2.0},
    "festival":      {"ethnic_full": 3.0, "casual": 1.5},
    "travel":        {"casual": 2.0, "sporty": 2.0},
    "casual":        {"casual": 3.0, "streetwear": 1.5},
}

BODY_FIT_BIAS = {
    "pear":      {"wide_leg": 3.0, "flared": 2.5, "A-line": 2.0, "slim": 0.3},
    "apple":     {"loose": 3.0, "flowy": 2.0, "oversized": 2.0, "fitted": 0.4},
    "hourglass": {"fitted": 3.0, "regular": 2.0},
    "rectangle": {"structured": 2.5, "oversized": 2.0, "layered": 2.0},
    "petite":    {"regular": 2.0, "fitted": 2.0, "oversized": 0.3},
    "tall":      {"wide_leg": 2.0, "oversized": 2.0, "regular": 1.5},
    "plus":      {"loose": 2.5, "regular": 2.0, "fitted": 0.5},
}

SKINTONE_COLOR_BIAS = {
    "warm":    ["warm"],
    "cool":    ["cool"],
    "neutral": ["neutral", "cool", "warm"],
    "deep":    ["bright", "dark", "ethnic_colors"],
    "fair":    ["cool", "neutral", "bright"],
    "olive":   ["warm", "neutral", "ethnic_colors"],
}

# ─── Helper functions ──────────────────────────────────────────────────────────
def weighted_choice(options, weights=None):
    if weights is None:
        return random.choice(options)
    total = sum(weights)
    r = random.uniform(0, total)
    cumulative = 0
    for opt, w in zip(options, weights):
        cumulative += w
        if r <= cumulative:
            return opt
    return options[-1]


def get_biased_category(season, occasion):
    all_cats = list(CATEGORIES.keys())
    weights = []
    for cat in all_cats:
        w = 1.0
        if season in SEASON_CATEGORY_BIAS:
            w *= SEASON_CATEGORY_BIAS[season].get(cat, 1.0)
        # Ethnic bias for festival/wedding
        if occasion in ["wedding", "festival"] and cat.startswith("ethnic"):
            w *= 2.5
        elif occasion not in ["wedding", "festival"] and cat.startswith("ethnic"):
            w *= 0.5
        weights.append(w)
    return weighted_choice(all_cats, weights)


def get_biased_style(occasion):
    biases = OCCASION_STYLE_BIAS.get(occasion, {})
    weights = [biases.get(s, 1.0) for s in STYLES]
    return weighted_choice(STYLES, weights)


def get_biased_color(skin_tone):
    palettes = SKINTONE_COLOR_BIAS.get(skin_tone, ["neutral"])
    chosen_palette = random.choice(palettes)
    color_pool = COLORS.get(chosen_palette, COLORS["neutral"])
    return random.choice(color_pool)


def get_biased_fit(category, body_type):
    fits = FITS.get(category, ["regular"])
    biases = BODY_FIT_BIAS.get(body_type, {})
    weights = [biases.get(f, 1.0) for f in fits]
    return weighted_choice(fits, weights)


def get_material(season):
    if season in MATERIALS:
        return random.choice(MATERIALS[season])
    return random.choice(MATERIALS["all"])


def generate_item(item_id):
    season = random.choice(SEASONS)
    occasion = random.choice(OCCASIONS)
    body_type = random.choice(BODY_TYPES)
    skin_tone = random.choice(SKIN_TONES)
    comfort = random.choice(COMFORT_LEVELS)

    category = get_biased_category(season, occasion)
    style = get_biased_style(occasion)
    color = get_biased_color(skin_tone)
    fit = get_biased_fit(category, body_type)
    pattern = random.choice(PATTERNS)
    material = get_material(season)
    brand = random.choice(BRANDS)
    price = random.choice(PRICE_RANGES)

    # Sub-category item
    item_options = CATEGORIES[category]
    item_name = random.choice(item_options)

    return {
        "id": item_id,
        "item_name": item_name,
        "category": category,
        "color": color,
        "pattern": pattern,
        "style": style,
        "fit": fit,
        "season": season,
        "occasion": occasion,
        "body_type": body_type,
        "skin_tone": skin_tone,
        "comfort_level": comfort,
        "material": material,
        "brand": brand,
        "price_range": price,
    }


def generate_dataset(n=50000, output_path="data/fashion_dataset.csv"):
    print(f"Generating {n} fashion items...")
    items = [generate_item(i) for i in range(1, n + 1)]
    df = pd.DataFrame(items)

    # Post-processing: enforce a few hard rules
    # Gym items must be sporty
    df.loc[df["occasion"] == "gym", "style"] = "sporty"
    df.loc[df["occasion"] == "gym", "material"] = df.loc[df["occasion"] == "gym", "material"].apply(
        lambda _: random.choice(["jersey", "polyester", "spandex", "mesh"])
    )
    # Winter items → add heavier materials
    df.loc[df["season"] == "winter", "material"] = df.loc[df["season"] == "winter", "material"].apply(
        lambda m: m if m in MATERIALS["winter"] else random.choice(MATERIALS["winter"])
    )
    # Formal events → no sporty
    df.loc[df["occasion"] == "formal_event", "style"] = df.loc[df["occasion"] == "formal_event", "style"].apply(
        lambda s: s if s not in ["sporty", "streetwear"] else "formal"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset saved to {output_path} ({len(df)} rows)")

    # Save metadata
    meta = {
        "total_items": len(df),
        "categories": df["category"].value_counts().to_dict(),
        "styles": df["style"].value_counts().to_dict(),
        "seasons": df["season"].value_counts().to_dict(),
        "occasions": df["occasion"].value_counts().to_dict(),
    }
    with open("data/dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("✅ Metadata saved to data/dataset_meta.json")
    return df


if __name__ == "__main__":
    generate_dataset(n=50000)
