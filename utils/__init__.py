"""
X-Fashion: Utility Modules
- Image mapping & fallback
- UI helper functions
- Graph-ready item node representation
"""

import os
from typing import Optional, Dict
from dataclasses import asdict
import base64
import json

# ─── Image Category Mapping ────────────────────────────────────────────────────
# Maps item names → icon emoji (used as visual fallback in Streamlit)
ITEM_EMOJIS: Dict[str, str] = {
    # Western tops
    "tshirt": "👕", "blouse": "👚", "crop_top": "👙", "tank_top": "🎽",
    "shirt": "👔", "polo": "👕", "sweater": "🧥", "hoodie": "🧥",
    "turtleneck": "👕", "camisole": "👙",
    # Western bottoms
    "jeans": "👖", "trousers": "👖", "shorts": "🩳", "skirt": "👗",
    "leggings": "🩱", "wide_leg_pants": "👖", "palazzo": "👗",
    "cargo_pants": "👖", "culottes": "👗", "mini_skirt": "👗",
    # Dresses
    "maxi_dress": "👗", "midi_dress": "👗", "mini_dress": "👗",
    "wrap_dress": "👗", "shift_dress": "👗", "bodycon_dress": "👗",
    # Outerwear
    "blazer": "🥼", "jacket": "🧥", "coat": "🧥", "trench_coat": "🧥",
    "denim_jacket": "🧥", "bomber_jacket": "🧥", "cardigan": "🧶",
    "puffer_jacket": "🧥", "windbreaker": "🧥",
    # Shoes
    "sneakers": "👟", "heels": "👠", "boots": "🥾", "loafers": "🥿",
    "sandals": "👡", "flats": "🥿", "ankle_boots": "🥾",
    "oxfords": "👞", "mules": "🥿", "platforms": "👠",
    # Accessories
    "belt": "🪢", "watch": "⌚", "sunglasses": "🕶️", "handbag": "👜",
    "scarf": "🧣", "hat": "🎩", "necklace": "📿", "earrings": "💎",
    "bracelet": "📿", "backpack": "🎒",
    # Ethnic tops
    "kurti": "👘", "kurta": "👘", "blouse_ethnic": "👘", "choli": "👘", "anarkali_top": "👘",
    # Ethnic bottoms
    "salwar": "👗", "palazzo_ethnic": "👗", "lehenga_skirt": "👗",
    "dhoti_pants": "👖", "sharara": "👗",
    # Ethnic full
    "saree": "🥻", "lehenga": "👗", "salwar_kameez": "👘",
    "anarkali": "👘", "churidar_suit": "👘",
    # Ethnic outerwear
    "dupatta": "🧣", "shawl": "🧣", "waistcoat_ethnic": "🥼", "cape_ethnic": "🧥",
    # Ethnic shoes
    "juttis": "👡", "kolhapuri": "👡", "wedges_ethnic": "👡",
    "mojari": "👡", "heels_ethnic": "👠",
    # Ethnic accessories
    "maang_tikka": "💎", "jhumkas": "💎", "bangles": "📿",
    "potli_bag": "👜", "kamarband": "🪢", "nath": "💎",
    # Fallback
    "default": "👗",
}

# Color code mapping for visual swatches
COLOR_HEX: Dict[str, str] = {
    "beige": "#F5F0E8", "camel": "#C19A6B", "rust": "#B7410E", "mustard": "#FFDB58",
    "terracotta": "#E2725B", "coral": "#FF7F50", "peach": "#FFDAB9", "gold": "#FFD700",
    "brown": "#795548", "cream": "#FFFDD0", "ivory": "#FFFFF0", "orange": "#FF8C00",
    "navy": "#001F5B", "cobalt": "#0047AB", "steel_grey": "#71797E", "silver": "#C0C0C0",
    "icy_blue": "#99C5C4", "lavender": "#E6E6FA", "mint": "#98FF98", "teal": "#008080",
    "charcoal": "#36454F", "white": "#FFFFFF", "black": "#000000", "dusty_rose": "#DCAE96",
    "white": "#FFFFFF", "grey": "#808080", "nude": "#F2D2BD", "off_white": "#FAF9F6",
    "red": "#CC0000", "yellow": "#FFE600", "electric_blue": "#0050EF", "hot_pink": "#FF69B4",
    "lime_green": "#32CD32", "magenta": "#FF00FF", "purple": "#800080",
    "midnight_blue": "#191970", "dark_green": "#006400", "maroon": "#800000",
    "dark_brown": "#5C4033", "burgundy": "#800020", "deep_purple": "#4B0082",
    "rani_pink": "#E4007C", "parrot_green": "#61B329", "saffron": "#FF6700",
    "turmeric": "#FFC200", "indigo": "#4B0082", "peacock_blue": "#005F6A",
    "emerald": "#50C878", "peacock_blue": "#005F6A",
    "default": "#888888",
}


def get_color_hex(color_name: str) -> str:
    return COLOR_HEX.get(color_name, COLOR_HEX["default"])


def get_item_emoji(item_name: str) -> str:
    return ITEM_EMOJIS.get(item_name, ITEM_EMOJIS["default"])


def get_image_path(category: str, item_name: str, images_dir: str = "images") -> Optional[str]:
    """
    Try to find an image for the given item.
    Priority: item-specific > category > fallback
    """
    # Try exact item image
    for ext in ["jpg", "png", "jpeg", "webp"]:
        path = os.path.join(images_dir, f"{item_name}.{ext}")
        if os.path.exists(path):
            return path
    # Try category image
    for ext in ["jpg", "png", "jpeg", "webp"]:
        path = os.path.join(images_dir, f"{category}.{ext}")
        if os.path.exists(path):
            return path
    # Fallback
    for fname in ["default.jpg", "default.png"]:
        path = os.path.join(images_dir, fname)
        if os.path.exists(path):
            return path
    return None


def format_item_name(name: str) -> str:
    """Convert snake_case to Title Case"""
    return name.replace("_", " ").title()


def score_to_stars(score: float) -> str:
    """Convert numeric score to star rating display"""
    filled = round(score / 20)
    return "⭐" * filled + "☆" * (5 - filled)


def score_to_grade(score: float) -> str:
    if score >= 90: return "S"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    return "D"


# ─── Graph Node Representation ─────────────────────────────────────────────────
# These utilities are designed for future GNN integration
# Each clothing item becomes a node; edges encode compatibility

class ItemNode:
    """
    Represents a clothing item as a graph node.
    Features vector ready for GNN embedding.
    """
    FEATURE_KEYS = ["style", "season", "occasion", "body_type", "skin_tone", "color", "fit", "comfort_level"]

    # Vocabularies for one-hot encoding
    VOCABS = {
        "style": ["casual", "streetwear", "minimal", "formal", "sporty", "party", "vintage", "elegant"],
        "season": ["summer", "winter", "spring", "autumn", "all"],
        "occasion": ["daily", "office", "party", "date", "travel", "gym", "formal_event", "wedding", "festival", "casual"],
        "body_type": ["pear", "apple", "hourglass", "rectangle", "petite", "tall", "plus"],
        "skin_tone": ["warm", "cool", "neutral", "deep", "fair", "olive"],
        "color": ["warm", "cool", "neutral", "bright", "dark"],  # uses color family
        "fit": ["loose", "regular", "fitted", "oversized", "wide_leg", "slim", "flared", "structured"],
        "comfort_level": ["loose", "regular", "tight"],
    }

    def __init__(self, item_dict: dict):
        self.node_id = item_dict.get("id", 0)
        self.attributes = item_dict
        self.feature_vector = self._encode()

    def _encode(self) -> list:
        """Encode item attributes as a flat feature vector for ML models."""
        vector = []
        for key in self.FEATURE_KEYS:
            vocab = self.VOCABS.get(key, [])
            val = self.attributes.get(key, "")
            if key == "color":
                # Map color name to family
                from recommendation_engine import FashionKnowledge
                val = FashionKnowledge.get_color_family(val)
            one_hot = [1 if v == val else 0 for v in vocab]
            vector.extend(one_hot)
        return vector

    def edge_weight(self, other: "ItemNode") -> float:
        """
        Compute compatibility score between two clothing items.
        This is the edge weight in the fashion graph.
        Used for future GNN training.
        """
        score = 0.0
        # Same style → compatible
        if self.attributes.get("style") == other.attributes.get("style"):
            score += 0.3
        # Same season → compatible
        if self.attributes.get("season") == other.attributes.get("season") or \
           "all" in [self.attributes.get("season"), other.attributes.get("season")]:
            score += 0.2
        # Same occasion → compatible
        if self.attributes.get("occasion") == other.attributes.get("occasion"):
            score += 0.2
        # Color harmony
        from recommendation_engine import FashionKnowledge
        c1, c2 = self.attributes.get("color", ""), other.attributes.get("color", "")
        harmonize, _ = FashionKnowledge.colors_harmonize(c1, c2)
        if harmonize:
            score += 0.3
        return round(min(1.0, score), 3)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "attributes": self.attributes,
            "feature_vector_dim": len(self.feature_vector),
        }


def build_item_graph_sample(df, sample_size: int = 1000) -> dict:
    """
    Build a sample graph representation of fashion items.
    Returns nodes + edges dict for JSON export.
    Used for future GNN training pipeline.
    """
    sample = df.sample(min(sample_size, len(df)))
    nodes = []
    node_objects = []
    for _, row in sample.iterrows():
        node = ItemNode(row.to_dict())
        nodes.append(node.to_dict())
        node_objects.append(node)

    # Build compatibility edges (sparse: only high-compatibility pairs)
    edges = []
    for i in range(min(len(node_objects), 500)):
        for j in range(i + 1, min(len(node_objects), 500)):
            w = node_objects[i].edge_weight(node_objects[j])
            if w >= 0.5:  # only keep strong compatibility edges
                edges.append({"source": node_objects[i].node_id, "target": node_objects[j].node_id, "weight": w})

    return {"nodes": nodes, "edges": edges, "total_nodes": len(nodes), "total_edges": len(edges)}
