"""
X-Fashion v2: Image & Visual Utilities
- SVG outfit illustrations (stable, no API needed)
- Image upload category detection
- Outfit vibe generator
- Hairstyle suggestions
- Shopping links
"""

from typing import Optional, Dict, Tuple
import hashlib

# ─── Color HEX map (merged / extended from utils) ─────────────────────────────
COLOR_HEX: Dict[str, str] = {
    "beige": "#F5F0E8", "camel": "#C19A6B", "rust": "#B7410E", "mustard": "#FFDB58",
    "terracotta": "#E2725B", "coral": "#FF7F50", "peach": "#FFDAB9", "gold": "#FFD700",
    "brown": "#795548", "cream": "#FFFDD0", "ivory": "#FFFFF0", "orange": "#FF8C00",
    "navy": "#001F5B", "cobalt": "#0047AB", "steel_grey": "#71797E", "silver": "#C0C0C0",
    "icy_blue": "#99C5C4", "lavender": "#E6E6FA", "mint": "#98FF98", "teal": "#008080",
    "charcoal": "#36454F", "white": "#FFFFFF", "black": "#000000", "dusty_rose": "#DCAE96",
    "grey": "#808080", "nude": "#F2D2BD", "off_white": "#FAF9F6",
    "red": "#CC0000", "yellow": "#FFE600", "electric_blue": "#0050EF", "hot_pink": "#FF69B4",
    "lime_green": "#32CD32", "magenta": "#FF00FF", "purple": "#800080",
    "midnight_blue": "#191970", "dark_green": "#006400", "maroon": "#800000",
    "dark_brown": "#5C4033", "burgundy": "#800020", "deep_purple": "#4B0082",
    "rani_pink": "#E4007C", "parrot_green": "#61B329", "saffron": "#FF6700",
    "turmeric": "#FFC200", "indigo": "#4B0082", "peacock_blue": "#005F6A",
    "emerald": "#50C878", "default": "#AAAAAA",
}

# ─── Outfit Vibes ──────────────────────────────────────────────────────────────
OUTFIT_VIBES = {
    ("casual", "summer"):   ("Effortless Daytime Energy", "☀️"),
    ("casual", "winter"):   ("Cosy Street Chic", "🧸"),
    ("casual", "spring"):   ("Fresh Bloom Casual", "🌸"),
    ("casual", "autumn"):   ("Warm Toned Weekend Feel", "🍂"),
    ("elegant", "summer"):  ("Breezy Soirée Glow", "✨"),
    ("elegant", "winter"):  ("Velvet Evening Aura", "🌙"),
    ("elegant", "spring"):  ("Soft Romantic Petal", "🌷"),
    ("elegant", "autumn"):  ("Golden Hour Luxe", "🍁"),
    ("formal", "summer"):   ("Crisp Boardroom Power", "💼"),
    ("formal", "winter"):   ("Dark Academia Refined", "📖"),
    ("formal", "spring"):   ("Polished Spring Presence", "🌿"),
    ("formal", "autumn"):   ("Executive Autumn Edge", "🏛️"),
    ("party", "summer"):    ("Neon Sunset Slay", "🌅"),
    ("party", "winter"):    ("Glam Midnight Edit", "🥂"),
    ("party", "spring"):    ("Pastel Party Mood", "🎉"),
    ("party", "autumn"):    ("Dark Glam Statement", "🎭"),
    ("sporty", "summer"):   ("Active Summer Hustle", "🏃"),
    ("sporty", "winter"):   ("Cold-Weather Athlete", "❄️"),
    ("streetwear", "summer"): ("Urban Heat Wave", "🔥"),
    ("streetwear", "winter"): ("Hypebeast Winter Layer", "🧊"),
    ("minimal", "summer"):  ("Clean Girl Aesthetic", "🤍"),
    ("minimal", "winter"):  ("Quiet Luxury Minimalist", "🕊️"),
    ("vintage", "autumn"):  ("Retro Autumnal Nostalgia", "📷"),
    ("vintage", "spring"):  ("Cottage Core Spring", "🌻"),
}

HAIRSTYLE_MAP = {
    ("casual", "pear"):      ["Loose waves", "High bun", "Messy ponytail"],
    ("casual", "apple"):     ["Side-swept blowout", "Half-up half-down", "Slick bun"],
    ("elegant", "hourglass"):["Hollywood waves", "Chignon updo", "Sleek low ponytail"],
    ("formal", "rectangle"): ["Structured blowout", "French twist", "Low bun"],
    ("party", "petite"):     ["Voluminous curls", "High ponytail", "Textured updo"],
    ("sporty", "tall"):      ["High ponytail", "French braid", "Space buns"],
    ("streetwear", "plus"):  ["Bold afro-inspired", "Colourful braids", "Top knot"],
    ("minimal", "hourglass"):["Sleek straight", "Low ponytail", "Effortless curtain bangs"],
    ("vintage", "pear"):     ["Victory rolls", "Soft finger waves", "Pin curls"],
}
DEFAULT_HAIRSTYLES = ["Effortless waves", "Classic blowout", "Sleek bun"]

# ─── Shopping Links ─────────────────────────────────────────────────────────────
SHOP_PLATFORMS = [
    {
        "name": "Myntra",
        "icon": "🛍️",
        "color": "#FF3F6C",
        "url_template": "https://www.myntra.com/{query}",
        "bg": "#fff0f3",
    },
    {
        "name": "Amazon",
        "icon": "📦",
        "color": "#FF9900",
        "url_template": "https://www.amazon.in/s?k={query}",
        "bg": "#fff8ee",
    },
    {
        "name": "Nykaa",
        "icon": "💄",
        "color": "#FC2779",
        "url_template": "https://www.nykaafashion.com/search?q={query}",
        "bg": "#fff0f6",
    },
    {
        "name": "Savana",
        "icon": "🌿",
        "color": "#3A6B4C",
        "url_template": "https://savana.in/search?q={query}",
        "bg": "#f0f7f3",
    },
]


def get_shop_links(item_name: str, color: str = "") -> str:
    """Generate 4 shopping platform links for an item."""
    query = f"{color}+{item_name}".replace("_", "+").strip("+")
    buttons = []
    for p in SHOP_PLATFORMS:
        url = p["url_template"].format(query=query)
        buttons.append(
            f'<a href="{url}" target="_blank" class="shop-btn" '
            f'style="--btn-color:{p["color"]}">'
            f'{p["icon"]} {p["name"]}</a>'
        )
    return '<div class="shop-row">' + "".join(buttons) + "</div>"


# ─── Outfit Vibe ──────────────────────────────────────────────────────────────
def get_outfit_vibe(style: str, weather: str) -> Tuple[str, str]:
    key = (style, weather)
    if key in OUTFIT_VIBES:
        return OUTFIT_VIBES[key]
    # Fallback
    fallbacks = {
        "casual": ("Relaxed Everyday Look", "🌿"),
        "elegant": ("Timeless Elegant Edit", "✨"),
        "formal": ("Sharp Power Dressing", "💼"),
        "party": ("All Eyes On You", "🎉"),
        "sporty": ("Athletic Energy", "⚡"),
        "streetwear": ("Urban Edge", "🔥"),
        "minimal": ("Less Is More", "🤍"),
        "vintage": ("Retro Revival", "📷"),
        "bohemian": ("Free Spirit Boho", "🌙"),
    }
    return fallbacks.get(style, ("Signature Style", "✦"))


def get_hairstyle_suggestions(style: str, body_type: str) -> list:
    return HAIRSTYLE_MAP.get((style, body_type), DEFAULT_HAIRSTYLES)


# ─── SVG Outfit Illustration ───────────────────────────────────────────────────
def generate_outfit_svg(top_color: str, bottom_color: str, shoe_color: str,
                         outer_color: str, outfit_type: str = "western",
                         style: str = "casual") -> str:
    """
    Generates a clean SVG mannequin-style outfit card.
    Uses actual outfit colors. No external dependencies.
    """
    tc = COLOR_HEX.get(top_color, "#C8B8A2")
    bc = COLOR_HEX.get(bottom_color, "#7B8FA1")
    sc = COLOR_HEX.get(shoe_color, "#3D3D3D")
    oc = COLOR_HEX.get(outer_color, "#8B7355")

    # Darken color for shadow effect
    def darken(hex_color: str, factor: float = 0.75) -> str:
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return hex_color
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return "#{:02x}{:02x}{:02x}".format(int(r * factor), int(g * factor), int(b * factor))

    tc_dark = darken(tc)
    bc_dark = darken(bc)
    sc_dark = darken(sc)
    oc_dark = darken(oc)

    is_ethnic = outfit_type == "ethnic"

    if is_ethnic:
        # Saree/Lehenga silhouette
        body_svg = f"""
        <!-- Ethnic silhouette -->
        <!-- Blouse/top -->
        <ellipse cx="100" cy="115" rx="30" ry="22" fill="{tc}" stroke="{tc_dark}" stroke-width="1"/>
        <!-- Dupatta drape -->
        <path d="M70 110 Q55 135 60 200 Q65 220 75 240" stroke="{oc}" stroke-width="12" fill="none" stroke-linecap="round" opacity="0.85"/>
        <!-- Lehenga/skirt flare -->
        <path d="M72 137 Q60 190 50 270 Q75 280 100 280 Q125 280 150 270 Q140 190 128 137 Z" fill="{bc}" stroke="{bc_dark}" stroke-width="1"/>
        <!-- Skirt pattern suggestion -->
        <path d="M72 137 Q80 190 100 200 Q120 190 128 137" fill="{tc}" opacity="0.3"/>
        """
    else:
        # Western silhouette
        body_svg = f"""
        <!-- Top/Shirt -->
        <path d="M75 105 L70 90 L85 82 L100 88 L115 82 L130 90 L125 105 L128 140 L72 140 Z" 
              fill="{tc}" stroke="{tc_dark}" stroke-width="1.5"/>
        <!-- Collar -->
        <path d="M91 82 L100 95 L109 82" fill="none" stroke="{tc_dark}" stroke-width="1.5"/>
        <!-- Sleeves -->
        <path d="M75 105 L60 130 L68 133 L80 110" fill="{tc}" stroke="{tc_dark}" stroke-width="1"/>
        <path d="M125 105 L140 130 L132 133 L120 110" fill="{tc}" stroke="{tc_dark}" stroke-width="1"/>
        <!-- Outerwear layer hint -->
        <path d="M72 108 L65 105 L62 130 L68 133" fill="{oc}" opacity="0.7" stroke="{oc_dark}" stroke-width="1"/>
        <path d="M128 108 L135 105 L138 130 L132 133" fill="{oc}" opacity="0.7" stroke="{oc_dark}" stroke-width="1"/>
        <!-- Bottom/Pants/Skirt -->
        <rect x="72" y="140" width="24" height="110" rx="4" fill="{bc}" stroke="{bc_dark}" stroke-width="1"/>
        <rect x="104" y="140" width="24" height="110" rx="4" fill="{bc}" stroke="{bc_dark}" stroke-width="1.2"/>
        <!-- Waistband -->
        <rect x="72" y="140" width="56" height="10" rx="2" fill="{bc_dark}" opacity="0.5"/>
        """

    svg = f"""<svg viewBox="0 0 200 340" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad{top_color[:3]}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#F8F5F0;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#EDE9E4;stop-opacity:1"/>
    </linearGradient>
    <filter id="softShadow">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-opacity="0.12"/>
    </filter>
  </defs>

  <!-- Background -->
  <rect width="200" height="340" fill="url(#bgGrad{top_color[:3]})" rx="12"/>

  <!-- Neck -->
  <ellipse cx="100" cy="78" rx="12" ry="8" fill="#D4A574" opacity="0.7"/>

  <!-- Head (abstract oval) -->
  <ellipse cx="100" cy="55" rx="22" ry="26" fill="#D4A574"/>
  <!-- Hair suggestion -->
  <ellipse cx="100" cy="38" rx="23" ry="16" fill="#5C3D2E" opacity="0.85"/>

  {body_svg}

  <!-- Shoes -->
  <ellipse cx="84" cy="262" rx="14" ry="6" fill="{sc}" stroke="{sc_dark}" stroke-width="1"/>
  <ellipse cx="116" cy="262" rx="14" ry="6" fill="{sc}" stroke="{sc_dark}" stroke-width="1"/>
  <rect x="70" y="252" width="28" height="12" rx="5" fill="{sc}" stroke="{sc_dark}" stroke-width="1"/>
  <rect x="102" y="252" width="28" height="12" rx="5" fill="{sc}" stroke="{sc_dark}" stroke-width="1"/>

  <!-- Color palette strip at bottom -->
  <rect x="30" y="290" width="22" height="22" rx="4" fill="{tc}" stroke="white" stroke-width="1.5"/>
  <rect x="58" y="290" width="22" height="22" rx="4" fill="{bc}" stroke="white" stroke-width="1.5"/>
  <rect x="86" y="290" width="22" height="22" rx="4" fill="{oc}" stroke="white" stroke-width="1.5"/>
  <rect x="114" y="290" width="22" height="22" rx="4" fill="{sc}" stroke="white" stroke-width="1.5"/>
  <text x="148" y="305" font-size="8" fill="#999" font-family="sans-serif">palette</text>
</svg>"""
    return svg


# ─── Upload category detection ─────────────────────────────────────────────────
UPLOAD_KEYWORDS = {
    "top": ["shirt", "top", "blouse", "tshirt", "kurti", "kurta", "sweat", "hoodie", "tank", "crop", "polo"],
    "bottom": ["jean", "pant", "trouser", "skirt", "short", "legging", "palazzo", "salwar", "dhoti"],
    "dress": ["dress", "gown", "anarkali", "saree", "lehenga", "suit", "kameez"],
    "outerwear": ["jacket", "coat", "blazer", "cardigan", "shrug", "cape", "dupatta", "shawl"],
    "shoes": ["shoe", "heel", "boot", "sandal", "sneak", "loafer", "flat", "jutti", "kolhapuri"],
    "accessories": ["bag", "belt", "watch", "scarf", "hat", "jewel", "necklace", "earring", "bangle"],
}


def detect_category_from_filename(filename: str) -> str:
    """Detect clothing category from uploaded file name."""
    name = filename.lower().replace("_", " ").replace("-", " ")
    for category, keywords in UPLOAD_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return category
    return "top"  # default


def detect_dominant_color_from_name(filename: str) -> str:
    """Try to detect color hint from filename."""
    name = filename.lower()
    from image_utils import COLOR_HEX
    for color in COLOR_HEX:
        if color.replace("_", "") in name.replace("_", "").replace(" ", ""):
            return color
    return "default"
