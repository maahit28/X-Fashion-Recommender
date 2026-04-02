"""
X-Fashion: Recommendation Engine
Rule-based intelligence + ML scoring pipeline
Graph-ready architecture for future GNN integration
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import random
import os

# ─── Data Classes ──────────────────────────────────────────────────────────────
@dataclass
class UserProfile:
    weather: str
    occasion: str
    outfit_type: str
    body_type: str
    skin_tone: str
    style: str
    comfort: str
    color_preference: str


@dataclass
class ClothingItem:
    id: int
    item_name: str
    category: str
    color: str
    pattern: str
    style: str
    fit: str
    season: str
    occasion: str
    body_type: str
    skin_tone: str
    comfort_level: str
    material: str
    brand: str
    price_range: str
    score: float = 0.0


@dataclass
class Outfit:
    outfit_id: int
    top: Optional[ClothingItem]
    bottom: Optional[ClothingItem]
    outerwear: Optional[ClothingItem]
    shoes: Optional[ClothingItem]
    accessories: Optional[ClothingItem]
    explanations: List[str] = field(default_factory=list)
    score: float = 0.0
    style_label: str = ""
    outfit_type: str = "western"


# ─── Fashion Knowledge Base ────────────────────────────────────────────────────
class FashionKnowledge:
    """Central fashion intelligence: color harmony, body-type rules, season rules."""

    # Color palette families
    WARM_COLORS = {"beige", "camel", "rust", "mustard", "terracotta", "coral", "peach", "gold", "brown", "cream", "ivory", "orange", "saffron", "turmeric"}
    COOL_COLORS = {"navy", "cobalt", "steel_grey", "silver", "icy_blue", "lavender", "mint", "teal", "charcoal", "white", "black", "dusty_rose", "peacock_blue", "indigo"}
    NEUTRAL_COLORS = {"white", "black", "grey", "beige", "cream", "nude", "off_white", "charcoal"}
    BRIGHT_COLORS = {"red", "yellow", "electric_blue", "hot_pink", "lime_green", "magenta", "orange", "purple", "rani_pink", "parrot_green"}
    DARK_COLORS = {"black", "midnight_blue", "dark_green", "maroon", "charcoal", "dark_brown", "burgundy", "deep_purple"}

    # Complementary color pairs (good combinations)
    COMPLEMENTARY_PAIRS = [
        {"navy", "white"}, {"black", "white"}, {"beige", "brown"}, {"camel", "white"},
        {"rust", "cream"}, {"mustard", "navy"}, {"teal", "coral"}, {"grey", "pink"},
        {"black", "gold"}, {"white", "gold"}, {"navy", "gold"}, {"olive", "cream"},
        {"burgundy", "beige"}, {"terracotta", "white"}, {"lavender", "white"},
        {"dusty_rose", "grey"}, {"cobalt", "white"}, {"charcoal", "ivory"},
        {"maroon", "gold"}, {"rani_pink", "gold"}, {"peacock_blue", "gold"},
    ]

    # Color clashes to avoid
    COLOR_CLASHES = [
        {"red", "orange"}, {"red", "hot_pink"}, {"lime_green", "yellow"},
        {"electric_blue", "purple"}, {"orange", "pink"}, {"red", "green"},
    ]

    # Season → allowed/forbidden categories
    SEASON_RULES = {
        "summer": {
            "allowed_materials": ["cotton", "linen", "chiffon", "georgette", "jersey"],
            "forbidden_items": ["sweater", "coat", "turtleneck", "puffer_jacket", "boots", "trench_coat", "woolens"],
            "preferred_fits": ["loose", "regular"],
        },
        "winter": {
            "required_outerwear": True,
            "preferred_items": ["coat", "turtleneck", "sweater", "boots", "ankle_boots"],
            "preferred_materials": ["wool", "fleece", "velvet", "corduroy", "knit"],
        },
        "spring": {
            "optional_outerwear": True,
            "preferred_items": ["cardigan", "blazer", "trench_coat"],
        },
        "autumn": {
            "optional_outerwear": True,
            "preferred_items": ["coat", "jacket", "boots", "ankle_boots"],
        },
    }

    # Body type → style recommendations
    BODY_TYPE_RULES = {
        "pear": {
            "prefer_top": ["loose", "oversized", "flowy"],
            "prefer_bottom": ["wide_leg", "flared", "A-line"],
            "avoid_bottom": ["skinny", "slim"],
            "prefer_outerwear": ["structured", "regular"],
            "explanation": "balances wider hips with volume on top and wide-leg silhouettes",
        },
        "apple": {
            "prefer_top": ["loose", "flowy", "oversized"],
            "prefer_outerwear": ["structured", "fitted"],
            "avoid_top": ["fitted", "cropped"],
            "explanation": "creates definition with structured outerwear and flowy tops",
        },
        "hourglass": {
            "prefer_top": ["fitted", "regular"],
            "prefer_bottom": ["fitted", "regular"],
            "explanation": "showcases curves with fitted silhouettes",
        },
        "rectangle": {
            "prefer_top": ["structured", "oversized"],
            "prefer_outerwear": ["structured", "oversized"],
            "prefer_bottom": ["wide_leg", "flared"],
            "explanation": "creates the illusion of curves with structured and layered pieces",
        },
        "petite": {
            "avoid_top": ["oversized"],
            "avoid_outerwear": ["oversized", "maxi"],
            "prefer_bottom": ["regular", "fitted"],
            "explanation": "avoids oversized pieces that can overwhelm a petite frame",
        },
        "tall": {
            "prefer_outerwear": ["oversized", "long"],
            "prefer_bottom": ["wide_leg", "regular"],
            "explanation": "embraces long coats and wide-fit bottoms that suit a tall frame",
        },
        "plus": {
            "prefer_top": ["loose", "regular", "flowy"],
            "prefer_bottom": ["regular", "wide_leg"],
            "avoid_top": ["fitted", "tight"],
            "explanation": "prioritizes comfort and flow with relaxed silhouettes",
        },
    }

    # Occasion → required style attributes
    OCCASION_RULES = {
        "office": {
            "allowed_styles": ["formal", "minimal", "elegant"],
            "forbidden_styles": ["sporty", "streetwear"],
            "required_items": ["blazer", "shirt", "blouse", "trousers", "loafers", "oxfords"],
        },
        "party": {
            "allowed_styles": ["party", "streetwear", "vintage", "elegant"],
            "preferred_patterns": ["sequin", "solid", "abstract"],
        },
        "gym": {
            "allowed_styles": ["sporty"],
            "required_materials": ["jersey", "polyester", "spandex"],
            "forbidden_styles": ["formal", "elegant", "vintage"],
        },
        "formal_event": {
            "allowed_styles": ["formal", "elegant"],
            "forbidden_styles": ["sporty", "streetwear", "casual"],
        },
        "wedding": {
            "allowed_styles": ["elegant", "formal"],
            "preferred_ethnic": True,
        },
        "festival": {
            "preferred_ethnic": True,
            "allowed_styles": ["casual", "elegant", "party"],
        },
        "date": {
            "allowed_styles": ["elegant", "minimal", "casual", "party"],
        },
        "travel": {
            "allowed_styles": ["casual", "sporty", "minimal"],
            "preferred_fits": ["regular", "loose"],
        },
        "daily": {
            "allowed_styles": ["casual", "minimal", "streetwear"],
        },
    }

    # Outfit type → category mapping
    OUTFIT_CATEGORIES = {
        "western": {
            "top":        ["top", "dress"],
            "bottom":     ["bottom"],
            "outerwear":  ["outerwear"],
            "shoes":      ["shoes"],
            "accessories":["accessories"],
        },
        "ethnic": {
            "top":        ["ethnic_top", "ethnic_full"],
            "bottom":     ["ethnic_bottom"],
            "outerwear":  ["ethnic_outerwear"],
            "shoes":      ["ethnic_shoes"],
            "accessories":["ethnic_accessories"],
        },
        "indo_western": {
            "top":        ["top", "ethnic_top"],
            "bottom":     ["bottom", "ethnic_bottom"],
            "outerwear":  ["outerwear", "ethnic_outerwear"],
            "shoes":      ["shoes", "ethnic_shoes"],
            "accessories":["accessories", "ethnic_accessories"],
        },
        "college": {
            "top":        ["top"],
            "bottom":     ["bottom"],
            "outerwear":  ["outerwear"],
            "shoes":      ["shoes"],
            "accessories":["accessories"],
        },
        "formal": {
            "top":        ["top"],
            "bottom":     ["bottom"],
            "outerwear":  ["outerwear"],
            "shoes":      ["shoes"],
            "accessories":["accessories"],
        },
        "party": {
            "top":        ["top", "dress"],
            "bottom":     ["bottom"],
            "outerwear":  ["outerwear"],
            "shoes":      ["shoes"],
            "accessories":["accessories"],
        },
    }

    @classmethod
    def get_color_family(cls, color: str) -> str:
        if color in cls.WARM_COLORS:
            return "warm"
        if color in cls.COOL_COLORS:
            return "cool"
        if color in cls.NEUTRAL_COLORS:
            return "neutral"
        if color in cls.BRIGHT_COLORS:
            return "bright"
        if color in cls.DARK_COLORS:
            return "dark"
        return "neutral"

    @classmethod
    def colors_harmonize(cls, c1: str, c2: str) -> Tuple[bool, str]:
        pair = {c1, c2}
        if pair in cls.COLOR_CLASHES:
            return False, f"{c1} and {c2} clash"
        if any(pair == comp for comp in cls.COMPLEMENTARY_PAIRS):
            return True, f"{c1} and {c2} are a complementary pair"
        # Same family = harmonious
        f1, f2 = cls.get_color_family(c1), cls.get_color_family(c2)
        if f1 == f2:
            return True, f"both in {f1} palette"
        if f1 == "neutral" or f2 == "neutral":
            return True, "neutral color works with anything"
        return True, "compatible color families"  # default: allow with mild score


# ─── Recommendation Engine ─────────────────────────────────────────────────────
class XFashionEngine:
    """
    Core recommendation engine.
    Phase 1: Rule-based filtering
    Phase 2: Rule-based scoring
    Phase 3: Diversity selection
    Phase 4: Explainability generation
    (Graph-ready: items are nodes, scores are edge weights)
    """

    def __init__(self, dataset_path: str = "data/fashion_dataset.csv"):
        self.df = None
        self.knowledge = FashionKnowledge()
        self._load_dataset(dataset_path)

    def _load_dataset(self, path: str):
        if os.path.exists(path):
            self.df = pd.read_csv(path)
            print(f"✅ Loaded {len(self.df)} items from dataset")
        else:
            print(f"⚠️  Dataset not found at {path}. Generating...")
            from generate_dataset import generate_dataset
            self.df = generate_dataset(n=50000, output_path=path)

    # ── Phase 1: Filtering ─────────────────────────────────────────────────────
    def _filter_dataset(self, profile: UserProfile) -> pd.DataFrame:
        df = self.df.copy()

        # Season filter
        df = df[df["season"].isin([profile.weather, "all"])]

        # Occasion filter (broad match + fallback)
        occ_df = df[df["occasion"].isin([profile.occasion, "daily", "casual"])]
        if len(occ_df) < 100:
            occ_df = df  # fallback: use all

        # Style filter (broad + fallback)
        style_df = occ_df[occ_df["style"].isin([profile.style, "casual"])]
        if len(style_df) < 50:
            style_df = occ_df

        # Comfort filter
        comfort_map = {"loose": ["loose", "regular"], "regular": ["regular", "loose", "tight"], "tight": ["tight", "regular"]}
        allowed_comfort = comfort_map.get(profile.comfort, ["regular"])
        comfort_df = style_df[style_df["comfort_level"].isin(allowed_comfort)]
        if len(comfort_df) < 50:
            comfort_df = style_df

        return comfort_df

    # ── Phase 2: Item Scoring ──────────────────────────────────────────────────
    def _score_item(self, item: pd.Series, profile: UserProfile) -> Tuple[float, List[str]]:
        score = 50.0
        reasons = []

        # Body type fit match
        body_rules = self.knowledge.BODY_TYPE_RULES.get(profile.body_type, {})
        fit = item.get("fit", "regular")
        cat = item.get("category", "")

        if "top" in cat:
            if fit in body_rules.get("prefer_top", []):
                score += 15
                reasons.append(f"{fit} fit selected — suits {profile.body_type} body type")
            if fit in body_rules.get("avoid_top", []):
                score -= 20
        elif "bottom" in cat:
            if fit in body_rules.get("prefer_bottom", []):
                score += 15
                reasons.append(f"{fit} bottom — flattering for {profile.body_type} body type")
            if fit in body_rules.get("avoid_bottom", []):
                score -= 20
        elif "outerwear" in cat:
            if fit in body_rules.get("prefer_outerwear", []):
                score += 10
                reasons.append(f"Outerwear fit complements {profile.body_type} frame")

        # Skin tone color match
        color = item.get("color", "")
        color_family = self.knowledge.get_color_family(color)
        tone_biases = {
            "warm": ["warm"], "cool": ["cool"], "neutral": ["neutral", "cool", "warm"],
            "deep": ["bright", "dark"], "fair": ["cool", "neutral"], "olive": ["warm", "neutral"],
        }
        preferred_families = tone_biases.get(profile.skin_tone, ["neutral"])
        if color_family in preferred_families:
            score += 12
            reasons.append(f"{color.replace('_',' ').title()} complements {profile.skin_tone} skin tone")

        # Color preference
        pref_to_family = {"dark": "dark", "light": "cool", "neutral": "neutral", "bright": "bright"}
        if profile.color_preference != "any":
            target_family = pref_to_family.get(profile.color_preference, "neutral")
            if color_family == target_family:
                score += 10

        # Season material match
        season_rules = self.knowledge.SEASON_RULES.get(profile.weather, {})
        material = item.get("material", "")
        if material in season_rules.get("allowed_materials", []):
            score += 8
            reasons.append(f"{material.title()} fabric — ideal for {profile.weather}")
        if item.get("item_name", "") in season_rules.get("forbidden_items", []):
            score -= 30

        # Occasion match
        occasion_rules = self.knowledge.OCCASION_RULES.get(profile.occasion, {})
        item_style = item.get("style", "")
        if item_style in occasion_rules.get("allowed_styles", [item_style]):
            score += 10
        if item_style in occasion_rules.get("forbidden_styles", []):
            score -= 25

        return max(0.0, min(100.0, score)), reasons

    # ── Phase 3: Outfit Assembly ───────────────────────────────────────────────
    def _get_top_items_for_slot(self, filtered_df: pd.DataFrame, categories: List[str],
                                 profile: UserProfile, n: int = 20) -> List[ClothingItem]:
        slot_df = filtered_df[filtered_df["category"].isin(categories)]
        if slot_df.empty:
            # Fallback: any from these categories in full dataset
            slot_df = self.df[self.df["category"].isin(categories)]
        if slot_df.empty:
            return []

        # Score each item
        scored = []
        for _, row in slot_df.sample(min(len(slot_df), 200)).iterrows():
            s, reasons = self._score_item(row, profile)
            item = ClothingItem(**{k: row[k] for k in ClothingItem.__dataclass_fields__ if k in row}, score=s)
            scored.append((item, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored[:n]]

    def _assemble_outfit(self, profile: UserProfile, filtered_df: pd.DataFrame,
                         used_item_ids: set, outfit_id: int) -> Optional[Outfit]:
        cat_map = self.knowledge.OUTFIT_CATEGORIES.get(profile.outfit_type, self.knowledge.OUTFIT_CATEGORIES["western"])
        outfit_type = "ethnic" if profile.outfit_type in ["ethnic"] else ("western" if profile.outfit_type in ["western", "college", "formal", "party"] else "indo_western")

        slots = {}
        for slot_name, categories in cat_map.items():
            candidates = self._get_top_items_for_slot(filtered_df, categories, profile)
            # Filter out already used items
            candidates = [c for c in candidates if c.id not in used_item_ids]
            if not candidates:
                candidates = self._get_top_items_for_slot(self.df, categories, profile)
            if candidates:
                slots[slot_name] = candidates[0]  # pick best scored
            else:
                slots[slot_name] = None

        # Mark used
        for item in slots.values():
            if item:
                used_item_ids.add(item.id)

        outfit = Outfit(
            outfit_id=outfit_id,
            top=slots.get("top"),
            bottom=slots.get("bottom"),
            outerwear=slots.get("outerwear"),
            shoes=slots.get("shoes"),
            accessories=slots.get("accessories"),
            outfit_type=outfit_type,
        )
        return outfit

    # ── Phase 4: Explainability ────────────────────────────────────────────────
    def _generate_explanations(self, outfit: Outfit, profile: UserProfile) -> List[str]:
        explanations = []
        body_rules = self.knowledge.BODY_TYPE_RULES.get(profile.body_type, {})
        body_desc = body_rules.get("explanation", f"suits {profile.body_type} body type")

        # Top explanation
        if outfit.top:
            item = outfit.top
            _, reasons = self._score_item_to_dict(item, profile)
            fit_explanation = f"{item.fit.replace('_',' ')} fit on top — {body_desc}"
            explanations.append(fit_explanation)
            if reasons:
                explanations.extend(reasons[:1])

        # Bottom explanation
        if outfit.bottom:
            item = outfit.bottom
            explanations.append(
                f"{item.item_name.replace('_',' ').title()} selected — "
                f"{'wide-leg silhouette balances ' + profile.body_type + ' proportions' if item.fit in ['wide_leg','flared'] else item.fit + ' fit chosen for comfort and proportion'}"
            )

        # Color harmony
        colors_present = [i.color for i in [outfit.top, outfit.bottom, outfit.outerwear, outfit.shoes] if i]
        if len(colors_present) >= 2:
            harmonize, reason = self.knowledge.colors_harmonize(colors_present[0], colors_present[1])
            if harmonize:
                explanations.append(f"Color palette: {reason}")
            else:
                explanations.append(f"⚠️ Color note: {reason} — consider swapping for a neutral")

        # Skin tone
        if outfit.top:
            explanations.append(
                f"{outfit.top.color.replace('_',' ').title()} hue chosen — "
                f"complements {profile.skin_tone} skin tone"
            )

        # Season
        season_specific = {
            "winter": "layered with outerwear for warmth during winter",
            "summer": "lightweight, breathable fabric for summer comfort",
            "spring": "light layering appropriate for spring transitions",
            "autumn": "warm tones and optional layering for autumn weather",
        }
        explanations.append(season_specific.get(profile.weather, f"Outfit suited for {profile.weather} weather"))

        # Occasion
        occasion_specific = {
            "office": "professional silhouette appropriate for office wear",
            "party": "bold styling and evening-ready pieces for party look",
            "gym": "performance-fit materials for gym activity",
            "formal_event": "elegant, structured pieces for formal occasions",
            "date": "elevated casual look perfect for a date night",
            "wedding": "festive, occasion-appropriate ensemble",
            "travel": "comfortable, practical outfit for travel",
            "daily": "relaxed, versatile everyday styling",
        }
        explanations.append(occasion_specific.get(profile.occasion, f"Curated for {profile.occasion}"))

        # Shoes
        if outfit.shoes:
            explanations.append(
                f"{outfit.shoes.item_name.replace('_',' ').title()} — "
                f"{'comfortable footwear for ' + profile.occasion if profile.occasion in ['daily','travel','gym'] else 'completes the overall look'}"
            )

        return explanations[:7]  # cap at 7 for readability

    def _score_item_to_dict(self, item: ClothingItem, profile: UserProfile) -> Tuple[float, List[str]]:
        """Adapter: ClothingItem → dict-like for scorer"""
        row = {
            "fit": item.fit, "category": item.category, "color": item.color,
            "material": item.material, "style": item.style, "item_name": item.item_name
        }
        import pandas as pd
        return self._score_item(pd.Series(row), profile)

    def _score_outfit(self, outfit: Outfit, profile: UserProfile) -> float:
        items = [outfit.top, outfit.bottom, outfit.outerwear, outfit.shoes, outfit.accessories]
        items = [i for i in items if i]
        if not items:
            return 0.0

        item_scores = []
        for item in items:
            s, _ = self._score_item_to_dict(item, profile)
            item_scores.append(s)

        base = np.mean(item_scores)

        # Bonus: all items present
        completeness_bonus = len(items) * 2
        # Bonus: color harmony across outfit
        colors = [i.color for i in items]
        harmony_bonus = 0
        for i in range(len(colors) - 1):
            ok, _ = self.knowledge.colors_harmonize(colors[i], colors[i+1])
            if ok:
                harmony_bonus += 3

        return min(100.0, base + completeness_bonus + harmony_bonus)

    # ── Main Entrypoint ────────────────────────────────────────────────────────
    def recommend(self, profile: UserProfile, n_outfits: int = 5) -> List[Outfit]:
        # Step 1: Filter
        filtered = self._filter_dataset(profile)
        print(f"  ↳ Filtered to {len(filtered)} items for profile")

        # Step 2: Assemble diverse outfits
        outfits = []
        used_item_ids: set = set()

        attempts = 0
        while len(outfits) < n_outfits and attempts < n_outfits * 3:
            attempts += 1
            outfit = self._assemble_outfit(profile, filtered, used_item_ids, len(outfits) + 1)
            if outfit:
                # Step 3: Score outfit
                outfit.score = round(self._score_outfit(outfit, profile), 1)
                # Step 4: Generate explanations
                outfit.explanations = self._generate_explanations(outfit, profile)
                outfit.style_label = profile.style.title()
                outfits.append(outfit)

        # Sort by score descending
        outfits.sort(key=lambda o: o.score, reverse=True)
        return outfits
