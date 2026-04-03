# ✦ X-Fashion: Explainable AI Personal Stylist

> A production-grade AI fashion recommendation system with Rule-Based Intelligence, ML Scoring, and Explainable AI output.

---

## 🗂 Project Structure

```
x_fashion/
│
├── app.py                    # Streamlit web app (main entry)
├── recommendation_engine.py  # Core AI engine (filtering + scoring + XAI)
├── generate_dataset.py       # Synthetic dataset generator (50K items)
│
├── utils/
│   └── __init__.py           # Image utils, color mapping, graph node classes
│
├── data/
│   ├── fashion_dataset.csv   # Generated dataset (auto-created)
│   └── dataset_meta.json     # Dataset statistics
│
├── images/                   # (Optional) local clothing images
│   └── default.png           # Fallback image
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn
```

### 2. Generate the dataset

```bash
cd x_fashion
python generate_dataset.py
```

This creates `data/fashion_dataset.csv` with 50,000 fashion items.

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 System Architecture

```
User Input (Sidebar)
        │
        ▼
   UserProfile Dataclass
        │
        ▼
┌────────────────────────────┐
│   XFashionEngine           │
│                            │
│  1. Filter Dataset         │  ← Season / Occasion / Style / Comfort
│  2. Score Items            │  ← Body Type / Skin Tone / Color Harmony
│  3. Assemble Outfits       │  ← Slot-based: Top/Bottom/Outerwear/Shoes/Accessories
│  4. Diversity Check        │  ← Avoid repeated item IDs
│  5. Score Outfits          │  ← Completeness + Harmony bonus
│  6. Generate Explanations  │  ← XAI: per-item reasoning
└────────────────────────────┘
        │
        ▼
  5 Ranked Outfits + Explanations
```

---

## 🎯 Key Features

| Feature | Details |
|---|---|
| **Fashion Intelligence** | 5 rule systems: color harmony, body type, season, occasion, consistency |
| **Explainable AI** | Per-outfit natural language reasoning |
| **Dataset Scale** | 50K–100K items, rule-biased distributions |
| **Outfit Types** | Western, Ethnic (Indian), Indo-Western |
| **Body Types** | 7 types with specific fit rules |
| **Skin Tones** | 6 tones mapped to color families |
| **Scoring** | 0–100 score with grade (S/A/B/C/D) |
| **Diversity** | Used-item tracking prevents repeated outfits |
| **Graph-Ready** | ItemNode class with feature vectors for future GNN |

---

## 🔬 Fashion Intelligence Rules

### Color Harmony
- Warm tones (beige, rust, mustard) → complement warm skin tones
- Cool tones (navy, grey, mint) → complement cool skin tones
- Complementary pair bonuses (navy + white, rust + cream, etc.)
- Color clash detection (red + orange, lime + yellow, etc.)

### Body Type Rules
| Body Type | Recommendation |
|---|---|
| Pear | Wide-leg bottoms, structured outerwear, flowy tops |
| Apple | Flowy tops, structured outerwear, avoid fitted tops |
| Hourglass | Fitted silhouettes top and bottom |
| Rectangle | Structured + layered pieces, wide-leg bottoms |
| Petite | Avoid oversized; regular and fitted preferred |
| Tall | Long coats, wide-leg, oversized outerwear |
| Plus | Loose and regular fits, relaxed silhouettes |

### Season Rules
- Summer: cotton/linen/chiffon; no coats or boots
- Winter: wool/fleece/knit; outerwear required
- Spring/Autumn: optional layering; transitional fabrics

---

## 🕸 Graph Architecture (Future GNN)

Each clothing item is modelled as a graph node (`ItemNode` class in `utils/__init__.py`):

```python
node.feature_vector  # One-hot encoded attribute vector
node.edge_weight(other_node)  # Compatibility score [0, 1]
```

Export a compatibility graph:
```python
from utils import build_item_graph_sample
import pandas as pd
df = pd.read_csv("data/fashion_dataset.csv")
graph = build_item_graph_sample(df, sample_size=1000)
```

This graph can be fed directly to PyTorch Geometric for GNN training.

---

## 📊 Dataset Schema

| Column | Description |
|---|---|
| `id` | Unique item ID |
| `item_name` | Specific item (tshirt, kurti, jeans, etc.) |
| `category` | Category group (top, bottom, ethnic_full, etc.) |
| `color` | Color name |
| `pattern` | Pattern type (solid, floral, ikat, etc.) |
| `style` | Style tag (casual, formal, sporty, etc.) |
| `fit` | Fit type (loose, wide_leg, fitted, etc.) |
| `season` | Season suitability |
| `occasion` | Occasion suitability |
| `body_type` | Best-suited body type |
| `skin_tone` | Best-suited skin tone |
| `comfort_level` | Comfort category |
| `material` | Fabric material |
| `brand` | Brand name |
| `price_range` | budget / mid / premium / luxury |

---

## 🛠 Tech Stack

- **Python 3.9+**
- **Streamlit** — Web interface
- **Pandas + NumPy** — Data processing
- **Scikit-learn** — Ready for ML extensions
- **PyTorch** (optional) — GNN future integration

---

## 📈 Extending the System

1. **Add real images**: Place images in `images/` folder named `{item_name}.jpg`
2. **Add ML model**: Train a compatibility classifier on the graph edge data
3. **Add GNN**: Use `ItemNode.feature_vector` with PyTorch Geometric
4. **Add user history**: Track preferences for personalized scoring
5. **Scale dataset**: Increase `generate_dataset(n=100000)` for 100K items


## 🚧 Currently upgrading to v2

* Real outfit images (Pinterest-style)
* Upload-based outfit generation
* Buy the look + wishlist system
* Multi-page UX
