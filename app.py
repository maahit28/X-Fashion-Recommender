"""
X-Fashion: Explainable AI Personal Stylist
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import os
import sys
import json
import time

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from recommendation_engine import XFashionEngine, UserProfile, Outfit, ClothingItem
from utils import (
    get_color_hex, get_item_emoji, format_item_name,
    score_to_stars, score_to_grade
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="X-Fashion: AI Personal Stylist",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

  :root {
    --bg: #0d0d0f;
    --surface: #161618;
    --card: #1c1c20;
    --border: #2e2e35;
    --accent: #c9a96e;
    --accent2: #7c6ef5;
    --text: #e8e6e1;
    --muted: #888;
    --danger: #e05555;
    --success: #55b98d;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stRadio label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Main header */
  .xf-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
  }
  .xf-header h1 {
    font-size: 3rem;
    background: linear-gradient(135deg, var(--accent) 0%, #e8c88d 50%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
  }
  .xf-header p { color: var(--muted); font-size: 1rem; }

  /* Outfit card */
  .outfit-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.2s;
  }
  .outfit-card:hover { border-color: var(--accent); }

  .outfit-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
  }
  .outfit-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--accent);
  }
  .score-badge {
    background: linear-gradient(135deg, var(--accent) 0%, #c07d2e 100%);
    color: #1a1a1a;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
  }

  /* Item chip */
  .item-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.75rem;
    margin: 0.2rem;
    font-size: 0.82rem;
  }
  .color-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    border: 1px solid rgba(255,255,255,0.2);
  }
  .slot-label {
    color: var(--muted);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.8rem;
    margin-bottom: 0.3rem;
  }

  /* Explanation box */
  .xai-box {
    background: linear-gradient(135deg, rgba(124,110,245,0.08) 0%, rgba(201,169,110,0.08) 100%);
    border: 1px solid rgba(201,169,110,0.2);
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
  }
  .xai-box h4 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent);
    margin-bottom: 0.6rem;
  }
  .xai-item {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
    font-size: 0.83rem;
    color: #ccc;
    line-height: 1.4;
  }
  .xai-dot { color: var(--accent); flex-shrink: 0; }

  /* Stats bar */
  .stat-row {
    display: flex;
    gap: 1rem;
    margin-top: 0.8rem;
  }
  .stat-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    font-size: 0.75rem;
    color: var(--muted);
  }
  .stat-chip span { color: var(--text); font-weight: 500; }

  /* Profile summary */
  .profile-tag {
    display: inline-block;
    background: rgba(201,169,110,0.12);
    border: 1px solid rgba(201,169,110,0.25);
    color: var(--accent);
    border-radius: 20px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    margin: 0.2rem;
  }

  /* Generate button */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #a07828 100%) !important;
    color: #1a1a1a !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em !important;
    margin-top: 0.5rem !important;
  }
  .stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px);
  }

  /* Selectbox styling */
  .stSelectbox > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
  }

  /* Divider */
  hr { border-color: var(--border) !important; }

  /* Score bar */
  .score-bar-bg {
    height: 4px;
    background: var(--border);
    border-radius: 4px;
    margin-top: 0.5rem;
  }
  .score-bar-fill {
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
  }

  /* Empty state */
  .empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--muted);
  }
  .empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; }
  .stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; }
  .stTabs [data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

  /* Spinner */
  .stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "engine" not in st.session_state:
    st.session_state.engine = None
if "outfits" not in st.session_state:
    st.session_state.outfits = []
if "profile" not in st.session_state:
    st.session_state.profile = None


@st.cache_resource(show_spinner=False)
def load_engine():
    return XFashionEngine(dataset_path="data/fashion_dataset.csv")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="xf-header">
  <h1>✦ X-Fashion</h1>
  <p>Explainable AI Personal Stylist · Powered by Rule-Based Intelligence + ML Scoring</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Your Style Profile")
    st.markdown("---")

    weather = st.selectbox("🌤 Weather / Season", ["summer", "winter", "spring", "autumn"], index=0)
    occasion = st.selectbox("📅 Occasion", ["daily", "office", "party", "date", "travel", "gym", "formal_event", "wedding", "festival", "casual"], index=0)
    outfit_type = st.selectbox("👗 Outfit Type", ["western", "ethnic", "indo_western", "college", "formal", "party"], index=0)

    st.markdown("---")
    body_type = st.selectbox("🧍 Body Type", ["pear", "apple", "hourglass", "rectangle", "petite", "tall", "plus"], index=0)
    skin_tone = st.selectbox("🎨 Skin Tone", ["warm", "cool", "neutral", "deep", "fair", "olive"], index=0)

    st.markdown("---")
    style = st.selectbox("💫 Style Vibe", ["casual", "streetwear", "minimal", "formal", "sporty", "party", "vintage", "elegant"], index=0)
    comfort = st.selectbox("🤗 Comfort Level", ["loose", "regular", "tight"], index=1)
    color_pref = st.selectbox("🎨 Color Preference", ["any", "dark", "light", "neutral", "bright"], index=0)

    st.markdown("---")
    n_outfits = st.slider("Number of Outfits", min_value=3, max_value=8, value=5)

    st.markdown("")
    generate_btn = st.button("✨ Generate Outfit Ideas", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="color: #666; font-size: 0.73rem; line-height: 1.6;">
    <b style="color: #c9a96e;">X-Fashion Engine</b><br>
    🧠 Rule-Based Intelligence<br>
    📊 ML Scoring Pipeline<br>
    🔍 Explainable AI Output<br>
    🗂️ 50K+ Item Dataset<br>
    🕸️ Graph-Ready Architecture
    </div>
    """, unsafe_allow_html=True)

# ─── Helper: Render single item ───────────────────────────────────────────────
def render_item_chip(item: ClothingItem, slot: str) -> str:
    if item is None:
        return f'<span class="item-chip" style="opacity:0.4">— not found —</span>'
    emoji = get_item_emoji(item.item_name)
    color_hex = get_color_hex(item.color)
    name = format_item_name(item.item_name)
    color_name = item.color.replace("_", " ").title()
    return (
        f'<div class="slot-label">{slot}</div>'
        f'<span class="item-chip">{emoji} <strong>{name}</strong>'
        f'<span class="color-dot" style="background:{color_hex}" title="{color_name}"></span>'
        f'{color_name} · {item.fit.replace("_"," ")}'
        f'</span>'
    )


def render_outfit_card(outfit: Outfit, rank: int):
    score = outfit.score
    grade = score_to_grade(score)
    stars = score_to_stars(score)

    # Grade color
    grade_colors = {"S": "#FFD700", "A": "#c9a96e", "B": "#55b98d", "C": "#7c6ef5", "D": "#e05555"}
    grade_color = grade_colors.get(grade, "#888")

    card_html = f"""
    <div class="outfit-card">
      <div class="outfit-header">
        <div>
          <div class="outfit-title">Outfit #{rank} · {outfit.style_label} Look</div>
          <div style="font-size:0.78rem; color:#666; margin-top:0.2rem;">{outfit.outfit_type.title()} Style</div>
        </div>
        <div style="text-align:right;">
          <div class="score-badge">Grade {grade}</div>
          <div style="font-size:0.75rem; color:#888; margin-top:0.3rem;">{score}/100</div>
        </div>
      </div>
      <div class="score-bar-bg"><div class="score-bar-fill" style="width:{score}%"></div></div>
      <div style="margin-top:1rem; display:grid; grid-template-columns: 1fr 1fr; gap:0.5rem 1rem;">
    """

    slots = [
        ("TOP", outfit.top), ("BOTTOM", outfit.bottom),
        ("OUTERWEAR", outfit.outerwear), ("SHOES", outfit.shoes),
        ("ACCESSORIES", outfit.accessories),
    ]

    for slot_name, item in slots:
        card_html += render_item_chip(item, slot_name)

    card_html += "</div>"

    # XAI Explanations
    if outfit.explanations:
        card_html += '<div class="xai-box"><h4>🔍 Why this outfit?</h4>'
        for exp in outfit.explanations:
            card_html += f'<div class="xai-item"><span class="xai-dot">◆</span>{exp}</div>'
        card_html += "</div>"

    # Stat row
    items = [i for i in [outfit.top, outfit.bottom, outfit.outerwear, outfit.shoes, outfit.accessories] if i]
    brands = list(set(i.brand for i in items if i.brand))
    prices = list(set(i.price_range for i in items if i.price_range))
    materials = list(set(i.material for i in items if i.material))

    card_html += '<div class="stat-row">'
    if brands:
        card_html += f'<div class="stat-chip">🏷 <span>{", ".join(brands[:2])}</span></div>'
    if prices:
        card_html += f'<div class="stat-chip">💰 <span>{prices[0].title()}</span></div>'
    if materials:
        card_html += f'<div class="stat-chip">🧵 <span>{materials[0].replace("_"," ").title()}</span></div>'
    card_html += f'<div class="stat-chip">⭐ <span>{stars}</span></div>'
    card_html += "</div></div>"

    return card_html


# ─── Main Logic ───────────────────────────────────────────────────────────────
if generate_btn:
    profile = UserProfile(
        weather=weather,
        occasion=occasion,
        outfit_type=outfit_type,
        body_type=body_type,
        skin_tone=skin_tone,
        style=style,
        comfort=comfort,
        color_preference=color_pref,
    )

    with st.spinner("Analysing your style profile and curating outfits..."):
        try:
            engine = load_engine()
            outfits = engine.recommend(profile, n_outfits=n_outfits)
            st.session_state.outfits = outfits
            st.session_state.profile = profile
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.exception(e)

# ─── Display Outfits ──────────────────────────────────────────────────────────
if st.session_state.outfits:
    profile = st.session_state.profile
    outfits = st.session_state.outfits

    # Profile summary
    tags_html = "".join([
        f'<span class="profile-tag">{v.replace("_"," ").title()}</span>'
        for v in [profile.weather, profile.occasion, profile.outfit_type,
                  profile.body_type, profile.skin_tone, profile.style, profile.comfort]
    ])
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
      <div style="color:#888; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">
        Active Profile
      </div>
      {tags_html}
    </div>
    """, unsafe_allow_html=True)

    # Summary stats
    avg_score = sum(o.score for o in outfits) / len(outfits)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Outfits Generated", len(outfits))
    with c2:
        st.metric("Avg Outfit Score", f"{avg_score:.1f}/100")
    with c3:
        best = max(outfits, key=lambda o: o.score)
        st.metric("Best Score", f"{best.score}/100")
    with c4:
        st.metric("Style", profile.style.title())

    st.markdown("---")

    # Tabs: Grid view and ranked view
    tab1, tab2 = st.tabs(["🃏 Outfit Cards", "🏆 Ranked View"])

    with tab1:
        for i, outfit in enumerate(outfits, 1):
            st.markdown(render_outfit_card(outfit, i), unsafe_allow_html=True)

    with tab2:
        # Sorted by score
        sorted_outfits = sorted(outfits, key=lambda o: o.score, reverse=True)
        for rank, outfit in enumerate(sorted_outfits, 1):
            cols = st.columns([0.08, 0.92])
            with cols[0]:
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
                st.markdown(f"<div style='font-size:1.8rem;text-align:center;margin-top:0.5rem'>{medal}</div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(render_outfit_card(outfit, rank), unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">✦</div>
      <h3 style="font-family:'Playfair Display',serif; color:#c9a96e;">Ready to Style You</h3>
      <p>Fill in your style profile in the sidebar and click <strong>Generate Outfit Ideas</strong>.</p>
      <p style="font-size:0.85rem; margin-top:0.5rem;">
        Our AI will analyse your body type, skin tone, occasion, and preferences<br>
        to create personalised outfit recommendations with full explanations.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature showcase
    st.markdown("---")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div style="background:#161618; border:1px solid #2e2e35; border-radius:12px; padding:1.2rem; text-align:center;">
          <div style="font-size:1.8rem">🧠</div>
          <div style="font-family:'Playfair Display',serif; color:#c9a96e; margin:0.5rem 0;">Fashion Intelligence</div>
          <div style="color:#888; font-size:0.83rem">Body-type rules, color harmony, season and occasion filtering built-in</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div style="background:#161618; border:1px solid #2e2e35; border-radius:12px; padding:1.2rem; text-align:center;">
          <div style="font-size:1.8rem">🔍</div>
          <div style="font-family:'Playfair Display',serif; color:#c9a96e; margin:0.5rem 0;">Explainable AI</div>
          <div style="color:#888; font-size:0.83rem">Every outfit recommendation comes with detailed reasoning, not a black box</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div style="background:#161618; border:1px solid #2e2e35; border-radius:12px; padding:1.2rem; text-align:center;">
          <div style="font-size:1.8rem">🗂️</div>
          <div style="font-family:'Playfair Display',serif; color:#c9a96e; margin:0.5rem 0;">50K+ Item Dataset</div>
          <div style="color:#888; font-size:0.83rem">Rule-based synthetic dataset covering Western and Ethnic fashion categories</div>
        </div>
        """, unsafe_allow_html=True)