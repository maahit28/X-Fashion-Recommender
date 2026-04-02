"""
X-Fashion v3: Explainable AI Personal Stylist
v3 additions:
  - All 6 feature cards are now fully interactive (clickable toggles)
  - Real outfit inspiration images via Unsplash + Pexels fallback
  - Fashion Intelligence panel: colour, body type, season breakdown
  - Explainable AI panel: per-item decision trail
  - Buy The Look panel: full per-item links across 4 platforms
  - Upload panel: guided upload experience
  - Hairstyle panel: detailed hair style cards with face shape info
  - Dataset panel: live bar charts from CSV
ENGINE + AESTHETICS UNCHANGED.
"""

import streamlit as st
import os, sys, base64
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from recommendation_engine import XFashionEngine, UserProfile, Outfit, ClothingItem
from utils import get_color_hex, get_item_emoji, format_item_name, score_to_stars, score_to_grade
from image_utils import (
    generate_outfit_svg, get_shop_links, get_outfit_vibe,
    get_hairstyle_suggestions, detect_category_from_filename,
    detect_dominant_color_from_name, SHOP_PLATFORMS, COLOR_HEX,
)

st.set_page_config(page_title="X-Fashion · AI Stylist", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Outfit:wght@300;400;500;600&display=swap');
:root{--bg:#F7F4F0;--surface:#FFFFFF;--card:#FFFFFF;--sidebar:#1A1814;--border:#E8E2D9;--border2:#D4CCC0;--accent:#B5936B;--accent2:#8B6F47;--accent-soft:#F0E8DC;--ink:#1C1916;--muted:#8A8078;--muted2:#B8B0A6;--success:#5B8A6F;--danger:#C0574A;--tag-bg:#F0EBE4;--radius:14px;--radius-sm:8px;--shadow:0 2px 20px rgba(28,25,22,0.08);--shadow-lg:0 8px 40px rgba(28,25,22,0.13);--transition:0.25s cubic-bezier(0.4,0,0.2,1)}
html,body,[class*="css"]{font-family:'Outfit',sans-serif!important;background:var(--bg)!important;color:var(--ink)!important}
h1,h2,h3,h4{font-family:'Cormorant Garamond',serif!important}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:10px}
section[data-testid="stSidebar"]{background:var(--sidebar)!important;border-right:none!important;box-shadow:4px 0 30px rgba(0,0,0,0.15)}
section[data-testid="stSidebar"] *{color:#D4CCB8!important}
section[data-testid="stSidebar"] .stSelectbox>div>div{background:#28241F!important;border:1px solid #3A352C!important;border-radius:var(--radius-sm)!important;color:#E8E0D0!important}
section[data-testid="stSidebar"] .stSelectbox label,section[data-testid="stSidebar"] .stSlider label,section[data-testid="stSidebar"] .stFileUploader label{color:#6A6058!important;font-size:0.68rem!important;letter-spacing:0.1em!important;text-transform:uppercase}
section[data-testid="stSidebar"] hr{border-color:#2E2A24!important;margin:0.8rem 0!important}
.stButton>button{background:linear-gradient(135deg,#B5936B 0%,#8B6F47 100%)!important;color:#FFFFFF!important;font-family:'Outfit',sans-serif!important;font-weight:600!important;font-size:0.85rem!important;letter-spacing:0.08em!important;text-transform:uppercase;border:none!important;border-radius:10px!important;padding:0.7rem 1.5rem!important;width:100%!important;transition:all var(--transition)!important;box-shadow:0 4px 15px rgba(139,111,71,0.4)!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 25px rgba(139,111,71,0.5)!important}
.block-container{padding:0 2rem 3rem!important;max-width:1400px}
.xf-hero{padding:3rem 0 2rem;border-bottom:1px solid var(--border);margin-bottom:2.5rem}
.xf-wordmark{font-size:0.68rem;font-weight:600;letter-spacing:0.25em;text-transform:uppercase;color:var(--accent);background:var(--accent-soft);padding:0.3rem 0.8rem;border-radius:20px;display:inline-block;margin-bottom:0.7rem}
.xf-hero h1{font-size:2.8rem!important;font-weight:300;letter-spacing:-0.02em;color:var(--ink);margin:0;line-height:1.1}
.xf-hero h1 em{font-style:italic;color:var(--accent)}
.xf-hero p{color:var(--muted);font-size:0.88rem;margin:0.5rem 0 0}
.profile-strip{display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:2rem;align-items:center}
.profile-strip .plabel{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);margin-right:0.3rem}
.ptag{display:inline-flex;align-items:center;gap:0.3rem;background:var(--tag-bg);border:1px solid var(--border2);color:var(--ink);border-radius:20px;padding:0.22rem 0.65rem;font-size:0.73rem;font-weight:500}
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
.outfit-card{background:var(--card);border:1px solid var(--border);border-radius:20px;overflow:hidden;box-shadow:var(--shadow);transition:box-shadow var(--transition),transform var(--transition),border-color var(--transition);animation:fadeUp 0.4s ease both;margin-bottom:1.5rem}
.outfit-card:hover{box-shadow:var(--shadow-lg);transform:translateY(-4px);border-color:var(--accent)}
.outfit-card:nth-child(1){animation-delay:.05s}.outfit-card:nth-child(2){animation-delay:.12s}.outfit-card:nth-child(3){animation-delay:.19s}.outfit-card:nth-child(4){animation-delay:.26s}.outfit-card:nth-child(5){animation-delay:.33s}.outfit-card:nth-child(6){animation-delay:.4s}
.card-illus{position:relative;background:linear-gradient(145deg,#F2EDE6 0%,#E8E0D4 100%);height:320px;display:flex;align-items:center;justify-content:center;overflow:hidden}
.card-illus .insp-img{width:100%;height:320px;object-fit:cover;object-position:top center;display:block}
.card-illus .svg-fallback{height:295px;filter:drop-shadow(0 4px 14px rgba(0,0,0,0.1))}
.card-illus .img-overlay{position:absolute;inset:0;background:linear-gradient(to bottom,transparent 55%,rgba(28,25,22,.45) 100%)}
.grade-pill{position:absolute;top:12px;right:12px;width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'Cormorant Garamond',serif;font-size:1.1rem;font-weight:600;color:white;box-shadow:0 3px 12px rgba(0,0,0,.2);z-index:2}
.vibe-tag{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);background:rgba(255,255,255,.93);backdrop-filter:blur(10px);border-radius:20px;padding:.3rem .9rem;font-size:.73rem;font-weight:500;color:var(--ink);white-space:nowrap;box-shadow:0 2px 10px rgba(0,0,0,.1);z-index:2}
.rank-tag{position:absolute;top:12px;left:12px;background:var(--accent);color:white;border-radius:20px;padding:.22rem .65rem;font-size:.68rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;z-index:2}
.insp-src{position:absolute;bottom:12px;right:12px;background:rgba(255,255,255,.75);backdrop-filter:blur(6px);border-radius:8px;padding:.15rem .45rem;font-size:.6rem;color:#666;z-index:2}
.card-body{padding:1.3rem 1.4rem}
.card-title{font-family:'Cormorant Garamond',serif;font-size:1.22rem;font-weight:400;color:var(--ink);margin:0 0 .15rem}
.card-sub{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.9rem}
.score-track{height:3px;background:var(--border);border-radius:3px;margin-bottom:1.1rem;overflow:hidden}
.score-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--accent),var(--accent2))}
.items-sec{margin-bottom:1rem}
.item-row{display:flex;align-items:center;gap:.6rem;padding:.45rem 0;border-bottom:1px solid var(--border)}
.item-row:last-child{border-bottom:none}
.item-slot{font-size:.62rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted2);width:78px;flex-shrink:0}
.item-emoji{font-size:1rem}
.item-name{font-size:.82rem;font-weight:500;color:var(--ink);flex:1}
.item-fit{color:var(--muted2);font-weight:400;font-size:.7rem}
.cswatch{width:12px;height:12px;border-radius:3px;border:1.5px solid rgba(0,0,0,.08);flex-shrink:0}
.cname{font-size:.7rem;color:var(--muted)}
.xai-box{background:linear-gradient(135deg,#FBF7F2,#F5EFE6);border:1px solid #E8DDD0;border-radius:10px;padding:.9rem 1rem;margin-bottom:1rem}
.xai-lbl{font-size:.62rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--accent2);margin-bottom:.45rem}
.xai-pt{display:flex;align-items:flex-start;gap:.45rem;font-size:.78rem;color:#5C5650;margin-bottom:.25rem;line-height:1.4}
.xai-pt::before{content:"✓";color:var(--success);font-weight:700;flex-shrink:0}
.hair-row{display:flex;align-items:center;gap:.45rem;flex-wrap:wrap;margin-bottom:.9rem}
.hair-lbl{font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.hair-chip{background:#EDF2FF;color:#3D5BB5;border-radius:12px;padding:.2rem .6rem;font-size:.72rem;font-weight:500}
.shop-lbl{font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:.45rem}
.shop-row{display:flex;gap:.4rem;flex-wrap:wrap}
.shop-btn{display:inline-flex;align-items:center;gap:.3rem;padding:.3rem .65rem;border-radius:7px;font-size:.72rem;font-weight:500;text-decoration:none!important;border:1.5px solid var(--btn-color,#999);color:var(--btn-color,#999)!important;background:white;transition:all .2s ease}
.shop-btn:hover{background:var(--btn-color,#999)!important;color:white!important;transform:translateY(-1px);box-shadow:0 3px 10px rgba(0,0,0,.12)}
.feat-section-title{font-family:'Cormorant Garamond',serif;font-size:1.8rem;font-weight:300;color:var(--ink);margin:0 0 .3rem}
.feat-section-sub{font-size:.82rem;color:var(--muted);margin-bottom:2rem}
.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.1rem;margin-top:2.5rem}
.feat-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.3rem;text-align:center;cursor:pointer;transition:box-shadow var(--transition),transform var(--transition),border-color var(--transition);position:relative;overflow:hidden}
.feat-card:hover{box-shadow:var(--shadow-lg);transform:translateY(-3px);border-color:var(--accent)}
.feat-card.active{border-color:var(--accent);background:var(--accent-soft);box-shadow:var(--shadow-lg)}
.feat-icon{font-size:1.7rem;margin-bottom:.5rem}
.feat-title{font-family:'Cormorant Garamond',serif;font-size:1rem;color:var(--accent2);margin-bottom:.35rem}
.feat-desc{font-size:.78rem;color:var(--muted);line-height:1.5}
.feat-card .click-hint{font-size:.65rem;color:var(--muted2);margin-top:.5rem;letter-spacing:.05em}
@keyframes panelIn{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:translateY(0)}}
.ipanel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:2rem;margin-top:1.2rem;box-shadow:var(--shadow-lg);animation:panelIn .3s ease both}
.ipanel-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.5rem;border-bottom:1px solid var(--border);padding-bottom:1rem}
.ipanel-title{font-family:'Cormorant Garamond',serif;font-size:1.5rem;font-weight:400;color:var(--ink);margin:0 0 .2rem}
.ipanel-sub{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.intel-row{display:flex;gap:1.5rem;flex-wrap:wrap}
.intel-block{flex:1;min-width:220px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1.1rem}
.intel-block-title{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--accent2);margin-bottom:.7rem}
.intel-item{display:flex;align-items:center;gap:.5rem;margin-bottom:.4rem;font-size:.8rem;color:var(--ink)}
.intel-check{color:var(--success);font-weight:700;flex-shrink:0}
.intel-cross{color:var(--danger);font-weight:700;flex-shrink:0}
.harmony-pair{display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem}
.h-swatch{width:28px;height:28px;border-radius:6px;border:2px solid rgba(0,0,0,.08);flex-shrink:0}
.h-label{font-size:.78rem;color:var(--ink);flex:1}
.h-status{font-size:.72rem;font-weight:500}
.h-ok{color:var(--success)}
.xai-deep-card{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1.1rem;margin-bottom:1rem}
.xai-deep-label{font-size:.65rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--accent2);margin-bottom:.6rem}
.xai-line{display:flex;gap:.45rem;align-items:flex-start;font-size:.8rem;color:#5C5650;margin-bottom:.35rem;line-height:1.45}
.xai-line::before{content:"◆";color:var(--accent);font-size:.55rem;margin-top:.25rem;flex-shrink:0}
.buy-platform{border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem;text-align:center;transition:box-shadow var(--transition),transform var(--transition)}
.buy-platform:hover{box-shadow:var(--shadow-lg);transform:translateY(-2px)}
.buy-platform-icon{font-size:1.8rem;margin-bottom:.4rem}
.buy-platform-name{font-family:'Cormorant Garamond',serif;font-size:1rem;font-weight:600;margin-bottom:.3rem}
.buy-items-list{font-size:.72rem;color:var(--muted);margin-bottom:.8rem;line-height:1.7}
.buy-link{display:block;padding:.4rem .6rem;border-radius:7px;font-size:.73rem;font-weight:600;text-decoration:none!important;text-align:center;margin-bottom:.3rem;transition:all .2s;border:1.5px solid}
.buy-link:hover{transform:translateY(-1px);box-shadow:0 3px 10px rgba(0,0,0,.15)}
.hair-style-card{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1.1rem;text-align:center;margin-bottom:1rem}
.hair-style-icon{font-size:2rem;margin-bottom:.4rem}
.hair-style-name{font-family:'Cormorant Garamond',serif;font-size:1rem;color:var(--ink);margin-bottom:.3rem}
.hair-style-why{font-size:.75rem;color:var(--muted);line-height:1.5}
.hair-tip-box{background:linear-gradient(135deg,#FBF7F2,#F5EFE6);border:1px solid #E8DDD0;border-radius:10px;padding:1rem 1.2rem;margin-top:1rem;font-size:.82rem;color:#5C5650;line-height:1.6}
.ds-stat{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:.9rem 1rem;text-align:center}
.ds-stat-val{font-family:'Cormorant Garamond',serif;font-size:1.8rem;color:var(--accent);line-height:1}
.ds-stat-lbl{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:.2rem}
.ds-bar-row{display:flex;align-items:center;gap:.7rem;margin-bottom:.45rem}
.ds-bar-label{font-size:.77rem;color:var(--ink);width:110px;flex-shrink:0}
.ds-bar-track{flex:1;height:6px;background:var(--border);border-radius:4px;overflow:hidden}
.ds-bar-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent2))}
.ds-bar-count{font-size:.7rem;color:var(--muted);width:36px;text-align:right}
.upload-area{border:2px dashed var(--border2);border-radius:var(--radius);padding:2.5rem;text-align:center;background:var(--bg);transition:border-color var(--transition)}
.upload-area:hover{border-color:var(--accent)}
.upload-step{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1rem;text-align:center}
.upload-step-num{font-family:'Cormorant Garamond',serif;font-size:1.4rem;color:var(--accent)}
.upload-step-text{font-size:.78rem;color:var(--muted);margin-top:.3rem;line-height:1.4}
.empty-wrap{text-align:center;padding:4.5rem 2rem 2rem}
.empty-wrap .e-icon{font-size:3.5rem;opacity:.3;margin-bottom:1.2rem}
.empty-wrap h2{font-size:2.2rem;font-weight:300;color:var(--ink);margin-bottom:.5rem}
.empty-wrap p{color:var(--muted);font-size:.88rem;max-width:400px;margin:0 auto;line-height:1.6}
.sb-logo{padding:1.5rem 1rem .3rem;font-family:'Cormorant Garamond',serif;font-size:1.45rem;color:#E8DFC8!important;letter-spacing:.04em}
.sb-logo span{color:#B5936B!important}
.sb-tag{font-size:.62rem;color:#5C5448!important;letter-spacing:.14em;text-transform:uppercase;padding:0 1rem .8rem;display:block}
.det-badge{display:inline-flex;align-items:center;gap:.35rem;background:#EEF7EE;border:1px solid #B8DDB8;color:#3A6B3A;border-radius:20px;padding:.28rem .75rem;font-size:.75rem;font-weight:500;margin-top:.4rem}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;font-size:.78rem!important;letter-spacing:.06em;text-transform:uppercase}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--ink)!important;border-bottom:2px solid var(--accent)!important}
hr{border-color:var(--border)!important}
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:.9rem}
[data-testid="stMetricValue"]{font-family:'Cormorant Garamond',serif!important;font-size:1.9rem!important;color:var(--accent)!important}
[data-testid="stMetricLabel"]{font-size:.7rem!important;color:var(--muted)!important;text-transform:uppercase;letter-spacing:.06em}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Session state
for k,v in [("outfits",[]),("profile",None),("upload_category",None),
             ("upload_color",None),("active_feat",None),("uploaded_file_data",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource(show_spinner=False)
def load_engine():
    return XFashionEngine(dataset_path="data/fashion_dataset.csv")

@st.cache_data(show_spinner=False)
def load_dataset_stats():
    try:
        df = pd.read_csv("data/fashion_dataset.csv")
        return {
            "total": len(df),
            "categories": df["category"].value_counts().head(8).to_dict(),
            "styles": df["style"].value_counts().head(6).to_dict(),
            "colors": df["color"].value_counts().head(6).to_dict(),
            "occasions": df["occasion"].value_counts().head(6).to_dict(),
        }
    except Exception:
        return None

# ── Unsplash URL builder
def build_unsplash_url(outfit, style_val):
    style_terms = {"casual":"casual,street","elegant":"elegant,chic","formal":"formal,business",
                   "party":"party,glamour","sporty":"athletic,sport","streetwear":"streetwear,urban",
                   "minimal":"minimalist,clean","vintage":"vintage,retro"}
    terms = [style_terms.get(style_val, "fashion")]
    top_map = {"tshirt":"tshirt","blouse":"blouse","hoodie":"hoodie","sweater":"sweater",
               "shirt":"shirt","blazer":"blazer","crop_top":"crop+top","kurti":"kurti",
               "saree":"saree","lehenga":"lehenga","dress":"dress","maxi_dress":"maxi+dress"}
    if outfit.top:
        terms.append(top_map.get(outfit.top.item_name, "top"))
    if outfit.bottom:
        bmap = {"jeans":"jeans","trousers":"trousers","skirt":"skirt","wide_leg_pants":"wide+leg+pants"}
        b = bmap.get(outfit.bottom.item_name, "")
        if b: terms.append(b)
    terms.append("outfit,woman,fashion")
    query = ",".join(terms)
    seed = abs(hash(query + str(getattr(outfit, "outfit_id", 0)))) % 1000
    return f"https://source.unsplash.com/600x800/?{query}&sig={seed}"

PEXELS = {
    "casual":"https://images.pexels.com/photos/1536619/pexels-photo-1536619.jpeg?w=600&h=800&fit=crop",
    "elegant":"https://images.pexels.com/photos/1462637/pexels-photo-1462637.jpeg?w=600&h=800&fit=crop",
    "formal":"https://images.pexels.com/photos/2220316/pexels-photo-2220316.jpeg?w=600&h=800&fit=crop",
    "party":"https://images.pexels.com/photos/1375849/pexels-photo-1375849.jpeg?w=600&h=800&fit=crop",
    "sporty":"https://images.pexels.com/photos/2294342/pexels-photo-2294342.jpeg?w=600&h=800&fit=crop",
    "streetwear":"https://images.pexels.com/photos/1040945/pexels-photo-1040945.jpeg?w=600&h=800&fit=crop",
    "minimal":"https://images.pexels.com/photos/2220329/pexels-photo-2220329.jpeg?w=600&h=800&fit=crop",
    "vintage":"https://images.pexels.com/photos/1580271/pexels-photo-1580271.jpeg?w=600&h=800&fit=crop",
    "ethnic":"https://images.pexels.com/photos/2916814/pexels-photo-2916814.jpeg?w=600&h=800&fit=crop",
}

HAIR_DETAILS = {
    "Loose waves":("🌊","Effortless & romantic — softens structured outfits","Any face shape"),
    "High bun":("🎀","Polished & clean — perfect for office/formal vibes","Oval, square"),
    "Messy ponytail":("🐴","Relaxed & playful — great for casual street looks","All face shapes"),
    "Hollywood waves":("✨","Glamorous & timeless — elevates evening dresses","Oval, heart"),
    "Chignon updo":("🌸","Elegant & refined — the ultimate formal statement","All face shapes"),
    "Sleek low ponytail":("🖤","Minimal & chic — pairs with clean, modern outfits","Long faces"),
    "Structured blowout":("💼","Polished volume — boardroom-ready confidence","All face shapes"),
    "French twist":("🌟","Sophisticated classic — works for formal events","Oval, round"),
    "Low bun":("⚪","Understated elegance — minimal & effortless","Any face shape"),
    "Voluminous curls":("🌀","Bold & playful — brings energy to any look","Heart, oval"),
    "High ponytail":("🦄","Sleek & dynamic — sporty-chic energy","All face shapes"),
    "Textured updo":("💃","Artsy & expressive — makes a statement","Oval, square"),
    "French braid":("🌿","Athletic meets feminine — versatile & practical","All face shapes"),
    "Space buns":("✌️","Y2K-inspired fun — youthful streetwear vibe","Round, oval"),
    "Victory rolls":("🎞️","Vintage drama — iconic 40s/50s glamour","All face shapes"),
    "Soft finger waves":("🌙","Old Hollywood — pairs with vintage & ethnic looks","Oval, heart"),
    "Sleek straight":("🤍","Ultra-minimal — the quiet luxury signature","All face shapes"),
    "Effortless curtain bangs":("✂️","Flattering frame — softens strong jawlines","Square, diamond"),
    "Effortless waves":("🌊","Lived-in texture — works with any casual look","All face shapes"),
    "Classic blowout":("💫","Polished volume — timeless & universally flattering","All face shapes"),
    "Sleek bun":("⭕","Clean & professional — pairs with formal/minimal","Long, oval"),
    "Side-swept blowout":("🌬️","Softens round faces — elegant & approachable","Round, square"),
    "Half-up half-down":("🎗️","Versatile & feminine — bridges casual to semi-formal","All face shapes"),
    "Slick bun":("◾","Power dressing accessory — sleek & intentional","All face shapes"),
    "Bold afro-inspired":("✦","Celebrates natural texture — full & confident","All face shapes"),
    "Colourful braids":("🌈","Creative self-expression — street & festival vibes","All face shapes"),
    "Top knot":("🔵","Quick & chic — the everyday minimalist solution","Heart, oval"),
    "Pin curls":("📍","Delicate vintage look — great with ethnic wear","Oval, heart"),
}

# ── Sidebar
with st.sidebar:
    st.markdown('<div class="sb-logo">✦ X<span>Fashion</span></div><span class="sb-tag">AI Personal Stylist v3</span>', unsafe_allow_html=True)
    st.markdown("---")
    weather     = st.selectbox("Season / Weather",   ["summer","winter","spring","autumn"])
    occasion    = st.selectbox("Occasion",           ["daily","office","party","date","travel","gym","formal_event","wedding","festival","casual"])
    outfit_type = st.selectbox("Outfit Style",       ["western","ethnic","indo_western","college","formal","party"])
    st.markdown("---")
    body_type   = st.selectbox("Body Type",          ["pear","apple","hourglass","rectangle","petite","tall","plus"])
    skin_tone   = st.selectbox("Skin Tone",          ["warm","cool","neutral","deep","fair","olive"])
    st.markdown("---")
    style       = st.selectbox("Style Vibe",         ["casual","streetwear","minimal","formal","sporty","party","vintage","elegant"])
    comfort     = st.selectbox("Comfort Preference", ["loose","regular","tight"], index=1)
    color_pref  = st.selectbox("Color Palette",      ["any","dark","light","neutral","bright"])
    n_outfits   = st.slider("Number of Outfits", 3, 8, 5)
    st.markdown("---")
    st.markdown('<div style="color:#5C5448;font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.4rem">📸 Upload Your Item</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop a clothing photo", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    if uploaded_file:
        cat   = detect_category_from_filename(uploaded_file.name)
        color = detect_dominant_color_from_name(uploaded_file.name)
        st.session_state.upload_category = cat
        st.session_state.upload_color    = color
        data = uploaded_file.read()
        st.session_state.uploaded_file_data = data
        st.markdown(f'<div class="det-badge">✓ Detected: <b>{cat.replace("_"," ").title()}</b></div>', unsafe_allow_html=True)
        st.image(data, use_column_width=True)
    st.markdown("---")
    generate_btn = st.button("✦ Generate My Outfits", use_container_width=True)
    st.markdown('<div style="padding:1rem 0 0;font-size:.67rem;color:#3E3830;line-height:2.1">🧠 Rule-Based Intelligence<br>📊 ML Scoring Pipeline<br>🔍 Explainable AI (XAI)<br>🗂️ 50K+ Item Dataset<br>🕸️ Graph-Ready Architecture</div>', unsafe_allow_html=True)

# ── Hero
st.markdown('<div class="xf-hero"><div class="xf-wordmark">Explainable AI · Personal Stylist</div><h1>Dress <em>smarter</em>,<br>not just better.</h1><p>AI-curated outfits with full reasoning · Western &amp; Ethnic · 50K+ items · Buy The Look</p></div>', unsafe_allow_html=True)

def safe_color(item):
    return item.color if item else "default"

def render_card(outfit, rank, vibe, profile):
    score = outfit.score
    grade = score_to_grade(score)
    vibe_text, vibe_emoji = vibe
    gc = {"S":"#B8860B","A":"#8B6F47","B":"#5B8A6F","C":"#4A6BA3","D":"#C0574A"}.get(grade,"#888")
    insp_url   = build_unsplash_url(outfit, profile.style)
    pexels_url = PEXELS.get("ethnic" if outfit.outfit_type=="ethnic" else profile.style, PEXELS["casual"])
    svg = generate_outfit_svg(safe_color(outfit.top),safe_color(outfit.bottom),safe_color(outfit.shoes),safe_color(outfit.outerwear),outfit.outfit_type,outfit.style_label.lower())
    svg_b64 = base64.b64encode(svg.encode()).decode()
    html = f"""<div class="outfit-card">
  <div class="card-illus">
    <img class="insp-img" src="{insp_url}" onerror="this.src='{pexels_url}';this.onerror=function(){{this.style.display='none';this.nextElementSibling.style.display='block';}}" alt="outfit inspiration" loading="lazy"/>
    <img class="svg-fallback" src="data:image/svg+xml;base64,{svg_b64}" alt="outfit" style="display:none"/>
    <div class="img-overlay"></div>
    <div class="grade-pill" style="background:{gc}">{grade}</div>
    <div class="vibe-tag">{vibe_emoji} {vibe_text}</div>
    <div class="rank-tag">#{rank}</div>
    <div class="insp-src">📸 Unsplash</div>
  </div>
  <div class="card-body">
    <div class="card-title">{outfit.style_label} Look</div>
    <div class="card-sub">{outfit.outfit_type.replace("_"," ").title()} &middot; {score}/100</div>
    <div class="score-track"><div class="score-fill" style="width:{score}%"></div></div>
    <div class="items-sec">"""
    for slot_name, item in [("Top",outfit.top),("Bottom",outfit.bottom),("Outerwear",outfit.outerwear),("Shoes",outfit.shoes),("Accessories",outfit.accessories)]:
        if item is None:
            html += f'<div class="item-row"><div class="item-slot">{slot_name}</div><span style="color:#ccc;font-size:.73rem">—</span></div>'
        else:
            emoji=get_item_emoji(item.item_name); name=format_item_name(item.item_name)
            chex=get_color_hex(item.color); cname=item.color.replace("_"," ").title(); fit=item.fit.replace("_"," ")
            html += f'<div class="item-row"><div class="item-slot">{slot_name}</div><div class="item-emoji">{emoji}</div><div class="item-name">{name} <span class="item-fit">&middot; {fit}</span></div><div class="cswatch" style="background:{chex}" title="{cname}"></div><div class="cname">{cname}</div></div>'
    html += "</div>"
    if outfit.explanations:
        html += '<div class="xai-box"><div class="xai-lbl">🔍 Why this works</div>'
        for exp in outfit.explanations[:5]:
            clean=exp.split("—")[-1].strip() if "—" in exp else exp
            clean=(clean[:88]+"…") if len(clean)>88 else clean
            html += f'<div class="xai-pt">{clean}</div>'
        html += "</div>"
    hairs=get_hairstyle_suggestions(profile.style,profile.body_type)
    html += '<div class="hair-row"><div class="hair-lbl">💇 Hair vibes</div>'
    for h in hairs[:3]: html += f'<div class="hair-chip">{h}</div>'
    html += "</div>"
    primary=outfit.top or outfit.bottom
    if primary:
        html += f'<div class="shop-lbl">🛍️ Buy The Look</div>'
        html += get_shop_links(primary.item_name,primary.color)
    html += "</div></div>"
    return html

# ── Generate
if generate_btn:
    profile = UserProfile(weather=weather,occasion=occasion,outfit_type=outfit_type,body_type=body_type,skin_tone=skin_tone,style=style,comfort=comfort,color_preference=color_pref)
    with st.spinner("✦ Curating your personalised outfits…"):
        try:
            engine = load_engine()
            outfits = engine.recommend(profile, n_outfits=n_outfits)
            st.session_state.outfits = outfits
            st.session_state.profile = profile
            st.session_state.active_feat = None
        except Exception as e:
            st.error(f"Engine error: {e}"); st.exception(e)

# ── Feature card renderer (shared)
def render_feature_cards(prefix, has_outfits):
    FEATURES = [
        ("fashion_intel","🧠","Fashion Intelligence","Colour harmony, body rules, season logic"),
        ("xai","🔍","Explainable AI","Full reasoning behind your top outfit"),
        ("buy","🛍️","Buy The Look","Shop all items across 4 platforms"),
        ("upload","📸","Upload Your Item","Build an outfit around a piece you own"),
        ("hair","💇","Hairstyle Pairing","Curated hair looks for your profile"),
        ("dataset","🗂️","50K+ Items","Live dataset stats and category breakdown"),
    ]
    fc = st.columns(3)
    for i,(fid,icon,title,desc) in enumerate(FEATURES):
        col = fc[i%3]
        with col:
            is_active = st.session_state.active_feat == fid
            ac = "active" if is_active else ""
            st.markdown(f'<div class="feat-card {ac}"><div class="feat-icon">{icon}</div><div class="feat-title">{title}</div><div class="feat-desc">{desc}</div><div class="click-hint">{"▲ collapse" if is_active else "▼ explore"}</div></div>', unsafe_allow_html=True)
            if st.button(f'{"Hide" if is_active else "Open"} {title}', key=f"{prefix}_{fid}"):
                st.session_state.active_feat = None if is_active else fid
                st.rerun()

# ── Panel rendering (used in both states)
def render_active_panel(af, profile, outfits):
    if af is None: return
    first_outfit = outfits[0] if outfits else None

    if af == "fashion_intel":
        st.markdown('<div class="ipanel">', unsafe_allow_html=True)
        sub = f"personalised for {profile.body_type} · {profile.skin_tone} · {profile.weather}" if profile else "preview of rules"
        st.markdown(f'<div class="ipanel-header"><div><div class="ipanel-title">🧠 Fashion Intelligence Engine</div><div class="ipanel-sub">{sub}</div></div></div>', unsafe_allow_html=True)
        ic1,ic2,ic3 = st.columns(3)
        tone_palettes = {"warm":["beige","camel","rust","mustard","terracotta","gold"],"cool":["navy","cobalt","lavender","mint","steel_grey","icy_blue"],"neutral":["white","black","grey","nude","cream"],"deep":["burgundy","maroon","deep_purple","emerald","cobalt"],"fair":["lavender","dusty_rose","icy_blue","cream","mint"],"olive":["mustard","terracotta","camel","rust","teal"]}
        with ic1:
            tone_key = profile.skin_tone if profile else "neutral"
            recommended = tone_palettes.get(tone_key, ["beige","white","black"])
            st.markdown('<div class="intel-block"><div class="intel-block-title">🎨 Colour Harmony</div>', unsafe_allow_html=True)
            for c in recommended[:5]:
                chex=get_color_hex(c)
                st.markdown(f'<div class="harmony-pair"><div class="h-swatch" style="background:{chex}"></div><div class="h-label">{c.replace("_"," ").title()}</div><div class="h-status h-ok">✓ Suits {tone_key}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        body_rules_text = {
            "pear":["Wide-leg bottoms recommended","Volume on top balances hips","Structured outerwear preferred","Avoid skinny-fit bottoms","A-line silhouettes work well"],
            "apple":["Flowy tops draw attention up","Structured blazers define waist","Avoid fitted cropped tops","Empire-waist works great","Dark bottoms slim silhouette"],
            "hourglass":["Fitted silhouettes throughout","Wrap-style tops accentuate","Belted outerwear encouraged","Both slim & wide fits work","Balanced proportions key"],
            "rectangle":["Layering creates dimension","Structured pieces add shape","Wide-leg bottoms add curve","Peplum tops work well","Avoid uniform-width outfits"],
            "petite":["Avoid oversized proportions","Monochrome elongates frame","High-waist bottoms lengthen","Cropped jackets flatter","Avoid ankle-strap shoes"],
            "tall":["Long coats look stunning","Wide-leg & palazzo ideal","Oversized silhouettes work","Bold patterns translate well","Midi & maxi lengths flattering"],
            "plus":["Relaxed fits over tight","Vertical lines elongate","Wrap silhouettes work great","Avoid overly stiff fabrics","Monochrome is slimming"],
        }
        with ic2:
            bt = profile.body_type if profile else "hourglass"
            rules = body_rules_text.get(bt,[])
            st.markdown(f'<div class="intel-block"><div class="intel-block-title">🧍 Body Type · {bt.title()}</div>', unsafe_allow_html=True)
            for r in rules[:5]: st.markdown(f'<div class="intel-item"><span class="intel-check">✓</span>{r}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        season_rules = {
            "summer":{"✓ Allowed":["Cotton","Linen","Chiffon","Georgette"],"✗ Filtered":["Wool coats","Heavy boots","Turtlenecks","Fleece"]},
            "winter":{"✓ Allowed":["Wool","Fleece","Knit","Velvet"],"✗ Filtered":["Sheer fabrics","Sandals","Sleeveless tops"]},
            "spring":{"✓ Allowed":["Light cotton","Chiffon","Jersey"],"✗ Filtered":["Heavy wool","Puffer jackets"]},
            "autumn":{"✓ Allowed":["Cotton","Corduroy","Light wool"],"✗ Filtered":["Summer-only fabrics","Sandals"]},
        }
        with ic3:
            sw = profile.weather if profile else "summer"
            srules = season_rules.get(sw,{})
            st.markdown(f'<div class="intel-block"><div class="intel-block-title">🌤 Season · {sw.title()}</div>', unsafe_allow_html=True)
            for cat, items in srules.items():
                icon="✓" if "✓" in cat else "✗"; cls="intel-check" if "✓" in cat else "intel-cross"
                for itm in items[:3]: st.markdown(f'<div class="intel-item"><span class="{cls}">{icon}</span>{itm}</div>', unsafe_allow_html=True)
            occ = profile.occasion.replace("_"," ").title() if profile else "—"
            st.markdown(f'<div style="margin-top:.6rem;padding-top:.6rem;border-top:1px solid var(--border);font-size:.75rem;color:var(--muted)">Occasion: <b style="color:var(--ink)">{occ}</b> · filter active</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif af == "xai":
        if not first_outfit:
            st.markdown('<div class="ipanel"><div style="text-align:center;padding:2rem;color:var(--muted)">✦ Generate outfits first to see AI reasoning.</div></div>', unsafe_allow_html=True)
            return
        st.markdown('<div class="ipanel">', unsafe_allow_html=True)
        st.markdown(f'<div class="ipanel-header"><div><div class="ipanel-title">🔍 Explainable AI — Top Outfit Reasoning</div><div class="ipanel-sub">Full decision trail · Outfit #1 · Score {first_outfit.score}/100</div></div></div>', unsafe_allow_html=True)
        xc1,xc2 = st.columns(2)
        slots=[("Top",first_outfit.top),("Bottom",first_outfit.bottom),("Outerwear",first_outfit.outerwear),("Shoes",first_outfit.shoes),("Accessories",first_outfit.accessories)]
        for i,(slot_name,item) in enumerate(slots):
            col=xc1 if i%2==0 else xc2
            with col:
                if item:
                    chex=get_color_hex(item.color); emoji=get_item_emoji(item.item_name)
                    st.markdown(f'<div class="xai-deep-card"><div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.7rem"><div style="width:18px;height:18px;border-radius:4px;background:{chex};flex-shrink:0;border:1px solid rgba(0,0,0,.1)"></div><div class="xai-deep-label" style="margin:0">{slot_name}: {emoji} {format_item_name(item.item_name)}</div></div>', unsafe_allow_html=True)
                    p_st = profile.skin_tone if profile else "warm"
                    p_bt = profile.body_type if profile else "hourglass"
                    p_wt = profile.weather if profile else "summer"
                    p_sty = profile.style if profile else "casual"
                    p_occ = profile.occasion.replace("_"," ") if profile else "daily"
                    for r in [f"{item.color.replace('_',' ').title()} suits your {p_st} skin tone",f"{item.fit.replace('_',' ').title()} fit for {p_bt} body type",f"{item.material.title()} fabric ideal for {p_wt} weather",f"{item.style.title()} aligns with {p_sty} style",f"Chosen for {p_occ} occasion"]:
                        st.markdown(f'<div class="xai-line">{r}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        if first_outfit.explanations:
            st.markdown('<div style="margin-top:1rem"><div style="font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--accent2);margin-bottom:.6rem">Full Outfit Narrative</div>', unsafe_allow_html=True)
            for exp in first_outfit.explanations:
                st.markdown(f'<div class="xai-line" style="font-size:.82rem;margin-bottom:.4rem">{exp}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif af == "buy":
        if not first_outfit:
            st.markdown('<div class="ipanel"><div style="text-align:center;padding:2rem;color:var(--muted)">✦ Generate outfits first to see Buy The Look links.</div></div>', unsafe_allow_html=True)
            return
        st.markdown('<div class="ipanel"><div class="ipanel-header"><div><div class="ipanel-title">🛍️ Buy The Look</div><div class="ipanel-sub">Shop every item from your top outfit across 4 platforms</div></div></div>', unsafe_allow_html=True)
        items_to_buy=[(s,i) for s,i in [("Top",first_outfit.top),("Bottom",first_outfit.bottom),("Outerwear",first_outfit.outerwear),("Shoes",first_outfit.shoes),("Accessories",first_outfit.accessories)] if i]
        platform_configs=[
            {"name":"Myntra","icon":"🛍️","color":"#FF3F6C","url":"https://www.myntra.com/{q}"},
            {"name":"Amazon","icon":"📦","color":"#FF9900","url":"https://www.amazon.in/s?k={q}"},
            {"name":"Nykaa","icon":"💄","color":"#FC2779","url":"https://www.nykaafashion.com/search?q={q}"},
            {"name":"Savana","icon":"🌿","color":"#3A6B4C","url":"https://savana.in/search?q={q}"},
        ]
        bc1,bc2,bc3,bc4=st.columns(4)
        for col,plat in zip([bc1,bc2,bc3,bc4],platform_configs):
            with col:
                items_html="".join([f'<div>{get_item_emoji(i.item_name)} {format_item_name(i.item_name)}</div>' for s,i in items_to_buy])
                links_html="".join([f'<a href="{plat["url"].format(q=i.color.replace("_","+")+"+"+i.item_name.replace("_","+"))}" target="_blank" class="buy-link" style="border-color:{plat["color"]};color:{plat["color"]}">{s}: {format_item_name(i.item_name)}</a>' for s,i in items_to_buy[:4]])
                st.markdown(f'<div class="buy-platform"><div class="buy-platform-icon">{plat["icon"]}</div><div class="buy-platform-name" style="color:{plat["color"]}">{plat["name"]}</div><div class="buy-items-list">{items_html}</div>{links_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif af == "upload":
        st.markdown('<div class="ipanel"><div class="ipanel-header"><div><div class="ipanel-title">📸 Upload Your Item</div><div class="ipanel-sub">Upload a piece you own — we\'ll build a complete outfit around it</div></div></div>', unsafe_allow_html=True)
        if st.session_state.uploaded_file_data:
            uc1,uc2=st.columns([1,2])
            with uc1:
                st.image(st.session_state.uploaded_file_data, caption="Your uploaded item", use_column_width=True)
                cat=st.session_state.upload_category or "top"
                st.markdown(f'<div class="det-badge" style="margin-top:.5rem">✓ Category detected: <b>{cat.replace("_"," ").title()}</b></div>', unsafe_allow_html=True)
            with uc2:
                st.markdown(f'<div style="padding:1rem 0"><div style="font-family:\'Cormorant Garamond\',serif;font-size:1.2rem;margin-bottom:1rem">How it works</div><div class="xai-line">Your item is identified as a <b>{cat.replace("_"," ").title()}</b></div><div class="xai-line">The AI fixes this slot and builds the remaining outfit around it</div><div class="xai-line">Colour harmony, body type, and occasion rules still apply</div><div class="xai-line">The generated outfit complements your uploaded piece</div><div style="margin-top:1.2rem;font-size:.8rem;color:var(--muted)">💡 Tip: Name your file with the item type<br>e.g. <code>navy_blazer.jpg</code> or <code>floral_skirt.png</code></div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="upload-area"><div style="font-size:2.8rem;opacity:.5;margin-bottom:.8rem">📸</div><div style="font-family:\'Cormorant Garamond\',serif;font-size:1.3rem;color:var(--ink);margin-bottom:.4rem">Upload a Clothing Item</div><div style="font-size:.82rem;color:var(--muted)">Use the file uploader in the left sidebar to add your item</div></div>', unsafe_allow_html=True)
            ust1,ust2,ust3=st.columns(3)
            for col,(num,txt) in zip([ust1,ust2,ust3],[("1","Upload photo from sidebar (JPG, PNG, WebP)"),("2","AI detects category from filename automatically"),("3","Click Generate — your item anchors the look")]):
                with col: st.markdown(f'<div class="upload-step"><div class="upload-step-num">{num}</div><div class="upload-step-text">{txt}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif af == "hair":
        st.markdown('<div class="ipanel">', unsafe_allow_html=True)
        p_style = profile.style if profile else "casual"
        p_bt    = profile.body_type if profile else "hourglass"
        p_occ   = profile.occasion.replace("_"," ").title() if profile else "Daily"
        st.markdown(f'<div class="ipanel-header"><div><div class="ipanel-title">💇 Hairstyle Pairing</div><div class="ipanel-sub">Curated for {p_style.title()} · {p_bt.title()} · {p_occ}</div></div></div>', unsafe_allow_html=True)
        hairs = get_hairstyle_suggestions(p_style, p_bt)
        extra = ["Loose waves","Hollywood waves","French braid","High ponytail","Sleek bun"]
        all_hairs = list(dict.fromkeys(hairs+extra))[:6]
        hc1,hc2,hc3 = st.columns(3)
        for i,hair in enumerate(all_hairs):
            col=[hc1,hc2,hc3][i%3]
            d=HAIR_DETAILS.get(hair,("💇","Versatile and flattering style","All face shapes"))
            is_primary=hair in hairs
            border="border-color:var(--accent);background:var(--accent-soft)" if is_primary else ""
            with col:
                badge='&nbsp;<span style="font-size:.6rem;background:var(--accent);color:white;border-radius:10px;padding:.1rem .4rem;vertical-align:middle">✦ TOP PICK</span>' if is_primary else ""
                st.markdown(f'<div class="hair-style-card" style="{border}"><div class="hair-style-icon">{d[0]}</div><div class="hair-style-name">{hair}{badge}</div><div class="hair-style-why">{d[1]}</div><div style="font-size:.68rem;color:var(--muted2);margin-top:.4rem">Face shape: {d[2]}</div></div>', unsafe_allow_html=True)
        vibe_t,vibe_e=get_outfit_vibe(p_style, profile.weather if profile else "summer")
        st.markdown(f'<div class="hair-tip-box">💡 <b>Stylist Tip</b> for your <em>{vibe_t}</em> vibe: Your top picks ({", ".join(hairs[:2])}) were selected for <b>{p_style}</b> style and <b>{p_bt}</b> body type — creating the best visual balance with your outfit silhouettes.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif af == "dataset":
        stats = load_dataset_stats()
        st.markdown('<div class="ipanel"><div class="ipanel-header"><div><div class="ipanel-title">🗂️ Dataset Intelligence</div><div class="ipanel-sub">Live stats from the X-Fashion 50K item catalogue</div></div></div>', unsafe_allow_html=True)
        if stats:
            ds1,ds2,ds3,ds4=st.columns(4)
            for col,(val,lbl) in zip([ds1,ds2,ds3,ds4],[(str(stats["total"]//1000)+"K+","Total Items"),(str(len(stats["categories"])),"Categories"),(str(len(stats["styles"])),"Style Types"),(str(len(stats["occasions"])),"Occasions")]):
                with col: st.markdown(f'<div class="ds-stat"><div class="ds-stat-val">{val}</div><div class="ds-stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            da1,da2=st.columns(2)
            with da1:
                for title, data_dict in [("Top Categories",stats["categories"]),("Style Distribution",stats["styles"])]:
                    st.markdown(f'<div style="font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--accent2);margin:1rem 0 .8rem 0">{title}</div>', unsafe_allow_html=True)
                    mx=max(data_dict.values()) if data_dict else 1
                    for k,v in data_dict.items():
                        st.markdown(f'<div class="ds-bar-row"><div class="ds-bar-label">{k.replace("_"," ").title()[:14]}</div><div class="ds-bar-track"><div class="ds-bar-fill" style="width:{int(v/mx*100)}%"></div></div><div class="ds-bar-count">{v:,}</div></div>', unsafe_allow_html=True)
            with da2:
                for title, data_dict in [("Top Colors",stats["colors"]),("Occasion Split",stats["occasions"])]:
                    st.markdown(f'<div style="font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--accent2);margin:1rem 0 .8rem 0">{title}</div>', unsafe_allow_html=True)
                    mx=max(data_dict.values()) if data_dict else 1
                    for k,v in data_dict.items():
                        chex=get_color_hex(k) if title=="Top Colors" else "var(--accent)"
                        prefix_html=f'<div style="width:11px;height:11px;border-radius:3px;background:{chex};flex-shrink:0;border:1px solid rgba(0,0,0,.08)"></div>' if title=="Top Colors" else ""
                        st.markdown(f'<div class="ds-bar-row">{prefix_html}<div class="ds-bar-label">{k.replace("_"," ").title()[:13]}</div><div class="ds-bar-track"><div class="ds-bar-fill" style="width:{int(v/mx*100)}%;{"background:"+chex if title=="Top Colors" else ""}"></div></div><div class="ds-bar-count">{v:,}</div></div>', unsafe_allow_html=True)
        else:
            st.info("Run `python generate_dataset.py` first to generate dataset.", icon="ℹ️")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.outfits:
    profile=st.session_state.profile; outfits=st.session_state.outfits
    avg_score=sum(o.score for o in outfits)/len(outfits)
    best=max(outfits,key=lambda o:o.score)
    icons=["🌤","📅","👗","🧍","🎨","💫","🤗"]
    vals=[profile.weather,profile.occasion,profile.outfit_type,profile.body_type,profile.skin_tone,profile.style,profile.comfort]
    tags="".join([f'<span class="ptag">{i} {v.replace("_"," ").title()}</span>' for i,v in zip(icons,vals)])
    st.markdown(f'<div class="profile-strip"><span class="plabel">Profile</span>{tags}</div>', unsafe_allow_html=True)
    vibe_text,vibe_emoji=get_outfit_vibe(profile.style,profile.weather)
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Outfits",len(outfits))
    with c2: st.metric("Avg Score",f"{avg_score:.0f}/100")
    with c3: st.metric("Top Score",f"{best.score:.0f}/100")
    with c4: st.metric("Vibe",f"{vibe_emoji} {vibe_text[:16]}…" if len(vibe_text)>16 else f"{vibe_emoji} {vibe_text}")
    st.markdown("---")
    tab1,tab2,tab3=st.tabs(["✦ Gallery","🏆 Ranked","📊 Insights"])
    with tab1:
        pairs=[outfits[i:i+2] for i in range(0,len(outfits),2)]
        for pair in pairs:
            cols=st.columns(len(pair))
            for col,outfit in zip(cols,pair):
                rank=outfits.index(outfit)+1
                with col: st.markdown(render_card(outfit,rank,get_outfit_vibe(profile.style,profile.weather),profile),unsafe_allow_html=True)
    with tab2:
        medals=["🥇","🥈","🥉"]+[f"#{i}" for i in range(4,20)]
        for rank,outfit in enumerate(sorted(outfits,key=lambda o:-o.score),1):
            cols=st.columns([0.06,0.94])
            with cols[0]: st.markdown(f"<div style='font-size:1.5rem;text-align:center;margin-top:1.4rem'>{medals[rank-1]}</div>",unsafe_allow_html=True)
            with cols[1]: st.markdown(render_card(outfit,rank,get_outfit_vibe(profile.style,profile.weather),profile),unsafe_allow_html=True)
    with tab3:
        st.markdown("### 📊 Style Intelligence Report")
        col_a,col_b=st.columns(2)
        with col_a:
            st.markdown("**Color Distribution**")
            color_counts={}
            for o in outfits:
                for item in [o.top,o.bottom,o.outerwear,o.shoes,o.accessories]:
                    if item:
                        c=item.color.replace("_"," ").title()
                        color_counts[c]=color_counts.get(c,0)+1
            for color,cnt in sorted(color_counts.items(),key=lambda x:-x[1])[:8]:
                chex=get_color_hex(color.lower().replace(" ","_"))
                st.markdown(f'<div style="display:flex;align-items:center;gap:.6rem;margin:.3rem 0"><div style="width:11px;height:11px;border-radius:3px;background:{chex};border:1px solid #ddd;flex-shrink:0"></div><span style="font-size:.8rem;flex:1">{color}</span><div style="width:80px;height:5px;background:#EEE;border-radius:3px"><div style="height:5px;width:{cnt*20}%;background:{chex};border-radius:3px;max-width:100%"></div></div><span style="font-size:.7rem;color:#888;width:16px;text-align:right">{cnt}</span></div>', unsafe_allow_html=True)
        with col_b:
            st.markdown("**Outfit Score Breakdown**")
            grade_colors={"S":"#B8860B","A":"#8B6F47","B":"#5B8A6F","C":"#4A6BA3","D":"#C0574A"}
            for o in sorted(outfits,key=lambda x:-x.score):
                g=score_to_grade(o.score); gc=grade_colors.get(g,"#888")
                st.markdown(f'<div style="display:flex;align-items:center;gap:.6rem;margin:.4rem 0"><span style="font-size:.73rem;font-weight:700;color:{gc};width:18px">{g}</span><div style="flex:1;height:7px;background:#EEE;border-radius:4px"><div style="height:7px;width:{o.score}%;background:{gc};border-radius:4px;opacity:.8"></div></div><span style="font-size:.72rem;color:#888;width:48px;text-align:right">{o.score:.0f}/100</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="margin-bottom:1.5rem"><div class="feat-section-title">Explore the Intelligence</div><div class="feat-section-sub">Click any feature to see real data from your current recommendations</div></div>', unsafe_allow_html=True)
    render_feature_cards("with_outfits", has_outfits=True)
    render_active_panel(st.session_state.active_feat, profile, outfits)

else:
    st.markdown('<div class="empty-wrap"><div class="e-icon">✦</div><h2>Your Style, Explained</h2><p>Set your preferences in the sidebar and let our AI curate personalised outfit looks — with full explainability, colour harmony, and one-click shopping.</p></div>', unsafe_allow_html=True)
    st.markdown('<div style="margin-top:2rem;margin-bottom:1rem"><div class="feat-section-title">Explore Features</div><div class="feat-section-sub">Click any card — some panels work without generating outfits first</div></div>', unsafe_allow_html=True)
    render_feature_cards("empty", has_outfits=False)
    render_active_panel(st.session_state.active_feat, None, [])