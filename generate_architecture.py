"""
Generates docs/architecture.png — a visual overview of the
NewscastAI pipeline architecture.

Run from the project root:
    python generate_architecture.py

Output: docs/architecture.png
Requires: matplotlib (pip install matplotlib)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Canvas setup
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(18, 22))
ax.set_xlim(0, 18)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

C = {
    "bg":         "#0f1117",
    "panel":      "#1a1d27",
    "border":     "#2e3250",
    "crewai":     "#4f6ef7",   # blue  — CrewAI crew
    "langgraph":  "#7c3aed",   # purple — LangGraph
    "anthropic":  "#e05c5c",   # red   — Anthropic API
    "infra":      "#2d6a4f",   # green — infrastructure
    "neutral":    "#374151",   # grey  — utilities
    "text_main":  "#f0f4ff",
    "text_sub":   "#9ca3af",
    "text_label": "#d1d5db",
    "arrow":      "#6b7280",
    "arrow_hi":   "#a5b4fc",
}

def panel(x, y, w, h, color, alpha=0.15, radius=0.3):
    """Draw a rounded rectangle panel."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(rect)
    # border only
    border = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
        alpha=0.7,
        zorder=3,
    )
    ax.add_patch(border)

def box(x, y, w, h, color, label, sublabel=None, icon=None):
    """Draw a node box with label."""
    panel(x, y, w, h, color, alpha=0.2)
    cx = x + w / 2
    cy = y + h / 2
    if icon:
        ax.text(cx, cy + 0.18, icon, ha="center", va="center",
                fontsize=14, color=color, zorder=5)
        label_y = cy - 0.08
    else:
        label_y = cy + (0.12 if sublabel else 0)
    ax.text(cx, label_y, label,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold",
            color=C["text_main"], zorder=5)
    if sublabel:
        ax.text(cx, cy - 0.22, sublabel,
                ha="center", va="center",
                fontsize=7.5, color=C["text_sub"], zorder=5)

def arrow(x1, y1, x2, y2, color=None, label=None):
    """Draw a directional arrow between two points."""
    color = color or C["arrow_hi"]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->,head_width=0.25,head_length=0.18",
            color=color, lw=1.8,
        ),
        zorder=4,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my, label,
                ha="left", va="center",
                fontsize=7, color=C["text_sub"], zorder=5)

def section_title(x, y, text, color):
    """Draw a section label."""
    ax.text(x, y, text, ha="left", va="center",
            fontsize=8, fontweight="bold",
            color=color, alpha=0.85, zorder=5)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

ax.text(9, 21.3, "NewscastAI — System Architecture",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color=C["text_main"])
ax.text(9, 20.9, "Multi-agent retrieval  ·  LangGraph script pipeline  ·  Anthropic tool-use critique",
        ha="center", va="center",
        fontsize=9, color=C["text_sub"])

# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------

box(6.5, 20.0, 5, 0.65, C["neutral"],
    "User Topics", "e.g.  AI · Finance · Canada · Health")

# ---------------------------------------------------------------------------
# CREWAI CREW  (y: 15.8 – 19.6)
# ---------------------------------------------------------------------------

panel(1.2, 15.6, 15.6, 4.1, C["crewai"], alpha=0.07)
section_title(1.4, 19.5, "① CrewAI Retrieval Crew  —  Sequential Process", C["crewai"])

# four agent boxes side by side
agents = [
    ("QueryGenerator\nAgent",   "topics → keyword\nfacets",     "🔍"),
    ("Retriever\nAgent",        "FeedFetcherTool\n22 RSS feeds", "📡"),
    ("Ranker\nAgent",           "Credibility +\nRecency tools",  "📊"),
    ("Editorial\nAgent",        "topic selection\n+ slate",      "🗞"),
]
ax_starts = [1.5, 5.3, 9.1, 12.9]
for (lbl, sub, icon), ax_x in zip(agents, ax_starts):
    box(ax_x, 16.7, 3.4, 2.5, C["crewai"], lbl, sub, icon)

# arrows between agents
for i in range(3):
    arrow(ax_starts[i] + 3.4, 17.95, ax_starts[i+1], 17.95, C["crewai"])

# fallback window note
ax.text(9, 16.25,
        "Fallback window ladder:  7 days  →  30 days  →  1 year  →  no_news_today",
        ha="center", va="center",
        fontsize=8, color=C["text_sub"],
        style="italic")

# tool badges
tools = [
    (6.5,  16.25, "FeedFetcherTool"),
    (9.5,  16.25, "CredibilityCheckerTool"),
    (11.9, 16.25, "RecencyScorerTool"),
]
for tx, ty, tname in tools:
    ax.text(tx, ty, f"  {tname}  ",
            ha="center", va="center", fontsize=6.5,
            color=C["crewai"],
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=C["crewai"], alpha=0.15,
                      edgecolor=C["crewai"]))

# ---------------------------------------------------------------------------
# ARROW: crew → langgraph
# ---------------------------------------------------------------------------

arrow(9, 15.6, 9, 15.0, C["arrow_hi"],
      "{chosen_topic, items[]}")

# ---------------------------------------------------------------------------
# LANGGRAPH PIPELINE  (y: 11.0 – 15.0)
# ---------------------------------------------------------------------------

panel(1.2, 10.8, 15.6, 4.0, C["langgraph"], alpha=0.07)
section_title(1.4, 14.65, "② LangGraph Script Pipeline  —  StateGraph", C["langgraph"])

# nodes
lg_nodes = [
    (1.6,  11.8, "plan",     "build briefs\nfrom articles"),
    (4.9,  11.8, "draft",    "LLM script\ngeneration"),
    (8.2,  11.8, "validate", "Pydantic\nschema check"),
    (11.5, 11.8, "critique", "Anthropic\ntool use API"),
    (14.8, 11.8, "compress", "fit audio\nbudget"),
]
for nx, ny, nlbl, nsub in lg_nodes:
    c = C["anthropic"] if nlbl == "critique" else C["langgraph"]
    box(nx, ny, 3.0, 2.2, c, nlbl, nsub)

# forward arrows
for i in range(len(lg_nodes) - 1):
    x1 = lg_nodes[i][0] + 3.0
    x2 = lg_nodes[i+1][0]
    y  = lg_nodes[i][1] + 1.1
    arrow(x1, y, x2, y, C["langgraph"])

# critique reject loop arrow
ax.annotate(
    "", xy=(6.4, 11.3), xytext=(13.0, 11.3),
    arrowprops=dict(
        arrowstyle="->,head_width=0.22,head_length=0.16",
        color=C["anthropic"], lw=1.5,
        connectionstyle="arc3,rad=0.0",
    ),
    zorder=4,
)
ax.text(9.7, 10.95,
        "reject  (max 2×)  +  revision_instructions",
        ha="center", va="center",
        fontsize=7.5, color=C["anthropic"], style="italic")

# anthropic tool-use detail box
panel(10.0, 14.0, 6.5, 0.55, C["anthropic"], alpha=0.1)
tools_txt = (
    "score_factual_consistency  ·  score_narrative_flow  ·  "
    "score_tone_consistency  ·  score_humanification_readiness  ·  submit_critique"
)
ax.text(13.25, 14.28, tools_txt,
        ha="center", va="center",
        fontsize=6.5, color=C["anthropic"])
ax.text(10.2, 14.28, "🔧 tools:",
        ha="left", va="center",
        fontsize=7, color=C["anthropic"])

# ---------------------------------------------------------------------------
# ARROW: langgraph → humanification
# ---------------------------------------------------------------------------

arrow(9, 10.8, 9, 10.15, C["arrow_hi"],
      "episode {intro, sections[], outro}")

# ---------------------------------------------------------------------------
# HUMANIFICATION  (y: 9.1 – 10.15)
# ---------------------------------------------------------------------------

panel(3.5, 9.1, 11.0, 0.95, C["neutral"], alpha=0.15)
ax.text(9, 9.58,
        "③ HumanificationAgent  —  voice marker insertion",
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=C["text_main"])
ax.text(9, 9.22,
        "<pause>  ·  <breath>  ·  <emm>  ·  <emphasis>",
        ha="center", va="center",
        fontsize=8, color=C["text_sub"])

# ---------------------------------------------------------------------------
# ARROW: humanification → tts
# ---------------------------------------------------------------------------

arrow(9, 9.1, 9, 8.5, C["arrow_hi"])

# ---------------------------------------------------------------------------
# TTS + ASSEMBLER  (y: 7.3 – 8.5)
# ---------------------------------------------------------------------------

panel(1.2, 7.2, 7.0, 1.2, C["infra"], alpha=0.15)
ax.text(4.7, 7.85,
        "④ TTS  —  gTTS",
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=C["text_main"])
ax.text(4.7, 7.45,
        "text → MP3 clips per section",
        ha="center", va="center",
        fontsize=8, color=C["text_sub"])

panel(9.8, 7.2, 7.0, 1.2, C["infra"], alpha=0.15)
ax.text(13.3, 7.85,
        "⑤ AudioAssembler  —  pydub",
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=C["text_main"])
ax.text(13.3, 7.45,
        "intro + sections + outro → final MP3",
        ha="center", va="center",
        fontsize=8, color=C["text_sub"])

arrow(8.2, 7.8, 9.8, 7.8, C["infra"])

# ---------------------------------------------------------------------------
# ARROW: assembler → storage
# ---------------------------------------------------------------------------

arrow(9, 7.2, 9, 6.55, C["arrow_hi"],
      "/mnt/audio/user_{id}_{date}.mp3")

# ---------------------------------------------------------------------------
# INFRASTRUCTURE  (y: 4.0 – 6.5)
# ---------------------------------------------------------------------------

panel(1.2, 3.9, 15.6, 2.5, C["infra"], alpha=0.07)
section_title(1.4, 6.25, "⑥ Infrastructure", C["infra"])

infra = [
    (1.5,  4.55, "PostgreSQL",  "users + episodes",     "🗄"),
    (4.5,  4.55, "Redis",       "Celery broker",        "⚡"),
    (7.5,  4.55, "MinIO",       "audio object store",   "🪣"),
    (10.5, 4.55, "SearxNG",     "metasearch engine",    "🔎"),
    (13.5, 4.55, "vLLM",        "local LLM inference",  "🤖"),
]
for ix, iy, ilbl, isub, iico in infra:
    box(ix, iy, 2.8, 1.5, C["infra"], ilbl, isub, iico)

# ---------------------------------------------------------------------------
# DELIVERY  (y: 1.5 – 3.5)
# ---------------------------------------------------------------------------

panel(1.2, 1.4, 15.6, 1.9, C["neutral"], alpha=0.07)
section_title(1.4, 3.15, "⑦ Delivery", C["neutral"])

arrow(9, 3.9, 9, 3.35, C["arrow_hi"])

delivery = [
    (2.5,  1.65, "RSS Feed",    "/feed/{user_id}.rss",     "📻"),
    (7.5,  1.65, "REST API",    "/episodes/{id}/latest",   "🔗"),
    (12.5, 1.65, "Email",       "MP3 link on generation",  "📧"),
]
for dx, dy, dlbl, dsub, dico in delivery:
    box(dx, dy, 3.5, 1.4, C["neutral"], dlbl, dsub, dico)

# ---------------------------------------------------------------------------
# LEGEND
# ---------------------------------------------------------------------------

legend_items = [
    (C["crewai"],    "CrewAI  (retrieval)"),
    (C["langgraph"], "LangGraph  (script pipeline)"),
    (C["anthropic"], "Anthropic tool use  (critique)"),
    (C["infra"],     "Infrastructure"),
    (C["neutral"],   "Utilities / Delivery"),
]
lx, ly = 1.4, 0.85
for i, (lc, lt) in enumerate(legend_items):
    rect = FancyBboxPatch(
        (lx + i * 3.35, ly - 0.18), 0.4, 0.36,
        boxstyle="round,pad=0.05",
        facecolor=lc, edgecolor=lc, alpha=0.8, zorder=5,
    )
    ax.add_patch(rect)
    ax.text(lx + i * 3.35 + 0.55, ly,
            lt, ha="left", va="center",
            fontsize=7.5, color=C["text_label"], zorder=5)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

os.makedirs("docs", exist_ok=True)
out = "docs/architecture.png"
plt.tight_layout(pad=0)
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {out}")
