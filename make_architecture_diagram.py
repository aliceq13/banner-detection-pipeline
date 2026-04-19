"""
Renders a clean system-architecture diagram (PNG) for the banner pipeline.
Uses only PIL — no matplotlib / graphviz.

Output: results/architecture.png
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "results" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------- canvas ----------
W, H = 1600, 1260
BG = (255, 255, 255)
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# ---------- fonts ----------
def load(size, bold=False):
    candidates_bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    candidates_reg = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in (candidates_bold if bold else candidates_reg):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

F_TITLE    = load(38, bold=True)
F_SUBTITLE = load(20)
F_STAGE    = load(28, bold=True)
F_MODEL    = load(21, bold=True)
F_SUB      = load(19)
F_IO       = load(17)
F_SMALL    = load(15)
F_ARROW    = load(16, bold=True)
F_NUM      = load(22, bold=True)

# ---------- palette ----------
COL_INPUT  = (240, 240, 240)
COL_DETECT = (208, 232, 255)
COL_GEOM   = (218, 242, 216)
COL_VLM    = (255, 226, 195)
COL_OUTPUT = (255, 228, 233)
BORDER     = (60, 60, 60)
ARROW      = (90, 90, 90)
TEXT_DARK  = (20, 20, 20)
TEXT_MUTED = (95, 95, 95)
TEXT_IO    = (40, 70, 130)
TEXT_HL    = (160, 40, 40)

# ---------- title ----------
title = "Banner Detection Pipeline  ·  System Architecture"
tw = d.textlength(title, font=F_TITLE)
d.text(((W - tw) / 2, 30), title, font=F_TITLE, fill=TEXT_DARK)
sub = "Detection  →  Perspective Rectification  →  Multimodal Classification"
sw = d.textlength(sub, font=F_SUBTITLE)
d.text(((W - sw) / 2, 80), sub, font=F_SUBTITLE, fill=TEXT_MUTED)

# ---------- helpers ----------
def rounded_box(xy, fill, border=BORDER, radius=16, width=3):
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=border, width=width)

def centered_text(x_center, y, text, font, fill):
    tw = d.textlength(text, font=font)
    d.text((x_center - tw / 2, y), text, font=font, fill=fill)
    _, top, _, bot = font.getbbox(text)
    return (bot - top)

def arrow(x, y0, y1, label=None):
    d.line([(x, y0), (x, y1 - 16)], fill=ARROW, width=4)
    d.polygon([(x - 12, y1 - 16), (x + 12, y1 - 16), (x, y1)], fill=ARROW)
    if label:
        d.text((x + 22, (y0 + y1) / 2 - 11), label, font=F_ARROW, fill=ARROW)

# ---------- layout ----------
CX = W // 2
BOX_W = 760
X0, X1 = CX - BOX_W // 2, CX + BOX_W // 2

# Input
y = 150
box_h = 72
rounded_box((X0, y, X1, y + box_h), COL_INPUT)
line_h = centered_text(CX, y + 14, "Input Image", F_STAGE, TEXT_DARK)
centered_text(CX, y + 14 + line_h + 4, "street scene  /  CCTV frame", F_SUB, TEXT_MUTED)
y_bottom = y + box_h

# Stage 1
arrow(CX, y_bottom, y_bottom + 70, "RGB image")
y = y_bottom + 70
box_h = 180
rounded_box((X0, y, X1, y + box_h), COL_DETECT)
# stage index pill
d.rounded_rectangle((X0 + 22, y + 22, X0 + 88, y + 58), radius=12, fill=(60, 110, 180))
centered_text(X0 + 55, y + 26, "1", F_NUM, (255, 255, 255))
centered_text(CX, y + 28, "Detection", F_STAGE, TEXT_DARK)
centered_text(CX, y + 72, "YOLO26-seg    (medium,  imgsz 640)", F_MODEL, TEXT_DARK)
centered_text(CX, y + 104, "trained with COCO negatives   →   15× lower false-positive rate", F_SUB, TEXT_HL)
centered_text(CX, y + 138, "output:   segmentation polygon   +   boolean mask   +   bbox", F_IO, TEXT_IO)
y_bottom = y + box_h

# Stage 2
arrow(CX, y_bottom, y_bottom + 70, "polygon")
y = y_bottom + 70
box_h = 180
rounded_box((X0, y, X1, y + box_h), COL_GEOM)
d.rounded_rectangle((X0 + 22, y + 22, X0 + 88, y + 58), radius=12, fill=(70, 140, 80))
centered_text(X0 + 55, y + 26, "2", F_NUM, (255, 255, 255))
centered_text(CX, y + 28, "Perspective Rectification", F_STAGE, TEXT_DARK)
centered_text(CX, y + 72, "4-extreme-corner homography    (exact  4-point solution)", F_MODEL, TEXT_DARK)
centered_text(CX, y + 104, "output size from measured side lengths   →   aspect ratio preserved", F_SUB, TEXT_HL)
centered_text(CX, y + 138, "output:   fronto-parallel banner patch", F_IO, TEXT_IO)
y_bottom = y + box_h

# Stage 3
arrow(CX, y_bottom, y_bottom + 70, "rectified patch")
y = y_bottom + 70
box_h = 220
rounded_box((X0, y, X1, y + box_h), COL_VLM)
d.rounded_rectangle((X0 + 22, y + 22, X0 + 88, y + 58), radius=12, fill=(205, 120, 40))
centered_text(X0 + 55, y + 26, "3", F_NUM, (255, 255, 255))
centered_text(CX, y + 28, "Multimodal Classification", F_STAGE, TEXT_DARK)
centered_text(CX, y + 72, "Qwen3.5-9B    (image-text-to-text,  BF16)", F_MODEL, TEXT_DARK)
centered_text(CX, y + 106, "3-step prompt:   text extraction  →  reasoning  →  category", F_SUB, TEXT_DARK)
centered_text(CX, y + 138, "thinking mode ON    ·    input long-side 896 px", F_SUB, TEXT_HL)
centered_text(CX, y + 178, "output:   { category,  confidence,  extracted_text }", F_IO, TEXT_IO)
y_bottom = y + box_h

# Final Output
arrow(CX, y_bottom, y_bottom + 70)
y = y_bottom + 70
box_h = 90
rounded_box((X0, y, X1, y + box_h), COL_OUTPUT)
centered_text(CX, y + 18, "Final Output", F_STAGE, TEXT_DARK)
centered_text(CX, y + 60, "Party banner    /    Private banner    /    Public banner", F_MODEL, TEXT_DARK)

# ---------- footer caption ----------
footer = "evaluated on GT-matched crops via IoU ≥ 0.3     |     3-class output     |     single-GPU sequential execution"
fw = d.textlength(footer, font=F_SMALL)
d.text(((W - fw) / 2, H - 30), footer, font=F_SMALL, fill=(130, 130, 130))

img.save(OUT, "PNG")
print(f"[ok] saved: {OUT}   ({W} x {H})")
