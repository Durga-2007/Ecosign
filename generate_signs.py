"""
Generate realistic-looking hand sign images for ECOSIGN.
Uses PIL to draw actual hand shapes.
"""
from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("static/signs", exist_ok=True)

W, H = 220, 220
BG = (245, 235, 255)
SKIN = (255, 213, 170)
SKIN_DARK = (220, 170, 120)
PURPLE = (140, 74, 216)
WHITE = (255, 255, 255)

def new_img():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W-1, H-1], outline=PURPLE, width=4)
    return img, draw

def add_label(draw, text):
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.rectangle([0, H-36, W, H], fill=PURPLE)
    draw.text((W//2, H-18), text, font=font, fill=WHITE, anchor="mm")

def draw_palm(draw, cx, cy, r=55):
    """Draw open palm (hand facing outward)."""
    # Palm base
    draw.ellipse([cx-r, cy-20, cx+r, cy+50], fill=SKIN, outline=SKIN_DARK, width=2)
    # Thumb
    draw.ellipse([cx-r-15, cy-35, cx-r+18, cy+5], fill=SKIN, outline=SKIN_DARK, width=2)
    # Fingers
    finger_w = 18
    for i, fx in enumerate([cx-30, cx-10, cx+10, cx+30]):
        flen = [75, 85, 80, 65][i]
        draw.rounded_rectangle([fx-finger_w//2, cy-flen, fx+finger_w//2, cy], radius=9, fill=SKIN, outline=SKIN_DARK, width=2)

def draw_fist(draw, cx, cy, r=50):
    """Draw a closed fist."""
    draw.rounded_rectangle([cx-r, cy-35, cx+r, cy+30], radius=20, fill=SKIN, outline=SKIN_DARK, width=2)
    # Knuckle lines
    for kx in [cx-25, cx-8, cx+8, cx+25]:
        draw.arc([kx-8, cy-38, kx+8, cy-20], 0, 180, fill=SKIN_DARK, width=2)
    # Thumb
    draw.ellipse([cx+r-15, cy-10, cx+r+18, cy+30], fill=SKIN, outline=SKIN_DARK, width=2)

def draw_index_pointing(draw, cx, cy):
    """Draw hand with index finger pointing up."""
    # Palm
    draw.rounded_rectangle([cx-35, cy, cx+35, cy+55], radius=15, fill=SKIN, outline=SKIN_DARK, width=2)
    # Index finger (extended)
    draw.rounded_rectangle([cx-10, cy-70, cx+10, cy+10], radius=9, fill=SKIN, outline=SKIN_DARK, width=2)
    # Folded fingers (small bumps)
    for fx in [cx-25, cx+10, cx+26]:
        draw.rounded_rectangle([fx-8, cy-10, fx+8, cy+15], radius=7, fill=SKIN, outline=SKIN_DARK, width=2)
    # Thumb
    draw.ellipse([cx+30, cy+5, cx+52, cy+35], fill=SKIN, outline=SKIN_DARK, width=2)

def draw_two_hands(draw, cx, cy):
    """Draw two open palms side by side."""
    # Left hand
    lx = cx - 45
    draw.ellipse([lx-35, cy-15, lx+35, cy+45], fill=SKIN, outline=SKIN_DARK, width=2)
    for i, fx in enumerate([lx-20, lx-5, lx+10, lx+22]):
        flen = [55, 65, 58, 48][i]
        draw.rounded_rectangle([fx-7, cy-flen, fx+7, cy], radius=6, fill=SKIN, outline=SKIN_DARK, width=2)
    # Right hand
    rx = cx + 45
    draw.ellipse([rx-35, cy-15, rx+35, cy+45], fill=SKIN, outline=SKIN_DARK, width=2)
    for i, fx in enumerate([rx-22, rx-10, rx+5, rx+20]):
        flen = [48, 58, 65, 55][i]
        draw.rounded_rectangle([fx-7, cy-flen, fx+7, cy], radius=6, fill=SKIN, outline=SKIN_DARK, width=2)

def draw_thumb_up(draw, cx, cy):
    """Draw thumbs up."""
    # Fist body
    draw.rounded_rectangle([cx-35, cy-10, cx+35, cy+55], radius=18, fill=SKIN, outline=SKIN_DARK, width=2)
    # Thumb pointing up
    draw.rounded_rectangle([cx-45, cy-65, cx-18, cy+5], radius=12, fill=SKIN, outline=SKIN_DARK, width=2)

def draw_nodding_fist(draw, cx, cy):
    """Draw a fist (YES sign)."""
    draw_fist(draw, cx, cy)
    # Motion lines above
    for i, my in enumerate([cy-55, cy-65, cy-75]):
        draw.arc([cx-20, my-5, cx+20, my+5], 0, 180, fill=PURPLE, width=2)

def draw_two_fingers_close(draw, cx, cy):
    """Draw index + middle finger closing (NO sign)."""
    # Palm
    draw.rounded_rectangle([cx-35, cy+10, cx+35, cy+60], radius=15, fill=SKIN, outline=SKIN_DARK, width=2)
    # Index finger
    draw.rounded_rectangle([cx-20, cy-55, cx-4, cy+20], radius=8, fill=SKIN, outline=SKIN_DARK, width=2)
    # Middle finger
    draw.rounded_rectangle([cx+4, cy-55, cx+20, cy+20], radius=8, fill=SKIN, outline=SKIN_DARK, width=2)
    # Thumb
    draw.ellipse([cx+28, cy+10, cx+50, cy+40], fill=SKIN, outline=SKIN_DARK, width=2)

def draw_flat_chest(draw, cx, cy):
    """Draw flat hand at chest moving down (PLEASE)."""
    draw.ellipse([cx-45, cy-15, cx+45, cy+30], fill=SKIN, outline=SKIN_DARK, width=2)
    for i, fx in enumerate([cx-28, cx-9, cx+9, cx+28]):
        flen = [48, 60, 60, 48][i]
        draw.rounded_rectangle([fx-9, cy-flen, fx+9, cy], radius=8, fill=SKIN, outline=SKIN_DARK, width=2)
    # Circular motion arrow
    draw.arc([cx-55, cy+20, cx+55, cy+70], 20, 340, fill=PURPLE, width=3)
    # Arrow head
    draw.polygon([(cx+55, cy+45), (cx+45, cy+35), (cx+48, cy+55)], fill=PURPLE)

# ==================== GENERATE ALL SIGNS ====================
signs = [
    ("hello",   draw_palm,           "HELLO"),
    ("stop",    draw_palm,           "STOP"),
    ("welcome", draw_two_hands,      "WELCOME"),
    ("help",    draw_thumb_up,       "HELP"),
    ("yes",     draw_nodding_fist,   "YES"),
    ("no",      draw_two_fingers_close, "NO"),
    ("please",  draw_flat_chest,     "PLEASE"),
    ("sign",    draw_index_pointing, "SIGN"),
]

for key, draw_fn, label in signs:
    img, draw = new_img()
    draw_fn(draw, W//2, H//2 - 10)
    add_label(draw, label)
    path = f"static/signs/{key}.png"
    img.save(path)
    print(f"[OK] {key} -> {path}")

print("All hand sign images generated!")
