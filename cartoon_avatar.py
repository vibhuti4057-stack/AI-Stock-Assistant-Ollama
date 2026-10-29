from PIL import Image, ImageDraw
import os

VISEME_MOUTH_SHAPES = {
    "X": "closed", "A": "closed", "B": "slightly_open", "C": "open_wide",
    "D": "open_round", "E": "open_medium", "F": "small_round",
    "G": "teeth", "H": "wide_smile",
}

EMOTION_COLORS = {
    "happy": (255, 200, 60), "excited": (255, 120, 60), "curious": (150, 100, 255),
    "concerned": (100, 140, 255), "calm": (0, 212, 255), "neutral": (0, 212, 255),
}

def draw_face(mouth_shape="closed", emotion="neutral", filename="face.png"):
    size = 300
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    face_color = EMOTION_COLORS.get(emotion, (0, 212, 255)) + (255,)
    draw.ellipse([20, 20, 280, 280], fill=(30, 39, 97, 255))
    draw.ellipse([30, 30, 270, 270], fill=face_color)

    # Eyebrows shift slightly per emotion for extra expressiveness
    brow_y = 95 if emotion in ("concerned",) else 100
    draw.line([85, brow_y, 125, brow_y - (5 if emotion=="excited" else 0)], fill=(20,20,30,255), width=5)
    draw.line([175, brow_y, 215, brow_y - (5 if emotion=="excited" else 0)], fill=(20,20,30,255), width=5)

    draw.ellipse([90, 110, 120, 140], fill=(255,255,255,255))
    draw.ellipse([180, 110, 210, 140], fill=(255,255,255,255))
    draw.ellipse([100, 120, 115, 135], fill=(10,10,20,255))
    draw.ellipse([190, 120, 205, 135], fill=(10,10,20,255))

    cx, cy = 150, 195
    shapes = {
        "closed":        lambda: draw.line([cx-28, cy, cx+28, cy], fill=(20,20,30,255), width=6),
        "slightly_open": lambda: draw.ellipse([cx-24, cy-7, cx+24, cy+7], fill=(60,20,30,255)),
        "open_wide":     lambda: draw.ellipse([cx-34, cy-24, cx+34, cy+24], fill=(60,20,30,255)),
        "open_round":    lambda: draw.ellipse([cx-18, cy-28, cx+18, cy+28], fill=(60,20,30,255)),
        "open_medium":   lambda: draw.ellipse([cx-26, cy-14, cx+26, cy+14], fill=(60,20,30,255)),
        "small_round":   lambda: draw.ellipse([cx-11, cy-11, cx+11, cy+11], fill=(60,20,30,255)),
        "teeth":         lambda: draw.rectangle([cx-24, cy-4, cx+24, cy+4], fill=(255,255,255,255)),
        "wide_smile":    lambda: draw.arc([cx-34, cy-18, cx+34, cy+24], start=15, end=165, fill=(20,20,30,255), width=6),
    }
    shapes.get(mouth_shape, shapes["closed"])()

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    img.save(filename)
    return filename
def generate_all_faces(emotion="neutral", output_dir="faces"):
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for shape in set(VISEME_MOUTH_SHAPES.values()):
        path = os.path.join(output_dir, f"face_{emotion}_{shape}.png")
        draw_face(shape, emotion, path)
        paths[shape] = path
    return paths

def get_face_path(viseme_letter, emotion="neutral"):
    shape = VISEME_MOUTH_SHAPES.get(viseme_letter, "closed")
    return f"faces/face_{emotion}_{shape}.png"