import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, url_for


# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image
import cv2
import numpy as np

# ---------------- Flask Setup ----------------
app = Flask(__name__)
BASE_DIR = Path(__file__).parent

# Where to store user uploads and results (web-accessible via /static/...)
STATIC_DIR = BASE_DIR / "static"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
RESULT_FOLDER = STATIC_DIR / "results"
FRAME_FOLDER = STATIC_DIR / "frames"

for p in (UPLOAD_FOLDER, RESULT_FOLDER, FRAME_FOLDER):
    os.makedirs(p, exist_ok=True)

ALLOWED_IMAGE_EXTS = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXTS = {"mp4", "mov", "avi", "mkv"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Load Image Model (.pt) ----------------
IMAGE_MODEL_PATH = BASE_DIR / "deepfake_facecrop_best.pth"
THRESH = 0.5   # threshold for REAL

from torchvision.models import resnet18, ResNet18_Weights

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

def load_image_model(path):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features,256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256,2)
    )

    state_dict = torch.load(path, map_location=device)

    # Remove module. prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"🔥 FaceCrop Model loaded: {path}")
    return model

image_model = load_image_model(IMAGE_MODEL_PATH)

# ---------------- Load Video Model (.pth) ----------------
VIDEO_MODEL_PATH = BASE_DIR / "global_model_round3.pth"
def load_video_resnet18(path):
    if not path.exists():
        print("WARNING: video model not found:", path)
        return None
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    state_dict = torch.load(str(path), map_location=device)
    # strip DataParallel
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"🎥 Video model loaded from {path}")
    return model

video_model = load_video_resnet18(VIDEO_MODEL_PATH)


# ---------------- Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ---------------- IMAGE PREDICTION (simple wrapper) ----------------
def predict_image(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    probs_list = []

    if len(faces) == 0:
        # fallback: whole image
        img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = image_model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        probs_list.append(probs)
    else:
        for (x,y,w,h) in faces:
            crop = img_bgr[y:y+h, x:x+w]
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = image_model(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            probs_list.append(probs)

    avg = np.mean(np.stack(probs_list), axis=0)

    fake_prob = float(avg[0])
    real_prob = float(avg[1])

    pred_index = int(np.argmax(avg))
    pred = ["Fake", "Real"][pred_index]

    return pred, fake_prob, real_prob


# ----------------------------------------------------------
# Create GIF animation (optional)
# ----------------------------------------------------------
def create_animation(frame_probs):
    if not frame_probs:
        return None
    x = list(range(len(frame_probs)))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(10, len(x)))
    ax.set_xlabel("Frame index (sampled)")
    ax.set_ylabel("Fake probability")
    ax.grid(True, alpha=0.3)

    line, = ax.plot([], [], color="#00E5FF", lw=2)

    def update(i):
        line.set_data(x[:i], frame_probs[:i])
        return (line,)

    ani = animation.FuncAnimation(fig, update, frames=len(x)+1, interval=180, blit=True, repeat=False)
    name = f"{uuid.uuid4().hex}_animated.gif"
    path = RESULT_FOLDER / name
    ani.save(str(path), writer="pillow")
    plt.close(fig)
    return f"/static/results/{name}"


# ---------------- VIDEO PREDICTION ----------------
def predict_video(video_path, sample_every_n_frames=10):
    """
    Process video, sample frames, run video_model on each sampled frame.
    Returns:
        avg_fake: float
        line_img_rel: str (relative static image path for line)
        bar_img_rel: str (relative static image path for bar)
        frame_probs: list[float] (per sampled frame)
        animation_rel: str or None (rel path to GIF)
        best_frame_rel: str (rel path to saved best frame image)
        best_frame_index_original: int (frame number in original video, approximate)
        result_type: "fake" or "real"
        best_value: float (value of best)
    """
    cap = cv2.VideoCapture(str(video_path))
    sampled_frames = []   # list of BGR frames (numpy arrays)
    frame_probs = []
    frame_idx = 0
    sampled_frame_indices = []  # original frame index for each sample

    if video_model is None:
        # model missing: return safe defaults
        return 0.0, None, None, [], None, None, None, None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n_frames == 0:
            sampled_frame_indices.append(frame_idx)
            sampled_frames.append(frame.copy())
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            x = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = video_model(x)
                probs = torch.softmax(out, dim=1)
                fake_prob = float(probs[0, 1].cpu().item())
            frame_probs.append(fake_prob)
        frame_idx += 1

    cap.release()

    if len(frame_probs) == 0:
        return 0.0, None, None, [], None, None, None, None, None

    avg_fake = float(np.mean(frame_probs))
    x_vals = list(range(len(frame_probs)))

    # -- STATIC LINE IMAGE (cyan, keep simple) --
    fig = plt.figure(figsize=(10, 3))
    plt.plot(x_vals, frame_probs, linewidth=2, color="cyan")
    plt.fill_between(x_vals, frame_probs, alpha=0.15, color="cyan")
    plt.ylim(0, 1)
    plt.xlabel("Sample index")
    plt.ylabel("Fake probability")
    plt.title("Fake Probability (Line)")
    line_name = f"{uuid.uuid4().hex}_line.png"
    plt.savefig(RESULT_FOLDER / line_name, bbox_inches="tight")
    plt.close(fig)
    line_img_rel = f"/static/results/{line_name}"

    # -- STATIC BAR IMAGE (per-bar color: red >0.5 else green) --
    bar_colors = ["#FF5252" if p > 0.75 else "#4CAF50" for p in frame_probs]
    fig2 = plt.figure(figsize=(10, 3))
    plt.bar(x_vals, frame_probs, color=bar_colors)
    plt.ylim(0, 1)
    plt.xlabel("Sample index")
    plt.ylabel("Fake probability")
    plt.title("Fake Probability (Bars)")
    bar_name = f"{uuid.uuid4().hex}_bar.png"
    plt.savefig(RESULT_FOLDER / bar_name, bbox_inches="tight")
    plt.close(fig2)
    bar_img_rel = f"/static/results/{bar_name}"

    # Determine best frame:
    if avg_fake > 0.75:
        # fake video -> choose highest fake
        best_idx_sample = int(np.argmax(frame_probs))
        result_type = "fake"
        best_value = frame_probs[best_idx_sample]
    else:
        # real video -> choose highest real = 1 - fake
        real_probs = [1.0 - p for p in frame_probs]
        best_idx_sample = int(np.argmax(real_probs))
        result_type = "real"
        best_value = real_probs[best_idx_sample]

    # Map sample index to original frame index and frame image
    best_frame_original_index = sampled_frame_indices[best_idx_sample]
    best_frame_bgr = sampled_frames[best_idx_sample]

    # Save best frame
    best_frame_name = f"{uuid.uuid4().hex}_best.jpg"
    best_frame_path = FRAME_FOLDER / best_frame_name
    cv2.imwrite(str(best_frame_path), best_frame_bgr)
    best_frame_rel = f"/static/frames/{best_frame_name}"

    # Create animation GIF (optional)
    animation_rel = create_animation(frame_probs)

    return (
        avg_fake,
        line_img_rel,
        bar_img_rel,
        frame_probs,
        animation_rel,
        best_frame_rel,
        best_frame_original_index,
        result_type,
        float(best_value),
    )


# ---------------- FLASK ROUTES ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/image", methods=["GET", "POST"])
def image_detection():
    result = None
    upload_url = None

    if request.method == "POST":
        file = request.files.get("image_file")

        if file and file.filename.split(".")[-1].lower() in ALLOWED_IMAGE_EXTS:

            # create unique file name
            ext = file.filename.split(".")[-1]
            unique_name = f"{uuid.uuid4().hex}.{ext}"
            filepath = UPLOAD_FOLDER / unique_name
            file.save(str(filepath))

            # URL to display image
            upload_url = f"/static/uploads/{unique_name}"

            # Face-crop prediction
            pred, fake_prob, real_prob = predict_image(str(filepath))

            # UI color
            color = "#4CAF50" if pred == "Real" else "#FF5252"

            result = (
                f"<b style='color:{color}; font-size:22px'>{pred}</b> "
                f"(fake={fake_prob:.3f}, real={real_prob:.3f})"
            )

    return render_template("image.html", prediction=result, upload_url=upload_url)

@app.route("/video", methods=["GET", "POST"])
def video_detection():
    # default safe values
    prediction = None
    details = {
        "line_graph": None,
        "bar_graph": None,
        "animation_graph": None,
        "frame_probs": [],
        "best_frame_path": None,
        "best_frame_idx": None,
        "result_type": None,
        "best_frame_value": None,
        "video_path": None,
    }

    if request.method == "POST":
        file = request.files.get("video_file")
        if file and file.filename.split(".")[-1].lower() in ALLOWED_VIDEO_EXTS:
            filename = secure_filename(file.filename)
            save_path = UPLOAD_FOLDER / filename
            file.save(str(save_path))

            (
                avg,
                line_graph,
                bar_graph,
                frame_probs,
                animation_graph,
                best_frame_path,
                best_frame_idx,
                result_type,
                best_value,
            ) = predict_video(str(save_path))

            label = "Fake" if avg > 0.75 else "Real"
            color = "#FF5252" if label == "Fake" else "#4CAF50"
            prediction = {
                "label": label,
                "color": color,
                "avg": round(float(avg), 3),
                "video_path": f"/static/uploads/{filename}",
            }

            details.update({
                "line_graph": line_graph,
                "bar_graph": bar_graph,
                "animation_graph": animation_graph,
                "frame_probs": frame_probs or [],
                "best_frame_path": best_frame_path,
                "best_frame_idx": best_frame_idx,
                "result_type": result_type,
                "best_frame_value": round(float(best_value), 4) if best_value is not None else None,
                "video_path": f"/static/uploads/{filename}",
            })

    return render_template("video.html", prediction=prediction, details=details)
    


if __name__ == "__main__":
    app.run(debug=True)
