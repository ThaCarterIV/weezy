import os
import json
import sys
import numpy as np
import tensorflow as tf
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Spotify authentication
SCOPE = "user-library-read user-library-modify user-top-read"
CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

if not all([CLIENT_ID, CLIENT_SECRET, REDIRECT_URI]):
    print("Please set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, and SPOTIPY_REDIRECT_URI env vars.")
    sys.exit(1)

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    scope=SCOPE,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    open_browser=False,
))

# File to persist training data
DATA_FILE = "training_data.json"
FEATURE_KEYS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
]

# Load existing dataset
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    # Backwards compatibility if old file format
    if "ids" not in data:
        data["ids"] = [None] * len(data.get("labels", []))
        data.setdefault("features", [])
        data.setdefault("labels", [])
else:
    data = {"ids": [], "features": [], "labels": []}

# Build model
input_shape = (len(FEATURE_KEYS),)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=input_shape),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# If we already have data, train the model
if data["features"]:
    X = np.array(data["features"], dtype=np.float32)
    y = np.array(data["labels"], dtype=np.float32)
    model.fit(X, y, epochs=10, verbose=0)

# Utility to extract features from a track id

def get_features(track_id):
    feats = sp.audio_features([track_id])[0]
    return [feats[k] for k in FEATURE_KEYS]

# Recommend tracks using Spotify's API

def get_recommendations(seed_tracks, limit=10):
    recs = sp.recommendations(seed_tracks=seed_tracks[:5], limit=limit)
    return [t["id"] for t in recs["tracks"]]

# Simple interaction loop

def main():
    # Use user's top tracks as initial seeds
    top = sp.current_user_top_tracks(limit=5)
    seed_ids = [t["id"] for t in top["items"]]

    while True:
        rec_ids = get_recommendations(seed_ids, limit=10)
        for tid in rec_ids:
            track = sp.track(tid)
            name = track["name"]
            artists = ", ".join(a["name"] for a in track["artists"])
            print(f"\n{name} - {artists}")
            preview = track.get("preview_url")
            if preview:
                print(f"Preview: {preview}")
            choice = input("Swipe [r]ight to save, [l]eft to skip, [q]uit: ").strip().lower()
            if choice == "q":
                # Save dataset on exit
                with open(DATA_FILE, "w") as f:
                    json.dump(data, f)
                return
            elif choice in {"r", "l"}:
                label = 1.0 if choice == "r" else 0.0
                if label == 1.0:
                    sp.current_user_saved_tracks_add([tid])
                feats = get_features(tid)
                data["ids"].append(tid)
                data["features"].append(feats)
                data["labels"].append(label)
                # Incremental training
                X = np.array([feats], dtype=np.float32)
                y = np.array([label], dtype=np.float32)
                model.fit(X, y, epochs=1, verbose=0)
        # After a batch, update seed tracks based on likes
        seed_ids = [tid for tid, lbl in zip(data["ids"], data["labels"]) if lbl == 1.0 and tid]
        if not seed_ids:
            seed_ids = rec_ids[:5]

if __name__ == "__main__":
    main()
