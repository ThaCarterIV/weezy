# Music Discovery Prototype

This repo demonstrates a simple music discovery app that integrates with
Spotify. Users can swipe through song recommendations and add tracks they
like directly to their Spotify library. A TensorFlow-based model learns from
these swipes to personalize future recommendations.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Spotify API credentials must be supplied via the following environment
variables:

- `SPOTIPY_CLIENT_ID`
- `SPOTIPY_CLIENT_SECRET`
- `SPOTIPY_REDIRECT_URI`

## Running

```bash
python app.py
```

You will be prompted to authenticate with Spotify on first run. Afterwards,
the script presents recommended tracks. Swipe right (enter `r`) to save a song
or left (`l`) to skip. The model updates incrementally based on your
choices.
