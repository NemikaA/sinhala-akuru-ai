import os
import urllib.request

FONT_URL = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansSinhala/NotoSansSinhala-Bold.ttf"
TARGET_FILE = os.path.join(os.path.dirname(__file__), "NotoSansSinhala-Bold.ttf")

def download_font():
    if os.path.exists(TARGET_FILE):
        print(f"Font already exists at: {TARGET_FILE}")
        return
    print(f"Downloading NotoSansSinhala-Bold.ttf from Google Fonts repository...")
    try:
        urllib.request.urlretrieve(FONT_URL, TARGET_FILE)
        print(f"✔ Downloaded font to: {TARGET_FILE}")
    except Exception as e:
        print(f"Warning: Could not download font automatically ({e}).")
        print("You can place any Sinhala Unicode font (.ttf) here manually or rely on Windows system font (Iskoola Pota).")

if __name__ == "__main__":
    download_font()
