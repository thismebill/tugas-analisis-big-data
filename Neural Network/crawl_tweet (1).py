"""
Langkah 1 - Crawling tweet topik "Kualitas Udara"
Jalankan di komputer Anda (butuh Node.js dan auth_token akun X/Twitter).

Cara paling mudah: tweet-harvest
    1. Install Node.js (https://nodejs.org)
    2. Login X/Twitter di browser, buka DevTools > Application > Cookies,
       salin nilai cookie "auth_token".
    3. Isi AUTH_TOKEN di bawah, lalu: python crawl_tweet.py

Hasil akhir: tweet_data.csv dengan kolom created_at, username, full_text
"""
import subprocess
import pandas as pd

AUTH_TOKEN = "ISI_AUTH_TOKEN_ANDA"
KEYWORD = "kualitas udara lang:id"   # bisa ganti: "polusi udara", "kabut asap", "ISPU"
LIMIT = 500                           # minimal 300, ambil lebih agar aman setelah dedup
RAW_FILE = "raw_tweets.csv"

cmd = [
    "npx", "-y", "tweet-harvest@latest",
    "-o", RAW_FILE,
    "-s", KEYWORD,
    "-l", str(LIMIT),
    "--token", AUTH_TOKEN,
]
print("Menjalankan:", " ".join(cmd[:-1]), "***")
subprocess.run(cmd, check=True)

# tweet-harvest menyimpan ke folder tweets-data/
df = pd.read_csv(f"tweets-data/{RAW_FILE}")
df = df.rename(columns={"user_screen_name": "username"}) if "user_screen_name" in df.columns else df
df = df[["created_at", "username", "full_text"]].dropna(subset=["full_text"])
df = df.drop_duplicates(subset="full_text").reset_index(drop=True)
df.to_csv("tweet_data.csv", index=False, encoding="utf-8")
print(f"Tersimpan tweet_data.csv: {len(df)} tweet")
if len(df) < 300:
    print("PERINGATAN: kurang dari 300 tweet, tambah LIMIT atau ganti/tambah keyword.")
