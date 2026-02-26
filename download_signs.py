import requests
import os

os.makedirs("static/signs", exist_ok=True)

# ASL sign GIF sources
SIGNS = {
    "hello":   "https://www.handspeak.com/word/h/hello.gif",
    "stop":    "https://www.handspeak.com/word/s/stop.gif",
    "welcome": "https://www.handspeak.com/word/w/welcome.gif",
    "help":    "https://www.handspeak.com/word/h/help.gif",
    "yes":     "https://www.handspeak.com/word/y/yes.gif",
    "no":      "https://www.handspeak.com/word/n/no.gif",
    "please":  "https://www.handspeak.com/word/p/please.gif",
    "sign":    "https://www.handspeak.com/word/s/sign.gif",
}

headers = {"User-Agent": "Mozilla/5.0"}

for name, url in SIGNS.items():
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200 and len(r.content) > 500:
            path = f"static/signs/{name}.gif"
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"[OK] {name} -> {path} ({len(r.content)} bytes)")
        else:
            print(f"[FAIL] {name} -> HTTP {r.status_code}")
    except Exception as e:
        print(f"[ERROR] {name} -> {e}")

print("Done!")
