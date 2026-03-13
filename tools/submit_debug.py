#!/usr/bin/python3
"""Debug script to submit PrimeNet result and show full response."""
import sys
from http import cookiejar
from urllib.parse import urlencode
from urllib.request import build_opener, HTTPCookieProcessor

baseUrl = "https://www.mersenne.org/"
primenet = build_opener(HTTPCookieProcessor(cookiejar.CookieJar()))

pw = open("/mnt/d/AI/Frey-PRPLL/run/primenet_password.txt").read().strip()

# Login
login_data = urlencode({"user_login": "Frey-Sherpa", "user_password": pw}).encode('utf-8')
r = primenet.open(baseUrl, login_data).read().decode("utf-8")
if "Frey-Sherpa<br>logged in" not in r:
    print("LOGIN FAILED")
    sys.exit(1)
print("Login OK")

# Submit result
result = open("/mnt/d/AI/Frey-PRPLL/run/results.txt").read().strip()
data = urlencode({"data": result}).encode('utf-8')
res = primenet.open(baseUrl + "manual_result/default.php", data).read().decode("utf-8")

# Show relevant parts of response
for keyword in ["error", "Error", "credit", "Credit", "success", "reject", "thank", "Thank", "already", "result"]:
    start = 0
    while True:
        idx = res.find(keyword, start)
        if idx == -1:
            break
        # Print surrounding context
        snippet = res[max(0, idx-50):idx+100].replace('\n', ' ').strip()
        print(f"Found '{keyword}' at {idx}: ...{snippet}...")
        start = idx + 1

print("\n--- Full response body (trimmed) ---")
# Find the main content
main_start = res.find("<main")
main_end = res.find("</main>")
if main_start >= 0 and main_end >= 0:
    import re
    main_html = res[main_start:main_end]
    clean = re.sub('<[^>]+>', ' ', main_html)
    clean = re.sub(r'\s+', ' ', clean).strip()
    print(clean[:500])
else:
    print(res[:500])
