"""Fetch the public UCI 'Default of Credit Card Clients' dataset into data/."""
import urllib.request, zipfile
from pathlib import Path
DATA = Path(__file__).resolve().parents[1] / "data"; DATA.mkdir(exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
z = DATA / "cc.zip"
print("downloading", URL)
urllib.request.urlretrieve(URL, z)
zipfile.ZipFile(z).extractall(DATA)
print("done ->", [p.name for p in DATA.glob("*.xls")])
