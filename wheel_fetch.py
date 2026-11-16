"""Офлайн-загрузчик колёс: pip блокирован прокси, urllib - нет.

python wheel_fetch.py <pkg> [pkg==ver ...]  -> _wheels/*.whl
Разрешение версий: pypi JSON (последняя стабильная с win-совместимым
колесом py3-none-any / cp310 / cp3-cp310); torch берем с
download.pytorch.org/whl/cu121.
"""
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WHEELS = os.path.join(HERE, "train", "_wheels")
os.makedirs(WHEELS, exist_ok=True)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pip/24.0"})
    return urllib.request.urlopen(req, timeout=120).read()


def good_wheel(name, py_ver="cp310"):
    n = name.lower()
    if not n.endswith(".whl"):
        return False
    if "py3-none-any" in n:
        return True
    if "win_amd64" not in n:
        return False
    return py_ver in n or "abi3" in n


def from_pypi(pkg):
    base = pkg.split("==")[0]
    want = pkg.split("==")[1] if "==" in pkg else None
    data = json.loads(fetch(f"https://pypi.org/pypi/{base}/json"))
    ver = want or data["info"]["version"]
    rels = data["releases"].get(ver, [])
    for r in rels:
        if good_wheel(r["filename"]):
            return r["url"], r["filename"], ver
    # искать среди всех версий вниз
    for v in sorted(data["releases"], reverse=True):
        for r in data["releases"][v]:
            if good_wheel(r["filename"]):
                return r["url"], r["filename"], v
    raise RuntimeError(f"нет подходящего колеса для {pkg}")


def from_pytorch(pkg="torch"):
    html = fetch("https://download.pytorch.org/whl/cu121/torch/").decode()
    cands = re.findall(r'href="([^"]+torch-[^"]+cp310-cp310-win_amd64\.whl)[^"]*"', html)
    if not cands:
        raise RuntimeError("cu121 cp310 win wheel не найден")
    url = cands[-1]
    return url, url.split("/")[-1], "cu121"


for pkg in sys.argv[1:]:
    if pkg.startswith("torch"):
        url, fname, ver = from_pytorch()
    else:
        url, fname, ver = from_pypi(pkg)
    dst = os.path.join(WHEELS, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] {fname}")
        continue
    print(f"[get ] {pkg} -> {fname} ({ver})", flush=True)
    data = fetch(url)
    with open(dst, "wb") as f:
        f.write(data)
    print(f"[ ok ] {fname} {len(data)/1e6:.0f} МБ", flush=True)
