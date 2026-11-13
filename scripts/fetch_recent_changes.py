"""Récupère les dernières modifications de wikixd.fabmob.io et les écrit dans widget/data.json.
Lancé automatiquement par GitHub Actions (voir .github/workflows/wiki-widget.yml)."""
import json, urllib.request, urllib.parse, datetime as dt, xml.etree.ElementTree as ET
from pathlib import Path

WIKI = "https://wikixd.fabmob.io"
API = WIKI + "/api.php"
DAYS, LIMIT = 7, 50
UA = {"User-Agent": "wikixd-widget/1.0 (https://github.com/GabrielPlassat/wikixd)"}
OUT = Path(__file__).resolve().parent.parent / "widget" / "data.json"

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read()

def page_url(title):
    return f"{WIKI}/index.php?title={urllib.parse.quote(title.replace(' ', '_'))}"

def from_json_api():
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    p = urllib.parse.urlencode({
        "action": "query", "list": "recentchanges", "format": "json", "formatversion": "2",
        "rcprop": "title|timestamp|user|comment|sizes|ids|flags", "rcshow": "!bot",
        "rctype": "edit|new", "rclimit": LIMIT, "rcend": since})
    data = json.loads(get(f"{API}?{p}"))
    out = []
    for c in data["query"]["recentchanges"]:
        new = c.get("type") == "new"
        out.append({
            "title": c["title"], "url": page_url(c["title"]),
            "diff": None if new else f"{page_url(c['title'])}&diff={c['revid']}&oldid={c['old_revid']}",
            "date": c["timestamp"], "user": c.get("user", ""), "comment": c.get("comment", ""),
            "delta": c.get("newlen", 0) - c.get("oldlen", 0), "isNew": new})
    return out

def from_atom():
    feed = (f"{API}?hidebots=1&translations=filter&urlversion=2&days={DAYS}&limit={LIMIT}"
            "&action=feedrecentchanges&feedformat=atom")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(get(feed))
    out = []
    for e in root.findall("a:entry", ns):
        title = e.findtext("a:title", "", ns)
        link = (e.find("a:link", ns).get("href") if e.find("a:link", ns) is not None else "")
        new = "diff=" not in link
        out.append({"title": title, "url": page_url(title), "diff": None if new else link,
                    "date": e.findtext("a:updated", "", ns), "user": e.findtext("a:author/a:name", "", ns),
                    "comment": "", "delta": None, "isNew": new})
    return out

if __name__ == "__main__":
    try:
        items, source = from_json_api(), "api"
    except Exception as err:
        print("API JSON indisponible :", err)
        items, source = from_atom(), "atom"
    OUT.write_text(json.dumps({
        "generatedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} modifications écrites dans {OUT} (source : {source})")
