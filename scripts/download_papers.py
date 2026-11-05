"""Download the open-access arXiv papers used as the document set into data/papers/."""
import sys
import urllib.request
from pathlib import Path

PAPERS = {
    "attention_is_all_you_need.pdf": "1706.03762",
    "adam_optimizer.pdf": "1412.6980",
    "batch_normalization.pdf": "1502.03167",
    "wavenet.pdf": "1609.03499",
    "resnet.pdf": "1512.03385",
}
OUT = Path(__file__).resolve().parents[1] / "data" / "papers"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, arxiv_id in PAPERS.items():
        dest = OUT / name
        if dest.exists():
            print(f"skip {name} (exists)")
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "rag-evaluation-lab/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if not data.startswith(b"%PDF"):
            sys.exit(f"{url} did not return a PDF")
        dest.write_bytes(data)
        print(f"saved {name} ({len(data) // 1024} KB)")


if __name__ == "__main__":
    main()
