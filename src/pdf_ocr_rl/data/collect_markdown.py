"""Collect markdown files from open sources for dataset creation."""

import json
import os
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

GITHUB_API = "https://api.github.com"

# Curated list of repos with high-quality markdown docs
ENGLISH_REPOS = [
    "facebook/react",
    "microsoft/TypeScript",
    "golang/go",
    "rust-lang/rust",
    "python/cpython",
    "kubernetes/kubernetes",
    "docker/docs",
    "nodejs/node",
    "vuejs/docs",
    "sveltejs/svelte",
]

JAPANESE_REPOS = [
    "vuejs-translations/docs-ja",
    "electron/i18n",
    "willnet/rspec-style-guide",
]

# Direct URLs for Japanese technical docs
JAPANESE_WIKI_PAGES = [
    "https://ja.wikipedia.org/api/rest_v1/page/html/Python",
    "https://ja.wikipedia.org/api/rest_v1/page/html/人工知能",
    "https://ja.wikipedia.org/api/rest_v1/page/html/機械学習",
    "https://ja.wikipedia.org/api/rest_v1/page/html/深層学習",
    "https://ja.wikipedia.org/api/rest_v1/page/html/自然言語処理",
]


def fetch_github_markdown_files(repo: str, max_files: int = 20, token: str | None = None) -> list[dict]:
    """Fetch markdown files from a GitHub repo via API.

    Returns list of dicts with 'name', 'content', 'path', 'repo'.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    results = []
    url = f"{GITHUB_API}/repos/{repo}/git/trees/HEAD?recursive=1"

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        tree = resp.json().get("tree", [])
    except (requests.RequestException, json.JSONDecodeError):
        print(f"  Failed to fetch tree for {repo}")
        return results

    md_files = [f for f in tree if f["path"].endswith(".md") and f["type"] == "blob"]
    md_files = md_files[:max_files]

    for f in md_files:
        try:
            raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{f['path']}"
            resp = requests.get(raw_url, timeout=15)
            resp.raise_for_status()
            content = resp.text

            # Filter: must be non-trivial (>200 chars, has structure)
            if len(content) < 200:
                continue
            if not any(marker in content for marker in ["#", "|", "```", "- ", "1. "]):
                continue

            results.append({
                "name": os.path.basename(f["path"]),
                "content": content,
                "path": f["path"],
                "repo": repo,
                "language": "en",
            })
        except requests.RequestException:
            continue

        time.sleep(0.2)  # Rate limiting

    return results


def fetch_japanese_wiki_markdown(urls: list[str] | None = None) -> list[dict]:
    """Fetch Japanese Wikipedia articles and convert HTML to simple markdown-like text."""
    if urls is None:
        urls = JAPANESE_WIKI_PAGES

    results = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            html = resp.text

            # Simple HTML to markdown conversion for training data
            text = html
            # Remove script/style tags
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            # Convert headings
            for i in range(6, 0, -1):
                text = re.sub(rf"<h{i}[^>]*>(.*?)</h{i}>", rf"{'#' * i} \1", text, flags=re.DOTALL)
            # Convert paragraphs
            text = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n\n", text, flags=re.DOTALL)
            # Convert lists
            text = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1", text, flags=re.DOTALL)
            # Convert bold/italic
            text = re.sub(r"<b>(.*?)</b>", r"**\1**", text)
            text = re.sub(r"<strong>(.*?)</strong>", r"**\1**", text)
            text = re.sub(r"<i>(.*?)</i>", r"*\1*", text)
            text = re.sub(r"<em>(.*?)</em>", r"*\1*", text)
            # Strip remaining HTML tags
            text = re.sub(r"<[^>]+>", "", text)
            # Clean up whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

            if len(text) > 200:
                title = url.split("/")[-1]
                results.append({
                    "name": f"{title}.md",
                    "content": text,
                    "path": url,
                    "repo": "wikipedia-ja",
                    "language": "ja",
                })
        except requests.RequestException:
            continue

        time.sleep(0.5)

    return results


def create_synthetic_japanese_markdown() -> list[dict]:
    """Create synthetic Japanese markdown documents with various formatting."""
    docs = []

    # Technical document with tables and code
    docs.append({
        "name": "python_basics_ja.md",
        "content": """# Pythonプログラミング基礎

## 概要

Pythonは、汎用プログラミング言語の一つです。コードの可読性が高く、初心者にも学びやすい言語として知られています。

## データ型

| 型名 | 説明 | 例 |
|------|------|-----|
| int | 整数 | `42` |
| float | 浮動小数点数 | `3.14` |
| str | 文字列 | `"こんにちは"` |
| list | リスト | `[1, 2, 3]` |
| dict | 辞書 | `{"名前": "太郎"}` |

## 基本文法

### 変数の定義

```python
名前 = "太郎"
年齢 = 25
print(f"{名前}さんは{年齢}歳です。")
```

### 条件分岐

```python
if 年齢 >= 20:
    print("成人です")
else:
    print("未成年です")
```

### ループ処理

1. **forループ**: リストの各要素を処理
2. **whileループ**: 条件が真の間繰り返す

```python
果物 = ["りんご", "みかん", "ぶどう"]
for item in 果物:
    print(item)
```

## まとめ

- Pythonは読みやすいコードが書ける
- 豊富なライブラリが利用可能
- データサイエンスや機械学習に広く使われている
""",
        "path": "synthetic/python_basics_ja.md",
        "repo": "synthetic-ja",
        "language": "ja",
    })

    # Document with mixed content
    docs.append({
        "name": "ml_overview_ja.md",
        "content": """# 機械学習入門

## 機械学習とは

機械学習（Machine Learning）は、データからパターンを学習し、予測や判断を行う技術です。

## 主要なアルゴリズム

### 教師あり学習

| アルゴリズム | 用途 | 特徴 |
|-------------|------|------|
| 線形回帰 | 回帰 | シンプルで解釈しやすい |
| ロジスティック回帰 | 分類 | 確率出力が可能 |
| ランダムフォレスト | 分類/回帰 | 過学習に強い |
| ニューラルネットワーク | 汎用 | 高い表現力 |

### 教師なし学習

- **クラスタリング**: K-means, DBSCAN
- **次元削減**: PCA, t-SNE
- **異常検知**: Isolation Forest

### 強化学習

強化学習では、エージェントが環境との相互作用を通じて最適な行動方針を学習します。

> 報酬 = R(状態, 行動)

## 評価指標

```
精度 (Accuracy) = 正解数 / 全体数
適合率 (Precision) = TP / (TP + FP)
再現率 (Recall) = TP / (TP + FN)
F1スコア = 2 × (適合率 × 再現率) / (適合率 + 再現率)
```

## 参考文献

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*
2. Goodfellow, I. et al. (2016). *Deep Learning*
""",
        "path": "synthetic/ml_overview_ja.md",
        "repo": "synthetic-ja",
        "language": "ja",
    })

    return docs


def collect_all(output_dir: str, max_per_repo: int = 15, github_token: str | None = None) -> list[dict]:
    """Collect markdown from all sources and save to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_docs = []

    # English repos
    print("Collecting English markdown from GitHub...")
    for repo in tqdm(ENGLISH_REPOS, desc="EN repos"):
        docs = fetch_github_markdown_files(repo, max_files=max_per_repo, token=github_token)
        all_docs.extend(docs)
        print(f"  {repo}: {len(docs)} files")

    # Japanese repos
    print("\nCollecting Japanese markdown from GitHub...")
    for repo in tqdm(JAPANESE_REPOS, desc="JA repos"):
        docs = fetch_github_markdown_files(repo, max_files=max_per_repo, token=github_token)
        for doc in docs:
            doc["language"] = "ja"
        all_docs.extend(docs)
        print(f"  {repo}: {len(docs)} files")

    # Japanese Wikipedia
    print("\nCollecting Japanese Wikipedia articles...")
    wiki_docs = fetch_japanese_wiki_markdown()
    all_docs.extend(wiki_docs)
    print(f"  Wikipedia JA: {len(wiki_docs)} articles")

    # Synthetic Japanese
    print("\nCreating synthetic Japanese documents...")
    synth_docs = create_synthetic_japanese_markdown()
    all_docs.extend(synth_docs)
    print(f"  Synthetic JA: {len(synth_docs)} documents")

    # Save individual files
    for i, doc in enumerate(all_docs):
        lang = doc["language"]
        safe_name = re.sub(r"[^\w\-.]", "_", doc["name"])
        filepath = output_path / lang / f"{i:04d}_{safe_name}"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(doc["content"], encoding="utf-8")

    # Save metadata
    metadata = [{k: v for k, v in doc.items() if k != "content"} for doc in all_docs]
    meta_path = output_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    en_count = sum(1 for d in all_docs if d["language"] == "en")
    ja_count = sum(1 for d in all_docs if d["language"] == "ja")
    print(f"\nTotal collected: {len(all_docs)} (EN: {en_count}, JA: {ja_count})")

    return all_docs


if __name__ == "__main__":
    collect_all("data/raw/markdown")
