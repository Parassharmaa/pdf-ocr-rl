"""Collect markdown files from open sources for dataset creation.

Sources:
  - GitHub repos (EN + JA): structured markdown docs
  - GitHub LaTeX repos: .tex files converted to markdown
  - Wikipedia (EN + JA): articles converted from HTML
  - arXiv: LaTeX papers converted to markdown via pandoc
  - HuggingFace olmOCR-mix: ready-made PDF page images + markdown (optional)
"""

import gzip
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

GITHUB_API = "https://api.github.com"
ARXIV_API = "http://export.arxiv.org/api/query"

# Curated list of repos with high-quality, diverse markdown docs
ENGLISH_REPOS = [
    # Frameworks & libraries (rich docs with tables, code, lists)
    "facebook/react",
    "microsoft/TypeScript",
    "vuejs/docs",
    "sveltejs/svelte",
    "angular/angular",
    "vercel/next.js",
    "remix-run/remix",
    "astro-build/astro",
    # Languages & runtimes
    "golang/go",
    "rust-lang/rust",
    "python/cpython",
    "nodejs/node",
    "denoland/deno",
    "zig-lang/zig",
    # Infrastructure & DevOps
    "kubernetes/kubernetes",
    "docker/docs",
    "hashicorp/terraform",
    "ansible/ansible",
    # Data & ML
    "pytorch/pytorch",
    "huggingface/transformers",
    "langchain-ai/langchain",
    "apache/spark",
    # Databases
    "redis/redis",
    "postgres/postgres",
    "supabase/supabase",
    # Tools & utilities
    "neovim/neovim",
    "git/git",
    "curl/curl",
    # Documentation-heavy repos (many structured markdown files)
    "github/docs",
    "mdn/content",
    "tldr-pages/tldr",
    "sindresorhus/awesome",
    "papers-we-love/papers-we-love",
]

JAPANESE_REPOS = [
    # Japanese translations of popular projects
    "vuejs-translations/docs-ja",
    "electron/i18n",
    "willnet/rspec-style-guide",
    # Japanese developer guides
    "avelino/awesome-go",
    "yohamta/donern",
    # Japanese tech community repos with markdown
    "hatena/hatena-textbook",
    "progit/progit2-ja",
    "azu/JavaScript-Plugin-Architecture",
    "shu223/iOS-Depth-Sampler",
    "nicklockwood/iVersion",
    # Japanese documentation projects
    "japanese-document/react.dev.ja",
    "maku77/maku77.github.io",
]

# Wikipedia articles - diverse topics for varied content structure
ENGLISH_WIKI_PAGES = [
    "Python_(programming_language)",
    "Machine_learning",
    "Artificial_intelligence",
    "World_Wide_Web",
    "Linux",
    "Internet_protocol_suite",
    "Quantum_computing",
    "Climate_change",
    "Solar_System",
    "Human_genome",
    "Renaissance",
    "Algorithm",
    "Database",
    "Cryptography",
    "Computer_vision",
    "Natural_language_processing",
    "Blockchain",
    "Cloud_computing",
    "Robotics",
    "Semiconductor",
]

JAPANESE_WIKI_PAGES = [
    "Python",
    "人工知能",
    "機械学習",
    "深層学習",
    "自然言語処理",
    "コンピュータ",
    "インターネット",
    "データベース",
    "オペレーティングシステム",
    "暗号",
    "ロボット工学",
    "量子コンピュータ",
    "ブロックチェーン",
    "クラウドコンピューティング",
    "半導体",
    "太陽系",
    "気候変動",
    "ルネサンス",
    "アルゴリズム",
    "コンピュータビジョン",
    "ソフトウェア工学",
    "線形代数学",
    "確率論",
    "統計学",
    "情報理論",
]

# arXiv categories to sample from (diverse content with tables, equations, code)
ARXIV_CATEGORIES = [
    "cs.CL",   # Computational Linguistics / NLP
    "cs.CV",   # Computer Vision
    "cs.LG",   # Machine Learning
    "cs.AI",   # Artificial Intelligence
    "cs.SE",   # Software Engineering
    "cs.DB",   # Databases
    "cs.CR",   # Cryptography
    "stat.ML", # Statistics - Machine Learning
    "math.OC", # Optimization and Control
    "physics.comp-ph",  # Computational Physics
]

# GitHub repos with LaTeX documents (papers, textbooks, lecture notes)
LATEX_REPOS = [
    "ElegantLaTeX/ElegantPaper",
    "ElegantLaTeX/ElegantBook",
    "ElegantLaTeX/ElegantNote",
    "sb2nov/resume",
    "posquit0/Awesome-CV",
    "hmemcpy/milern-latex",
    "tuhdo/os01",
    "dendibakh/perf-book",
    "aaronbloomfield/pdr",
    "liuxinyu95/AlgoXY",
    "HarisIqbal88/PlotNeuralNet",
    "xinychen/latex-cookbook",
    "vdumoulin/conv_arithmetic",
    "synercys/annotated_latex_equations",
    "terryum/awesome-deep-learning-papers",
]


def fetch_github_markdown_files(repo: str, max_files: int = 20, token: str | None = None) -> list[dict]:
    """Fetch markdown files from a GitHub repo via API."""
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

    # Prioritize files likely to have rich content
    def _score(f):
        p = f["path"].lower()
        score = 0
        if "readme" in p:
            score += 3
        if "guide" in p or "tutorial" in p or "doc" in p:
            score += 2
        if "contributing" in p or "changelog" in p:
            score += 1
        if "api" in p or "reference" in p:
            score += 2
        score -= p.count("/") * 0.3
        return -score

    md_files.sort(key=_score)
    md_files = md_files[:max_files]

    for f in md_files:
        try:
            raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{f['path']}"
            resp = requests.get(raw_url, timeout=15)
            resp.raise_for_status()
            content = resp.text

            if len(content) < 200:
                continue
            if not any(marker in content for marker in ["#", "|", "```", "- ", "1. "]):
                continue
            if len(content) > 50000:
                content = content[:50000]

            results.append({
                "name": os.path.basename(f["path"]),
                "content": content,
                "path": f["path"],
                "repo": repo,
                "language": "en",
            })
        except requests.RequestException:
            continue

        time.sleep(0.2)

    return results


def _wiki_html_to_markdown(html: str) -> str:
    """Convert Wikipedia HTML to markdown-like text."""
    text = html
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\d+\]", "", text)
    # Tables
    text = re.sub(r"<table[^>]*>", "\n", text, flags=re.DOTALL)
    text = re.sub(r"</table>", "\n", text)
    text = re.sub(r"<tr[^>]*>", "", text)
    text = re.sub(r"</tr>", "\n", text)
    text = re.sub(r"<th[^>]*>(.*?)</th>", r"| **\1** ", text, flags=re.DOTALL)
    text = re.sub(r"<td[^>]*>(.*?)</td>", r"| \1 ", text, flags=re.DOTALL)
    # Headings
    for i in range(6, 0, -1):
        text = re.sub(rf"<h{i}[^>]*>(.*?)</h{i}>", rf"{'#' * i} \1\n\n", text, flags=re.DOTALL)
    # Code
    text = re.sub(r"<pre[^>]*><code[^>]*>(.*?)</code></pre>", r"```\n\1\n```\n", text, flags=re.DOTALL)
    text = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
    # Paragraphs, lists
    text = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n\n", text, flags=re.DOTALL)
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", text, flags=re.DOTALL)
    # Bold/italic
    text = re.sub(r"<b>(.*?)</b>", r"**\1**", text)
    text = re.sub(r"<strong>(.*?)</strong>", r"**\1**", text)
    text = re.sub(r"<i>(.*?)</i>", r"*\1*", text)
    text = re.sub(r"<em>(.*?)</em>", r"*\1*", text)
    # Blockquotes
    text = re.sub(r"<blockquote[^>]*>(.*?)</blockquote>", r"> \1\n", text, flags=re.DOTALL)
    # Strip remaining HTML
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def fetch_wiki_markdown(pages: list[str], language: str = "en") -> list[dict]:
    """Fetch Wikipedia articles and convert HTML to markdown-like text."""
    results = []
    lang_code = "ja" if language == "ja" else "en"

    for title in tqdm(pages, desc=f"Wikipedia {lang_code.upper()}"):
        url = f"https://{lang_code}.wikipedia.org/api/rest_v1/page/html/{title}"
        try:
            resp = requests.get(url, timeout=15, headers={
                "Accept": "text/html",
                "User-Agent": "pdf-ocr-rl/1.0 (research project; https://github.com/Parassharmaa/pdf-ocr-rl)",
            })
            resp.raise_for_status()
            text = _wiki_html_to_markdown(resp.text)

            if len(text) > 200:
                if len(text) > 30000:
                    text = text[:30000]
                results.append({
                    "name": f"{title}.md",
                    "content": text,
                    "path": url,
                    "repo": f"wikipedia-{lang_code}",
                    "language": language,
                })
        except requests.RequestException:
            continue

        time.sleep(0.5)

    return results


# ---------------------------------------------------------------------------
# arXiv LaTeX → Markdown
# ---------------------------------------------------------------------------

def _latex_to_markdown(tex_content: str) -> str | None:
    """Convert LaTeX to markdown using pandoc."""
    try:
        result = subprocess.run(
            ["pandoc", "-f", "latex", "-t", "markdown", "--wrap=none"],
            input=tex_content,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and len(result.stdout.strip()) > 100:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _find_main_tex(files: list[str], contents: dict[str, str]) -> str | None:
    """Find the main .tex file in an arXiv source bundle."""
    tex_files = [f for f in files if f.endswith(".tex")]
    if not tex_files:
        return None
    if len(tex_files) == 1:
        return tex_files[0]

    # Heuristic: file with \documentclass is likely the main file
    for tf in tex_files:
        if tf in contents and r"\documentclass" in contents[tf]:
            return tf

    # Fallback: shortest name, or "main.tex", or "paper.tex"
    for name in ["main.tex", "paper.tex", "article.tex"]:
        if name in tex_files:
            return name
    return min(tex_files, key=len)


def fetch_arxiv_latex(
    categories: list[str] | None = None,
    max_papers: int = 50,
) -> list[dict]:
    """Fetch arXiv papers and convert LaTeX source to markdown via pandoc."""
    if categories is None:
        categories = ARXIV_CATEGORIES

    results = []
    papers_per_cat = max(2, max_papers // len(categories))

    for cat in tqdm(categories, desc="arXiv categories"):
        query = f"cat:{cat}"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": papers_per_cat,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            resp = requests.get(ARXIV_API, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException:
            print(f"  Failed to query arXiv for {cat}")
            continue

        # Parse Atom XML for paper IDs
        entries = re.findall(r"<id>http://arxiv\.org/abs/([^<]+)</id>", resp.text)
        if not entries:
            continue

        for arxiv_id in entries:
            if len(results) >= max_papers:
                break

            # Extract title from feed
            title_match = re.search(
                rf"<id>http://arxiv\.org/abs/{re.escape(arxiv_id)}</id>.*?<title>([^<]+)</title>",
                resp.text,
                re.DOTALL,
            )
            title = title_match.group(1).strip() if title_match else arxiv_id

            # Download source tarball
            source_url = f"https://arxiv.org/e-print/{arxiv_id}"
            try:
                src_resp = requests.get(source_url, timeout=30, headers={"Accept": "*/*"})
                src_resp.raise_for_status()
            except requests.RequestException:
                continue

            md_content = _extract_latex_from_response(src_resp.content)
            if md_content and len(md_content) > 300:
                if len(md_content) > 50000:
                    md_content = md_content[:50000]
                results.append({
                    "name": f"{arxiv_id.replace('/', '_')}.md",
                    "content": md_content,
                    "path": f"https://arxiv.org/abs/{arxiv_id}",
                    "repo": f"arxiv-{cat}",
                    "language": "en",
                })

            time.sleep(3)  # Be respectful to arXiv servers

        if len(results) >= max_papers:
            break

    return results


def _extract_latex_from_response(content: bytes) -> str | None:
    """Extract and convert LaTeX from an arXiv source response using pandoc."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # arXiv source can be: tar.gz, gz (single file), or raw tex
        tmp_path = Path(tmpdir)

        # Try as tar.gz first
        try:
            tar_path = tmp_path / "source.tar.gz"
            tar_path.write_bytes(content)
            with tarfile.open(tar_path, "r:*") as tar:
                tar.extractall(tmpdir, filter="data")

            # Find all extracted files
            all_files = []
            contents = {}
            for p in tmp_path.rglob("*"):
                if p.is_file() and p.suffix == ".tex":
                    rel = str(p.relative_to(tmp_path))
                    all_files.append(rel)
                    try:
                        contents[rel] = p.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        pass

            main_tex = _find_main_tex(all_files, contents)
            if main_tex and main_tex in contents:
                return _latex_to_markdown(contents[main_tex])
            return None
        except (tarfile.TarError, Exception):
            pass

        # Try as gzipped single file
        try:
            tex_content = gzip.decompress(content).decode("utf-8", errors="ignore")
            if r"\begin{document}" in tex_content or r"\section" in tex_content:
                return _latex_to_markdown(tex_content)
        except (gzip.BadGzipFile, Exception):
            pass

        # Try as raw tex
        try:
            tex_content = content.decode("utf-8", errors="ignore")
            if r"\begin{document}" in tex_content or r"\section" in tex_content:
                return _latex_to_markdown(tex_content)
        except Exception:
            pass

    return None


def fetch_github_latex_files(
    repos: list[str] | None = None,
    max_per_repo: int = 10,
    token: str | None = None,
) -> list[dict]:
    """Fetch .tex files from GitHub repos and convert to markdown via pandoc."""
    if repos is None:
        repos = LATEX_REPOS

    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    results = []

    for repo in tqdm(repos, desc="LaTeX repos"):
        url = f"{GITHUB_API}/repos/{repo}/git/trees/HEAD?recursive=1"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            tree = resp.json().get("tree", [])
        except (requests.RequestException, json.JSONDecodeError):
            print(f"  Failed to fetch tree for {repo}")
            continue

        tex_files = [f for f in tree if f["path"].endswith(".tex") and f["type"] == "blob"]
        # Prioritize main files
        for name in ["main.tex", "paper.tex", "article.tex", "thesis.tex", "book.tex"]:
            for f in tex_files:
                if f["path"].endswith(name):
                    tex_files.remove(f)
                    tex_files.insert(0, f)
                    break
        tex_files = tex_files[:max_per_repo]

        fetched = 0
        for f in tex_files:
            try:
                raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{f['path']}"
                resp = requests.get(raw_url, timeout=15)
                resp.raise_for_status()
                tex_content = resp.text

                if len(tex_content) < 300:
                    continue
                # Must look like actual LaTeX
                if not any(cmd in tex_content for cmd in [r"\section", r"\begin", r"\chapter", r"\title"]):
                    continue

                md_content = _latex_to_markdown(tex_content)
                if md_content and len(md_content) > 200:
                    if len(md_content) > 50000:
                        md_content = md_content[:50000]
                    results.append({
                        "name": f"{os.path.basename(f['path']).replace('.tex', '.md')}",
                        "content": md_content,
                        "path": f["path"],
                        "repo": f"latex-{repo}",
                        "language": "en",
                    })
                    fetched += 1
            except requests.RequestException:
                continue
            time.sleep(0.3)

        print(f"  {repo}: {fetched} tex files converted")

    return results


def fetch_olmocr_pairs(max_samples: int = 200, output_dir: str | None = None) -> list[dict]:
    """Download ready-made PDF-markdown pairs from olmOCR-mix on HuggingFace.

    These come pre-paired (image + markdown) so they bypass the render step.
    Returns image paths + markdown content saved to output_dir.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  HuggingFace datasets not installed, skipping olmOCR")
        return []

    if not output_dir:
        return []

    out_path = Path(output_dir)
    olmocr_dir = out_path / "olmocr"
    olmocr_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    try:
        # Load a small streaming slice — olmOCR-mix is very large
        ds = load_dataset(
            "allenai/olmOCR-mix-0225",
            split="train",
            streaming=True,
        )

        print(f"  Streaming up to {max_samples} samples from olmOCR-mix...")
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break

            # olmOCR-mix has 'image' (PIL) and 'text' (markdown) columns
            image = sample.get("image")
            text = sample.get("text", "")

            if not image or len(text) < 100:
                continue

            # Save image
            img_path = olmocr_dir / f"olmocr_{i:04d}.png"
            image.save(str(img_path))

            pairs.append({
                "image_path": str(img_path),
                "markdown": text,
                "language": "en",
                "source": "olmOCR-mix-0225",
                "page_start_char": 0,
                "page_end_char": len(text),
                "page_index": 0,
                "page_count": 1,
            })

        print(f"  olmOCR: {len(pairs)} pre-made pairs saved")
    except Exception as e:
        print(f"  Failed to load olmOCR dataset: {e}")

    return pairs


def collect_all(
    output_dir: str,
    max_per_repo: int = 15,
    max_arxiv: int = 100,
    github_token: str | None = None,
) -> list[dict]:
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

    # English Wikipedia
    print("\nCollecting English Wikipedia articles...")
    en_wiki_docs = fetch_wiki_markdown(ENGLISH_WIKI_PAGES, language="en")
    all_docs.extend(en_wiki_docs)
    print(f"  Wikipedia EN: {len(en_wiki_docs)} articles")

    # Japanese Wikipedia
    print("\nCollecting Japanese Wikipedia articles...")
    ja_wiki_docs = fetch_wiki_markdown(JAPANESE_WIKI_PAGES, language="ja")
    all_docs.extend(ja_wiki_docs)
    print(f"  Wikipedia JA: {len(ja_wiki_docs)} articles")

    # GitHub LaTeX repos
    print("\nCollecting LaTeX files from GitHub...")
    latex_docs = fetch_github_latex_files(token=github_token)
    all_docs.extend(latex_docs)
    print(f"  GitHub LaTeX: {len(latex_docs)} files")

    # arXiv LaTeX papers
    print("\nCollecting arXiv LaTeX papers...")
    arxiv_docs = fetch_arxiv_latex(max_papers=max_arxiv)
    all_docs.extend(arxiv_docs)
    print(f"  arXiv: {len(arxiv_docs)} papers")

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
