"""Render markdown files to PDF images for training pairs."""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm

# HTML template for rendering markdown to PDF
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<style>
body {{
    font-family: {font_family};
    font-size: {font_size}px;
    line-height: 1.6;
    max-width: 210mm;
    margin: {margin};
    padding: 20px 30px;
    color: #333;
}}
h1 {{ font-size: 1.8em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
h2 {{ font-size: 1.4em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
h3 {{ font-size: 1.2em; }}
code {{
    background: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 0.9em;
}}
pre {{
    background: #f4f4f4;
    padding: 12px;
    border-radius: 5px;
    overflow-x: auto;
    border: 1px solid #ddd;
}}
pre code {{ background: none; padding: 0; }}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}}
th {{ background: #f0f0f0; font-weight: bold; }}
blockquote {{
    border-left: 4px solid #ccc;
    margin: 1em 0;
    padding: 0.5em 1em;
    color: #666;
}}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
{content}
</body>
</html>"""


def markdown_to_html(md_content: str) -> str:
    """Convert markdown to HTML using Python's markdown library."""
    try:
        import markdown

        extensions = ["tables", "fenced_code", "codehilite", "toc"]
        html = markdown.markdown(md_content, extensions=extensions)
        return html
    except ImportError:
        # Fallback: basic conversion
        return _basic_md_to_html(md_content)


def _basic_md_to_html(md: str) -> str:
    """Basic markdown to HTML without external deps."""
    lines = md.split("\n")
    html_lines = []
    in_code = False
    in_table = False

    for line in lines:
        # Code blocks
        if line.strip().startswith("```"):
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                lang = line.strip()[3:]
                html_lines.append(f"<pre><code class='language-{lang}'>")
                in_code = True
            continue

        if in_code:
            html_lines.append(line)
            continue

        # Headings
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            html_lines.append(f"<h{level}>{text}</h{level}>")
            continue

        # Tables
        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue  # separator row
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            tag = "th" if not in_table or html_lines[-1] == "<table>" else "td"
            row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")
            continue
        elif in_table:
            html_lines.append("</table>")
            in_table = False

        # Bold/italic
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
        line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)

        # Lists
        if re.match(r"^[-*]\s", line):
            html_lines.append(f"<li>{line[2:]}</li>")
            continue
        if re.match(r"^\d+\.\s", line):
            text = re.sub(r"^\d+\.\s", "", line)
            html_lines.append(f"<li>{text}</li>")
            continue

        # Blockquotes
        if line.startswith(">"):
            html_lines.append(f"<blockquote>{line[1:].strip()}</blockquote>")
            continue

        # Paragraphs
        if line.strip():
            html_lines.append(f"<p>{line}</p>")
        else:
            html_lines.append("")

    if in_table:
        html_lines.append("</table>")

    return "\n".join(html_lines)


def render_html_to_pdf(html_content: str, output_path: str) -> bool:
    """Render HTML to PDF using weasyprint or wkhtmltopdf."""
    try:
        from weasyprint import HTML

        HTML(string=html_content).write_pdf(output_path)
        return True
    except ImportError:
        pass

    # Fallback to wkhtmltopdf if available
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False, encoding="utf-8") as f:
            f.write(html_content)
            tmp_html = f.name

        result = subprocess.run(
            ["wkhtmltopdf", "--quiet", "--page-size", "A4", tmp_html, output_path],
            capture_output=True,
            timeout=30,
        )
        os.unlink(tmp_html)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("Warning: No PDF renderer available (install weasyprint or wkhtmltopdf)")
    return False


def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 150) -> list[str]:
    """Convert PDF pages to images."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    images = []

    try:
        import fitz  # pymupdf

        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_path = str(output / f"page_{page_num:03d}.png")
            pix.save(img_path)
            images.append(img_path)
        doc.close()
    except ImportError:
        try:
            from pdf2image import convert_from_path

            pil_images = convert_from_path(pdf_path, dpi=dpi)
            for i, img in enumerate(pil_images):
                img_path = str(output / f"page_{i:03d}.png")
                img.save(img_path)
                images.append(img_path)
        except ImportError:
            print("Warning: No PDF-to-image converter available (install pymupdf or pdf2image)")

    return images


def render_dataset(
    markdown_dir: str,
    output_dir: str,
    font_sizes: list[int] | None = None,
    dpi: int = 150,
) -> list[dict]:
    """Render all markdown files to PDF images and create paired dataset.

    Returns list of dicts with 'image_path', 'markdown', 'language', 'source'.
    """
    if font_sizes is None:
        font_sizes = [11, 12, 14]

    md_dir = Path(markdown_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []

    # Find all markdown files
    md_files = list(md_dir.rglob("*.md"))
    if not md_files:
        print(f"No markdown files found in {markdown_dir}")
        return pairs

    print(f"Rendering {len(md_files)} markdown files to PDF images...")

    for md_file in tqdm(md_files, desc="Rendering"):
        md_content = md_file.read_text(encoding="utf-8")

        # Detect language from path
        lang = "ja" if "/ja/" in str(md_file) else "en"
        font_family = (
            "'Hiragino Kaku Gothic Pro', 'Noto Sans CJK JP', sans-serif"
            if lang == "ja"
            else "'Georgia', 'Times New Roman', serif"
        )

        # Pick a random font size for variety
        font_size = font_sizes[hash(str(md_file)) % len(font_sizes)]

        # Convert markdown to HTML
        html_body = markdown_to_html(md_content)
        full_html = HTML_TEMPLATE.format(
            lang=lang,
            font_family=font_family,
            font_size=font_size,
            margin="20mm 15mm" if hash(str(md_file)) % 2 == 0 else "25mm 20mm",
            content=html_body,
        )

        # Render to PDF
        stem = md_file.stem
        pdf_path = str(out_dir / f"{stem}.pdf")
        if not render_html_to_pdf(full_html, pdf_path):
            continue

        # Convert PDF to images
        img_dir = str(out_dir / "images" / stem)
        images = pdf_to_images(pdf_path, img_dir, dpi=dpi)

        if images:
            # For single-page docs, pair the whole markdown with the image
            # For multi-page, we still use the full markdown (simplified approach)
            for img_path in images:
                pairs.append({
                    "image_path": img_path,
                    "markdown": md_content,
                    "language": lang,
                    "source": str(md_file),
                    "font_size": font_size,
                })

        # Clean up PDF file to save space
        try:
            os.unlink(pdf_path)
        except OSError:
            pass

    # Save dataset metadata
    meta = [{k: v for k, v in p.items() if k != "markdown"} for p in pairs]
    meta_path = out_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    en_count = sum(1 for p in pairs if p["language"] == "en")
    ja_count = sum(1 for p in pairs if p["language"] == "ja")
    print(f"\nDataset: {len(pairs)} image-markdown pairs (EN: {en_count}, JA: {ja_count})")

    return pairs


if __name__ == "__main__":
    render_dataset("data/raw/markdown", "data/processed")
