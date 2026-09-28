#!/usr/bin/env python3
"""Audit tracked Markdown links; optionally probe external URLs conservatively.

Usage:
    python3 scripts/check_links.py
    python3 scripts/check_links.py --external README.md

The default offline audit checks local paths, heading anchors, URL hostnames,
and arXiv ID syntax. External probing is manual and rate-limited; anti-bot,
transient, and suspiciously small HTML responses are reported as unverified.
It does not verify paper titles, authors, or venue claims.
"""

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
WEB_URL = re.compile(r"https?://[^\s<>)]*")
ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?(?:\.pdf)?")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int
    kind: str
    target: str
    detail: str


def heading_slug(title: str) -> str:
    """Approximate GitHub's heading slugs for this repository's headings."""
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", title)
    title = re.sub(r"[^\w\s-]", "", title.lower())
    return re.sub(r"\s+", "-", title.strip())


def anchors_for(path: Path) -> set:
    headings = set()
    counts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING.match(line)
        if match:
            slug = heading_slug(match.group(1))
            index = counts.get(slug, 0)
            counts[slug] = index + 1
            headings.add(slug if index == 0 else f"{slug}-{index}")
    return headings


def external_result(url: str) -> Optional[Tuple[str, str]]:
    """Return an issue classification, or None only when the response looks usable."""
    host = (urlsplit(url).hostname or "").lower()
    if host == "openreview.net" or host.endswith(".openreview.net"):
        return "unverified", "OpenReview blocks or challenges automated clients; review manually"
    request = Request(url, headers={"User-Agent": "SSM-Link-Audit/1.0", "Accept": "*/*"})
    try:
        with urlopen(request, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                sample = response.read(257)
                if len(sample) < 256:
                    return "unverified", "very small HTML response (possible soft 404)"
    except HTTPError as error:
        if error.code in (404, 410):
            return "broken", f"HTTP {error.code}"
        return "unverified", f"HTTP {error.code} (may be access control or a transient error)"
    except (URLError, TimeoutError, ValueError) as error:
        return "unverified", str(error)
    return None


def audit_files(files: Sequence[Path], external: bool = False, delay: float = 1.0) -> Tuple[List[Issue], int]:
    issues = []
    total = 0
    external_cache = {}
    last_request = None
    anchor_cache = {}
    for path in files:
        path = Path(path).resolve()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            urls = [match.group().rstrip(".,;") for match in WEB_URL.finditer(line)]
            for url in urls:
                total += 1
                try:
                    parsed = urlsplit(url)
                    host = parsed.hostname
                except ValueError:
                    host = None
                if not host:
                    issues.append(Issue(path, line_number, "invalid-url", url, "missing or malformed hostname"))
                    continue
                if host in ("arxiv.org", "www.arxiv.org") and parsed.path.startswith(("/abs/", "/pdf/")):
                    identifier = parsed.path.split("/", 2)[2]
                    if not ARXIV_ID.fullmatch(identifier):
                        issues.append(Issue(path, line_number, "invalid-arxiv-id", url, "unexpected arXiv ID format"))
                        continue
                if external:
                    if url not in external_cache:
                        if last_request is not None:
                            time.sleep(max(0.0, delay - (time.monotonic() - last_request)))
                        last_request = time.monotonic()
                        external_cache[url] = external_result(url)
                    result = external_cache[url]
                    if result:
                        issues.append(Issue(path, line_number, result[0], url, result[1]))
            for match in MARKDOWN_LINK.finditer(line):
                target = match.group(1).strip()
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                total += 1
                target_path, separator, fragment = target.partition("#")
                destination = (path.parent / unquote(target_path)).resolve() if target_path else path
                if not destination.is_file():
                    issues.append(Issue(path, line_number, "missing-file", target, "local target does not exist"))
                elif separator and destination.suffix.lower() == ".md":
                    if destination not in anchor_cache:
                        anchor_cache[destination] = anchors_for(destination)
                    if unquote(fragment).lower() not in anchor_cache[destination]:
                        issues.append(Issue(path, line_number, "missing-anchor", target, "heading not found"))
    return issues, total


def tracked_markdown() -> List[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [ROOT / name for name in result.stdout.splitlines()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="Markdown files (default: tracked Markdown files)")
    parser.add_argument("--external", action="store_true", help="probe external URLs (slow; ambiguous results stay unverified)")
    parser.add_argument("--delay", type=float, default=1.0, help="minimum seconds between external probes (default: 1)")
    args = parser.parse_args(argv)
    if args.delay < 0:
        parser.error("--delay must be nonnegative")
    files = args.files or tracked_markdown()
    issues, total = audit_files(files, external=args.external, delay=args.delay)
    for issue in issues:
        try:
            display_path = issue.path.relative_to(ROOT)
        except ValueError:
            display_path = issue.path
        print(f"{display_path}:{issue.line}: {issue.kind}: {issue.target} ({issue.detail})")
    broken = sum(issue.kind != "unverified" for issue in issues)
    unverified = sum(issue.kind == "unverified" for issue in issues)
    print(f"Checked {total} links; {broken} actionable issue(s), {unverified} unverified.")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
