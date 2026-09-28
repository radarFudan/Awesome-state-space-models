from io import BytesIO
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import check_links


class LinkAuditTests(unittest.TestCase):
    def test_local_anchor_and_file_are_checked_from_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "paper.pdf").write_bytes(b"%PDF-1.4")
            page = root / "index.md"
            page.write_text(
                "# Paper List\n[here](#paper-list) [pdf](paper.pdf)\n"
                "[missing](#not-here) [lost](missing.pdf)\n",
                encoding="utf-8",
            )

            issues, _ = check_links.audit_files([page], external=False, delay=0)

            self.assertEqual(
                [(issue.line, issue.kind) for issue in issues],
                [(3, "missing-anchor"), (3, "missing-file")],
            )

    def test_malformed_arxiv_id_is_reported_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "papers.md"
            page.write_text(
                "One (https://arxiv.org/abs/2501.12345)\n"
                "Two [Arxiv](https://arxiv.org/abs/2501.abcde)\n",
                encoding="utf-8",
            )

            issues, _ = check_links.audit_files([page], external=False, delay=0)

            self.assertEqual([(issue.line, issue.kind) for issue in issues], [(2, "invalid-arxiv-id")])

    def test_malformed_hostname_is_reported_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "papers.md"
            page.write_text("[bad](https://[invalid)\n", encoding="utf-8")

            issues, _ = check_links.audit_files([page], external=False, delay=0)

            self.assertEqual([(issue.line, issue.kind) for issue in issues], [(1, "invalid-url")])

    def test_openreview_is_not_probed_and_is_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "papers.md"
            page.write_text("Paper (https://openreview.net/forum?id=abc)\n", encoding="utf-8")
            with patch.object(check_links, "urlopen", side_effect=AssertionError("unexpected network call")):
                issues, _ = check_links.audit_files([page], external=True, delay=0)

            self.assertEqual([(issue.line, issue.kind) for issue in issues], [(1, "unverified")])

    def test_external_statuses_separate_broken_from_unverified(self):
        def fake_open(request, timeout):
            self.assertEqual(timeout, 10)
            url = request.full_url
            if url.endswith("/gone"):
                raise HTTPError(url, 404, "Not Found", {}, None)
            if url.endswith("/blocked"):
                raise HTTPError(url, 403, "Forbidden", {}, None)
            response = BytesIO(b"<html>short</html>")
            response.headers = {"Content-Type": "text/html"}
            return response

        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "papers.md"
            page.write_text(
                "Gone (https://example.org/gone)\n"
                "Blocked (https://example.org/blocked)\n"
                "Tiny (https://example.org/tiny)\n",
                encoding="utf-8",
            )
            with patch.object(check_links, "urlopen", side_effect=fake_open):
                issues, _ = check_links.audit_files([page], external=True, delay=0)
            self.assertEqual(
                [(issue.line, issue.kind) for issue in issues],
                [(1, "broken"), (2, "unverified"), (3, "unverified")],
            )

    def test_cli_fails_for_broken_local_link(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "papers.md"
            page.write_text("[missing](lost.pdf)\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "check_links.py"), str(page)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("missing-file", result.stdout)


if __name__ == "__main__":
    unittest.main()
