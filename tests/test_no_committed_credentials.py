import re
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ASSIGNMENT = re.compile(
    r"^\s*CDSE_(?:USERNAME|PASSWORD)\d*\s*=\s*(['\"])[^'\"]+\1"
)
YAML_VALUE = re.compile(
    r"^\s*cdse_(?:username|password)\d*\s*:\s*[^\s#]+",
    re.IGNORECASE,
)


class CommittedCredentialTests(unittest.TestCase):
    def test_no_hardcoded_cdse_credentials_in_tracked_files(self):
        result = subprocess.run(
            ["git", "ls-files", "*.py", "*.yaml", "*.yml"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        findings = []
        for relative_path in result.stdout.splitlines():
            path = REPO_ROOT / relative_path
            if not path.is_file():
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                if line.lstrip().startswith("#"):
                    continue
                if PYTHON_ASSIGNMENT.match(line) or YAML_VALUE.match(line):
                    findings.append(f"{relative_path}:{line_number}")

        self.assertEqual(
            findings,
            [],
            "Hard-coded CDSE credentials found in tracked files: "
            + ", ".join(findings),
        )


if __name__ == "__main__":
    unittest.main()
