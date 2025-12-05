#!/usr/bin/env python3
"""
Download COSMIC Cancer Gene Census data.

This script downloads the Cancer Gene Census from COSMIC (Catalogue of
Somatic Mutations in Cancer) for use with the gene function polarity rule.

The Cancer Gene Census contains ~764 genes with validated roles in cancer,
classified as oncogenes, tumor suppressor genes (TSG), or both.

Usage:
    # With environment variables (recommended):
    COSMIC_EMAIL=your@email.com COSMIC_PASSWORD=yourpassword python scripts/download_cosmic_cgc.py

    # With command-line arguments:
    python scripts/download_cosmic_cgc.py --email your@email.com --password yourpassword

    # Specify COSMIC version (default: v103):
    python scripts/download_cosmic_cgc.py --version v103

Requirements:
    - COSMIC account (register at https://cancer.sanger.ac.uk/cosmic/register)
    - Academic use is free; commercial use requires license

Output:
    - data/cosmic/Cosmic_CancerGeneCensus_v{VERSION}_GRCh38.tsv
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# Default COSMIC version
DEFAULT_VERSION = "v103"

# API endpoint for scripted downloads
COSMIC_API_URL = "https://cancer.sanger.ac.uk/api/mono/products/v1/downloads/scripted"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "cosmic"


def get_credentials(args: argparse.Namespace) -> tuple[str, str]:
    """Get COSMIC credentials from args or environment."""
    email = args.email or os.environ.get("COSMIC_EMAIL")
    password = args.password or os.environ.get("COSMIC_PASSWORD")

    if not email or not password:
        print("Error: COSMIC credentials required.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Set environment variables:", file=sys.stderr)
        print("  export COSMIC_EMAIL=your@email.com", file=sys.stderr)
        print("  export COSMIC_PASSWORD=yourpassword", file=sys.stderr)
        print("", file=sys.stderr)
        print("Or use command-line arguments:", file=sys.stderr)
        print("  --email your@email.com --password yourpassword", file=sys.stderr)
        print("", file=sys.stderr)
        print("Register at: https://cancer.sanger.ac.uk/cosmic/register", file=sys.stderr)
        sys.exit(1)

    return email, password


def get_auth_string(email: str, password: str) -> str:
    """Generate Base64 authentication string."""
    credentials = f"{email}:{password}"
    return base64.b64encode(credentials.encode()).decode()


def get_download_url(auth_string: str, version: str) -> str:
    """Request temporary download URL from COSMIC API."""
    # Build the file path for Cancer Gene Census
    file_path = f"grch38/cosmic/{version}/Cosmic_CancerGeneCensus_Tsv_{version}_GRCh38.tar"

    url = f"{COSMIC_API_URL}?path={file_path}&bucket=downloads"

    request = Request(
        url,
        headers={
            "Authorization": f"Basic {auth_string}",
            "Accept": "application/json",
        },
    )

    print(f"Requesting download URL for COSMIC {version}...")

    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode())
            download_url = data.get("url")
            if not download_url:
                print(f"Error: No download URL in response: {data}", file=sys.stderr)
                sys.exit(1)
            return download_url
    except HTTPError as e:
        if e.code == 401:
            print("Error: Authentication failed. Check your COSMIC credentials.", file=sys.stderr)
        elif e.code == 403:
            print(
                "Error: Access forbidden. Your account may not have download access.",
                file=sys.stderr,
            )
        elif e.code == 404:
            print(f"Error: File not found. Version {version} may not exist.", file=sys.stderr)
        else:
            print(f"Error: HTTP {e.code}: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"Error: Network error: {e.reason}", file=sys.stderr)
        sys.exit(1)


def download_file(url: str, output_path: Path) -> None:
    """Download file from URL."""
    print(f"Downloading to {output_path}...")

    request = Request(url)

    try:
        with urlopen(request, timeout=300) as response:
            total_size = response.headers.get("Content-Length")
            if total_size:
                total_size = int(total_size)
                print(f"  File size: {total_size / 1024 / 1024:.1f} MB")

            with open(output_path, "wb") as f:
                downloaded = 0
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = downloaded / total_size * 100
                        print(f"\r  Progress: {pct:.1f}%", end="", flush=True)
                print()  # newline after progress
    except (HTTPError, URLError) as e:
        print(f"Error downloading file: {e}", file=sys.stderr)
        sys.exit(1)


def extract_tar(tar_path: Path, output_dir: Path) -> Path:
    """Extract tar archive and return path to TSV file."""
    print(f"Extracting {tar_path}...")

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(output_dir, filter="data")

    # Find the .tsv.gz file
    for f in output_dir.glob("*.tsv.gz"):
        print(f"  Found: {f.name}")
        # Decompress
        tsv_path = f.with_suffix("")  # Remove .gz
        print(f"  Decompressing to {tsv_path.name}...")
        with gzip.open(f, "rb") as gz_in:
            with open(tsv_path, "wb") as tsv_out:
                shutil.copyfileobj(gz_in, tsv_out)
        return tsv_path

    print("Error: No TSV file found in archive", file=sys.stderr)
    sys.exit(1)


def verify_data(tsv_path: Path) -> None:
    """Verify the downloaded data."""
    print(f"Verifying {tsv_path}...")

    with open(tsv_path, encoding="utf-8") as f:
        # Read header
        header = f.readline().strip().split("\t")
        if "GENE_SYMBOL" not in header or "ROLE_IN_CANCER" not in header:
            print("Error: Unexpected file format", file=sys.stderr)
            sys.exit(1)

        # Count genes
        gene_count = sum(1 for _ in f)

    print(f"  Columns: {len(header)}")
    print(f"  Genes: {gene_count}")

    # Count by role
    oncogene_count = 0
    tsg_count = 0
    both_count = 0

    with open(tsv_path, encoding="utf-8") as f:
        f.readline()  # Skip header
        role_idx = header.index("ROLE_IN_CANCER")
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) > role_idx:
                role = fields[role_idx].lower()
                if "oncogene" in role and "tsg" in role:
                    both_count += 1
                elif "oncogene" in role:
                    oncogene_count += 1
                elif "tsg" in role:
                    tsg_count += 1

    print(f"  Oncogenes: {oncogene_count}")
    print(f"  Tumor suppressors: {tsg_count}")
    print(f"  Both: {both_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download COSMIC Cancer Gene Census data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--email",
        help="COSMIC account email (or set COSMIC_EMAIL env var)",
    )
    parser.add_argument(
        "--password",
        help="COSMIC account password (or set COSMIC_PASSWORD env var)",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"COSMIC version to download (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--keep-tar",
        action="store_true",
        help="Keep the downloaded tar file after extraction",
    )

    args = parser.parse_args()

    # Get credentials
    email, password = get_credentials(args)
    auth_string = get_auth_string(email, password)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get download URL
    download_url = get_download_url(auth_string, args.version)

    # Download the file
    tar_filename = f"Cosmic_CancerGeneCensus_Tsv_{args.version}_GRCh38.tar"
    tar_path = args.output_dir / tar_filename
    download_file(download_url, tar_path)

    # Extract
    tsv_path = extract_tar(tar_path, args.output_dir)

    # Clean up tar file unless requested to keep
    if not args.keep_tar:
        print(f"Removing {tar_path}...")
        tar_path.unlink()
        # Also remove the .gz file
        gz_path = tsv_path.with_suffix(".tsv.gz")
        if gz_path.exists():
            gz_path.unlink()
        # Remove README if present
        for readme in args.output_dir.glob("README_*.txt"):
            readme.unlink()

    # Verify
    verify_data(tsv_path)

    print()
    print("Success! Cancer Gene Census saved to:")
    print(f"  {tsv_path}")


if __name__ == "__main__":
    main()
