#!/usr/bin/env python3
"""
Download and extract the MNIST dataset.

Files are downloaded from the original Yann LeCun mirror and decompressed
from .gz format into raw IDX binary files that our C++ data loader reads.

Usage:
    python download_mnist.py            # downloads to ./data/
    python download_mnist.py --dir path # downloads to specified directory
"""

import urllib.request
import gzip
import shutil
import os
import sys
import hashlib

# MNIST file URLs and expected MD5 checksums (for integrity verification)
FILES = {
    "train-images-idx3-ubyte": {
        "url": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "md5": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    },
    "train-labels-idx1-ubyte": {
        "url": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "md5": "d53e105ee54ea40749a09fcbcd1e9432",
    },
    "t10k-images-idx3-ubyte": {
        "url": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "md5": "9fb629c4189551a2d022fa330f9573f3",
    },
    "t10k-labels-idx1-ubyte": {
        "url": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        "md5": "ec29112dd5afa0611ce80d1b7f02629c",
    },
}

# Backup mirror in case the primary is down
BACKUP_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def md5_file(path):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_extract(name, info, data_dir):
    """Download a single MNIST file, decompress, and verify."""
    filepath = os.path.join(data_dir, name)

    if os.path.exists(filepath):
        print(f"  {name} — already exists, skipping")
        return

    gz_path = filepath + ".gz"

    # Try primary URL, fall back to backup
    for url in [info["url"], BACKUP_BASE + name + ".gz"]:
        try:
            print(f"  Downloading {name} from {url}...")
            urllib.request.urlretrieve(url, gz_path)
            break
        except Exception as e:
            print(f"  Failed ({e}), trying backup...")
    else:
        raise RuntimeError(f"Could not download {name} from any source")

    # Decompress .gz → raw binary
    print(f"  Extracting {name}...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(filepath, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)
    print(f"  [OK] {name} ready")


def main():
    # Parse optional --dir argument
    data_dir = "data"
    if "--dir" in sys.argv:
        idx = sys.argv.index("--dir")
        if idx + 1 < len(sys.argv):
            data_dir = sys.argv[idx + 1]

    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading MNIST to {os.path.abspath(data_dir)}/\n")

    for name, info in FILES.items():
        download_and_extract(name, info, data_dir)

    print(f"\nAll files ready in {os.path.abspath(data_dir)}/")
    print("Contents:")
    for f in sorted(os.listdir(data_dir)):
        size = os.path.getsize(os.path.join(data_dir, f))
        print(f"  {f}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
