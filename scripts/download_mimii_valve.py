"""Download the official archive, with optional parallel HTTP byte ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MD5 = {
    "valve": "bc456bef6bc28decc1f9e74ca11480c3",
    "fan": "2fd8c081d1adf559372763f204fd154f",
}


def digest(path):
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def download(path, workers, machine):
    url = f"https://zenodo.org/records/4740355/files/dev_data_{machine}.zip?download=1"
    checksum = MD5[machine]
    if workers < 1:
        raise ValueError("workers must be positive")
    if path.is_file():
        if digest(path) != checksum:
            raise ValueError(f"Invalid official archive checksum: {path}")
        return
    path.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory)
        combined = temporary / path.name
        if workers == 1:
            with urlopen(url, timeout=60) as response, combined.open("wb") as output:
                shutil.copyfileobj(response, output)
        else:
            headers = subprocess.check_output([
                "curl", "-fsSL", "--range", "0-0", "-D", "-", "-o", str(temporary / "probe"),
                url + "&range_probe=1"
            ], text=True)
            size = int(re.search(r"(?im)^content-range: bytes 0-0/(\d+)", headers)[1])
            step = min(16 * 1024 * 1024, (size + workers - 1) // workers)
            cache = path.with_suffix(".ranges")
            cache.mkdir(exist_ok=True)

            def fetch(start):
                end = min(size - 1, start + step - 1)
                part = cache / f"{start}-{end}"
                tail = part.with_suffix(".tail")
                headers_path = part.with_suffix(".headers")
                expected = end - start + 1
                for _ in range(8):
                    have = part.stat().st_size if part.exists() else 0
                    if have == expected:
                        print(f"Verified byte range {start}-{end}", flush=True)
                        return part
                    offset = start + have
                    tail.unlink(missing_ok=True)
                    headers_path.unlink(missing_ok=True)
                    subprocess.run([
                        "curl", "-fsSL", "--retry", "2", "--connect-timeout", "15",
                        "--max-time", "300", "-D", str(headers_path),
                        "--range", f"{offset}-{end}", "-o", str(tail),
                        f"{url}&range_start={offset}&range_end={end}",
                    ], check=False)
                    received = tail.stat().st_size if tail.exists() else 0
                    if (0 < received <= expected - have and headers_path.exists()
                            and f"content-range: bytes {offset}-{end}/{size}" in
                            headers_path.read_text(encoding="latin1").lower()):
                        with part.open("ab") as output, tail.open("rb") as stream:
                            shutil.copyfileobj(stream, output)
                        tail.unlink()
                        headers_path.unlink()
                if part.exists() and part.stat().st_size == expected:
                    return part
                raise RuntimeError(f"Incomplete byte range {start}-{end}, rerun to resume")

            with ThreadPoolExecutor(max_workers=workers) as executor, combined.open("wb") as output:
                for part in executor.map(fetch, range(0, size, step)):
                    with part.open("rb") as stream:
                        shutil.copyfileobj(stream, output)
        if digest(combined) != checksum:
            raise ValueError("Downloaded archive differs from Zenodo's official MD5")
        combined.replace(path)
        if workers > 1:
            shutil.rmtree(cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--machine", choices=tuple(MD5), default="valve")
    args = parser.parse_args()
    destination = ROOT / "data" / f"dev_data_{args.machine}.zip"
    download(destination, args.workers, args.machine)
    print(f"Verified archive: {destination}")
