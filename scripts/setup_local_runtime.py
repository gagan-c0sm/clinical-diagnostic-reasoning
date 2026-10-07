"""Fetch pinned official inference assets into this project, verifying hashes."""
import hashlib
import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "b11951"
REV = "bc640142c66e1fdd12af0bd68f40445458f3869b"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch(url, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".partial")
        print(f"Downloading {target.name}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            downloaded = 0
            next_report = 256 * 1024 * 1024
            while block := response.read(8 * 1024 * 1024):
                out.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    print(f'  {downloaded / 1024**3:.2f} GiB received', flush=True)
                    next_report += 256 * 1024 * 1024
        partial.replace(target)
    with target.open('rb') as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Hash mismatch: {target}")
    print(f"Verified {target.name}", flush=True)
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', choices=['4B','8B'], default='4B')
    args = parser.parse_args()
    revision = REV if args.model_size == '4B' else '7c41481f57cb95916b40956ab2f0b139b296d974'
    repo = f'Qwen/Qwen3-{args.model_size}-GGUF'
    runtime = ROOT / "tools" / "llama.cpp"
    runtime.mkdir(parents=True, exist_ok=True)
    release = get_json(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{TAG}")
    names = [f"llama-{TAG}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]
    manifest = {"runtime_release": TAG, "model_repo":repo, "model_revision": revision, "files": {}}
    for name in names:
        asset = next(a for a in release["assets"] if a["name"] == name)
        archive = ROOT / "data" / "downloads" / name
        expected = asset["digest"].removeprefix("sha256:")
        manifest["files"][name] = fetch(asset["browser_download_url"], archive, expected)
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if not (runtime/item.filename).resolve().is_relative_to(runtime.resolve()):
                    raise ValueError("Unsafe archive path")
                dest = runtime/item.filename
                if item.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif dest.exists():
                    with dest.open('rb') as source, z.open(item) as archive_source:
                        if hashlib.file_digest(source, 'sha256').digest() != hashlib.file_digest(archive_source, 'sha256').digest():
                            raise ValueError(f'Existing runtime differs: {dest}; use a fresh runtime directory')
                else:
                    z.extract(item, runtime)
    files = get_json(f"https://huggingface.co/api/models/{repo}/tree/{revision}")
    name = f"Qwen3-{args.model_size}-Q4_K_M.gguf"
    model = next(f for f in files if f["path"] == name)
    manifest["files"][name] = fetch(
        f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
        ROOT/"data"/"models"/name, model["lfs"]["oid"])
    (runtime/f"manifest-{args.model_size}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Project-local runtime ready.", flush=True)


if __name__ == "__main__":
    main()
