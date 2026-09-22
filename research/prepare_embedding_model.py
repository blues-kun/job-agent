#!/usr/bin/env python3
"""下载官方固定版本到仓库外；无鉴权令牌输出，不安装依赖。"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

REPO = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/home/xukunbo/.cache/job-agent/models/qwen3-embedding-0.6b"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, revision=REVISION, local_dir=args.model_dir, token=False,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "1_Pooling/*", "README.md"],
                      max_workers=4)
    model_file = args.model_dir / "model.safetensors"
    sha = hashlib.sha256()
    with model_file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    provenance = {"repo_id": REPO, "revision": REVISION,
                  "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "source": "official_huggingface_snapshot",
                  "model_safetensors_sha256": sha.hexdigest(), "weight_bytes": model_file.stat().st_size}
    (args.model_dir / "job_agent_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()
