"""研究产物的来源、私有路径和原子写入约定。"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def private_path(value):
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(ROOT):
        raise ValueError("研究数据、权重和标注必须放在仓库外")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2)
            stream.write("\n")
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl(path, values):
    path = private_path(path)
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent,prefix=".writing-")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value,ensure_ascii=False)+"\n")
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
