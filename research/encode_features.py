"""为内容版本生成真实BGE文本/技能特征；不按旧业务键猜测缓存。"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np

from research.common import private_path, read_jsonl, sha256, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--url",default="http://127.0.0.1:8091")
    args=parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1","localhost","::1"}:
        raise ValueError("特征编码仅允许本机端点")
    target=private_path(args.output)
    if target.exists():raise FileExistsError("特征文件已存在，请使用新的输出文件")
    jobs=read_jsonl(args.jobs)
    ids=[job["job_id"] for job in jobs]
    assert len(ids)==len(set(ids))
    skills=sorted({skill for job in jobs for skill in job["skills"]})
    documents=[f"{job['title']} {job['category']} {job['requirements']} {job['description']}" for job in jobs]
    with httpx.Client(timeout=120,trust_env=False) as client:
        health=client.get(args.url+"/health");health.raise_for_status();model=health.json()
        def encode(texts):
            output=[]
            for start in range(0,len(texts),48):
                response=client.post(args.url+"/encode",json={"texts":texts[start:start+48],"query":False})
                response.raise_for_status();value=response.json()
                if value["revision"]!=model["revision"] or value["model"]!=model["model"]:
                    raise ValueError("编码器版本在任务中途变化")
                output.extend(value["vectors"])
                if start%480==0:print(f"编码进度 {min(start+48,len(texts))}/{len(texts)}",flush=True)
            result=np.asarray(output,dtype=np.float32)
            assert result.shape==(len(texts),model["dimension"]) and np.isfinite(result).all()
            assert np.allclose(np.linalg.norm(result,axis=1),1,atol=.002)
            return result
        vectors=encode(documents);skill_vectors=encode(["技术技能："+skill for skill in skills])
    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=target.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream,job_ids=np.array(ids),text_vectors=vectors,skills=np.array(skills),skill_vectors=skill_vectors)
    temporary.replace(target)
    write_json(target.with_suffix(".manifest.json"),{"schema":"job-agent-features-v1","jobs_sha256":sha256(args.jobs),"features_sha256":sha256(target),"model":model,
        "template":"title-category-requirements-description-v2","document_hash":hashlib.sha256("\0".join(documents).encode()).hexdigest(),"jobs":len(jobs),"skills":len(skills),"dimension":model["dimension"]})
    print(json.dumps({"features":str(target),"shape":list(vectors.shape)},ensure_ascii=False))


if __name__=="__main__":main()
