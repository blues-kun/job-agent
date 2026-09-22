"""以显式研究版本启动本地工作台；不猜测多个版本，不自动开启训练。"""
import argparse
import json
import os
from pathlib import Path

import uvicorn

from job_agent.api import create_app
from research.common import private_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--benchmark",type=Path)
    parser.add_argument("--synthesis",type=Path)
    parser.add_argument("--retrieval-review",type=Path)
    parser.add_argument("--dense-cache",type=Path)
    parser.add_argument("--private-root",type=Path,default=Path.home()/".cache/job-agent")
    parser.add_argument("--coach-url",default="")
    parser.add_argument("--embedding-url",default="http://127.0.0.1:8091")
    parser.add_argument("--port",type=int,default=8090)
    args=parser.parse_args()
    dataset=private_path(args.dataset);manifest=json.loads((dataset/"manifest.json").read_text())
    os.environ["JOB_AGENT_DATA"]=manifest["source_path"]
    os.environ["JOB_AGENT_RESEARCH_DIR"]=str(dataset)
    for name,value in [("JOB_AGENT_BENCHMARK_DIR",args.benchmark),("JOB_AGENT_SYNTHESIS_DIR",args.synthesis),("JOB_AGENT_RETRIEVAL_REVIEW_DIR",args.retrieval_review),("JOB_AGENT_DENSE_DIR",args.dense_cache)]:
        if value:os.environ[name]=str(private_path(value))
        else:os.environ.pop(name,None)
    os.environ["JOB_AGENT_COACH_URL"]=args.coach_url
    os.environ["JOB_AGENT_EMBEDDING_URL"]=args.embedding_url
    uvicorn.run(create_app(private_root=private_path(args.private_root)),host="127.0.0.1",port=args.port,access_log=False)


if __name__=="__main__":main()
