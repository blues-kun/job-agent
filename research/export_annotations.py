"""导出未仲裁标注与评审代号一致性；绝不自动把同意票转成金标。"""
import argparse
from collections import defaultdict
from pathlib import Path

from job_agent.research_store import ResearchStore
from research.common import private_path,write_json,write_jsonl
from research.metrics import weighted_kappa


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["dataset","private-root","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();target=private_path(args.output)
    if target.exists():raise FileExistsError("导出目录已存在")
    store=ResearchStore(args.dataset,args.private_root);latest=store._latest();rows=[];paired=defaultdict(list)
    for (task,annotator),payload in latest.items():
        rows.append({"task_id":task,"annotator":annotator,**payload,"label_source":"human_entered_unadjudicated"})
        if payload.get("grade") is not None:paired[task].append((annotator,payload["grade"]))
    combinations=defaultdict(lambda:([],[]))
    for values in paired.values():
        for i,(first,a) in enumerate(sorted(values)):
            for second,b in sorted(values)[i+1:]:
                x,y=combinations[(first,second)];x.append(a);y.append(b)
    agreement=[{"reviewer_aliases":pair,"items":len(first),"quadratic_weighted_kappa":weighted_kappa(first,second),"identity_verified":False} for pair,(first,second) in combinations.items()]
    write_jsonl(target/"unadjudicated.jsonl",rows)
    write_json(target/"agreement.json",{"pairs":agreement,"human_gold_created":False,"note":"代号不同不证明由两个人独立完成；需核实身份、盲审及仲裁后再报告人工一致性。"})
    print(f"导出{len(rows)}条未仲裁输入；未生成金标。")


if __name__=="__main__":main()
