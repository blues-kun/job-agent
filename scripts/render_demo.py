"""生成120秒中文字幕演示：展示实测接口与研究结果，不冒充浏览器录屏。"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlparse

import httpx
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg

from research.common import private_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--url",default="http://127.0.0.1:8090")
    parser.add_argument("--font",type=Path,default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    parser.add_argument("--embedding-result",type=Path,required=True)
    args=parser.parse_args();target=private_path(args.output)
    if target.exists():raise FileExistsError("演示视频已存在")
    if urlparse(args.url).hostname not in {"127.0.0.1","localhost","::1"}:raise ValueError("演示仅访问本机")
    with httpx.Client(base_url=args.url,timeout=90,trust_env=False) as client:
        try:
            def get(path):
                response=client.get(path);response.raise_for_status();return response.json()
            def post(path,data):
                response=client.post(path,json=data);response.raise_for_status();return response.json()
            overview=get("/api/v2/overview");research=get("/api/v2/research")
            sample=next(item for item in get("/api/v2/samples")["samples"] if item["id"]=="python-junior")
            payload={"text":sample["text"],"preferences":sample["preferences"]}
            recommendation=post("/api/v2/recommend",payload)
            target_job=next(item for item in recommendation["jobs"] if item["gaps"])
            material={**payload,"job_id":target_job["id"]}
            diagnosis=post("/api/v2/diagnose",material);coach=post("/api/v2/coach",material);rewrite=post("/api/v2/rewrite",material)
            graph=get("/api/v2/graph/"+target_job["id"])
        finally:client.delete("/api/v2/feedback")
    embedding=json.loads(args.embedding_result.read_text())
    top=", ".join(row["name"] for row in overview["skills"][:5])
    matched=diagnosis["matched"][:2]
    pages=[
      ("从岗位需求，到可执行的求职行动",["职向 · 岗位需求与简历工作台",f"当前实测：{overview['total']:,} 个岗位内容版本 / {overview['category_count']} 个职类", "需求分析 → 简历匹配 → 双侧证据 → 补证行动 → 重新匹配", "本视频依据当前本地接口生成；不是浏览器录屏。"]),
      ("01 先看真实需求",[f"岗位中高频提及的技能：{top}", f"可比月薪广告样本：{overview['salary_count']:,} 条",f"广告区间中点 P25 / P50 / P75：{' / '.join(str(n) for n in overview['salary_quantiles'])} 元", "历史岗位快照不代表今天仍在招聘；提及技能不等于必须掌握。"]),
      ("02 带入简历，检查条件",[f"使用样例：{sample['name']}（完全虚构）",f"目标方向：{sample['preferences']['intent']}",f"城市、薪资、学历、经验过滤后：{recommendation['total_eligible']:,} 个无明确硬冲突岗位", f"多路召回与规则精排后返回 {len(recommendation['jobs'])} 项；当前检索：{recommendation['retriever']}", "信息残缺或只有技能清单时，系统先追问。"]),
      ("03 核对每一项匹配",[f"示例岗位：{target_job['title']}",*[f"{item['skill']}：简历“{item['resume_evidence']['quote']}” ↔ 岗位“{item['job_evidence']['quote']}”" for item in matched],f"需求图含 {len(graph['groups'])} 个要求组；任选一项不会被算成全部必需。","每处引用保留字段与起止位置；模型预测关系不进入事实证据。"]),
      ("04 从差距到行动",[f"当前补证动作模式：{coach['mode']}",*[f"{item['skill']}：{item['action']}" for item in coach['items'][:2]],f"简历整理：{rewrite.get('mode','事实保全')}；新增事实 {rewrite['new_claims']} 条。","用户核对后应用原段重排，再匹配；表达变好不冒充能力提高。"]),
      ("05 向量微调已经实跑",["Qwen3-Embedding-0.6B · LoRA · 600 步",f"开发代理 Recall@10：{embedding['metrics']['recall_at_10']['baseline']:.4f} → {embedding['metrics']['recall_at_10']['adapted']:.4f}",f"开发代理 nDCG@10：{embedding['metrics']['ndcg_at_10']['baseline']:.4f} → {embedding['metrics']['ndcg_at_10']['adapted']:.4f}","128 个固定标题查询 / 2,107 条候选；重载后的排名完全一致。","这是标题到 JD 的自监督结果，尚不能证明真实人岗提升。"]),
      ("06 图模型要经得起对照",["GraphSAGE / 技能池化 MLP / 随机边图，各训练三个种子。","真实图和随机图在当前自对齐任务上表现相同。","结论：目前没有图结构增益证据，不接入默认推荐。","后续用独立人岗标注与严格遮蔽任务检验，保留失败记录。"]),
      ("07 人工核验，让系统继续进步",[f"工作台已备 {research['queries']} 份待审核场景 / {research['task_count']:,} 条标注任务。","相关性、硬条件、引用、技能、幻觉、追问、事实保全、多样性与成本分别评估。","人工金标为空时保持空值，不生成漂亮的推荐指标。","本地体验：http://127.0.0.1:8090", "下一步：独立盲审 → 仲裁 → 分组排序训练 → 留出评测。"]),
    ]
    fonts={size:ImageFont.truetype(str(args.font),size) for size in [17,22,25,42]}
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="job-agent-demo-") as temp:
        directory=Path(temp)
        for index,(heading,lines) in enumerate(pages):
            frame=Image.new("RGB",(1280,720),"#f3f5ef");draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,18,720),fill="#246953")
            draw.text((62,45),"职向 / 研究与产品演示",font=fonts[22],fill="#28775e")
            draw.text((62,102),heading,font=fonts[42],fill="#173b30")
            y=202
            for line in lines:
                # 按实际字体宽度折行，中文不依赖空格。
                chunk="";wrapped=[]
                for char in line:
                    if draw.textlength(chunk+char,font=fonts[25])>1130:wrapped.append(chunk);chunk=""
                    chunk+=char
                if chunk:wrapped.append(chunk)
                for part in wrapped:draw.text((67,y),part,font=fonts[25],fill="#344b43");y+=41
                y+=18
            draw.line((62,636,1210,636),fill="#cad7ca",width=2)
            draw.rectangle((62,637,62+int(1148*(index+1)/len(pages)),642),fill="#28775e")
            draw.text((62,661),"基于实际接口和训练记录 · 所用简历为虚构样例 · 历史岗位快照",font=fonts[17],fill="#65746b")
            draw.text((1140,660),f"{index+1} / 8",font=fonts[22],fill="#28775e")
            frame.save(directory/f"frame-{index:02}.png")
        binary=imageio_ffmpeg.get_ffmpeg_exe()
        command=[binary,"-hide_banner","-loglevel","error","-framerate","1/15","-i",str(directory/"frame-%02d.png"),"-t","120","-r","24","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart","-n",str(target)]
        subprocess.run(command,check=True)
    manifest={"duration_seconds":120,"format":"1280x720 H264 MP4","audio":False,"browser_recording":False,"source":"当前API实测与归档训练指标","pages":[heading for heading,_ in pages],"source_snapshot":overview['snapshot']}
    target.with_suffix(".json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(str(target))


if __name__=="__main__":main()
