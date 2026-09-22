"""模型只在白名单行动和原段顺序中决策，事实由程序保全。"""
from urllib.parse import urlparse
import os
import httpx

ACTIONS={
    "clarify":"补充一段真实项目经历：背景、个人职责、使用方法与可核验结果。",
    "evidence":"如果实际做过，请补充代码、文档或项目结果；没有做过则保持为待学习项。",
    "practice":"将这项要求列入学习计划，完成可运行练习后再记录真实使用经历。",
    "interview":"用一个亲自完成的项目解释技术选择、操作步骤、结果与局限。",
}


class Coach:
    def __init__(self,url=None):
        self.url=(url or os.environ.get("JOB_AGENT_COACH_URL","")).rstrip("/")
        if self.url and urlparse(self.url).hostname not in {"localhost","127.0.0.1","::1"}:
            raise ValueError("本版简历模型只允许显式配置的本机服务")

    def decide(self,instruction,data):
        if not self.url:raise RuntimeError("未配置本地指令模型")
        with httpx.Client(timeout=90,trust_env=False) as client:
            response=client.post(self.url+"/decide",json={"instruction":instruction,"data":data})
            response.raise_for_status()
            return response.json()

    def diagnose(self,diagnosis):
        gaps=diagnosis["gaps"][:12]
        if not gaps:return {"mode":"规则","items":[],"message":"没有可确认的词典缺口，请继续核对完整岗位要求。"}
        try:
            output=self.decide('对尚未证明的要求选择最多3项最值得先采取的行动。只输出 {"items":[{"gap_index":0,"action":"evidence"}]}。action只能是clarify/evidence/practice/interview，gap_index必须是输入数组下标。不得输出其他字段。',
                               {"job_title":diagnosis["job"]["title"],"gaps":[{"skill":gap["skill"],"preferred":gap["preferred"],"source":gap["evidence"]["quote"]} for gap in gaps]})
            decision=output["decision"]
            if set(decision)!={"items"} or not isinstance(decision["items"],list) or not 1<=len(decision["items"])<=3:raise ValueError()
            items=[];seen=set()
            for row in decision["items"]:
                if set(row)!={"gap_index","action"} or type(row["gap_index"]) is not int or row["gap_index"] not in range(len(gaps)) or row["action"] not in ACTIONS or row["gap_index"] in seen:raise ValueError()
                seen.add(row["gap_index"]);gap=gaps[row["gap_index"]]
                items.append({"skill":gap["skill"],"action":ACTIONS[row["action"]],"evidence":gap["evidence"]})
            return {"mode":"本地LLM选择行动，程序绑定证据","items":items,"model":output["model"],"message":"模型选择优先顺序；行动模板不会把学习计划写成已有能力。"}
        except (httpx.HTTPError,ValueError,KeyError,TypeError,RuntimeError):
            return {"mode":"规则降级","items":[{"skill":gap["skill"],"action":ACTIONS["evidence"],"evidence":gap["evidence"]} for gap in gaps[:3]],"message":"本地模型未就绪或输出未通过格式核验，已使用证据模板。"}

    def reorder(self,chunks,title):
        output=self.decide('按目标岗位相关性给简历原段排序。只输出 {"order":[0,1]}，order必须恰好包含所有原段下标，每个一次。不得添加或改写任何文本。',{"job_title":title,"paragraphs":chunks})
        decision=output["decision"]
        order=decision.get("order")
        if set(decision)!={"order"} or not isinstance(order,list) or any(type(i) is not int for i in order) or sorted(order)!=list(range(len(chunks))):raise ValueError("模型段落顺序不合法")
        return [chunks[i] for i in order],output["model"]
