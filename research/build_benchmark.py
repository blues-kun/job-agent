#!/usr/bin/env python3
"""构建虚构、未审阅的评测场景与空标签候选池，仅写仓库外私有目录。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler

SCHEMA = "job-agent-benchmark-v1"
DIRECTIONS = [
    {"name":"Python后端", "categories":["Python","后端开发工程师"], "skills":["Python","SQL","FastAPI","Redis"], "project":"使用Python编写接口，使用SQL查询数据，并为接口编写测试", "keywords":["python","后端"]},
    {"name":"Java后端", "categories":["Java","后端开发工程师"], "skills":["Java","Spring Boot","MySQL","Redis"], "project":"使用Java与Spring Boot实现接口，使用MySQL保存业务数据并编写单元测试", "keywords":["java","后端"]},
    {"name":"前端", "categories":["前端开发工程师","JavaScript","Web前端"], "skills":["JavaScript","TypeScript","Vue","CSS"], "project":"使用JavaScript与Vue开发页面，使用CSS实现布局并对接口异常进行处理", "keywords":["前端","javascript","vue"]},
    {"name":"数据分析", "categories":["数据分析师","数据分析"], "skills":["Python","SQL","Excel","Pandas"], "project":"使用SQL整理数据，使用Python与Pandas完成描述分析，并核对报表口径", "keywords":["数据分析","数据分析师"]},
    {"name":"测试开发", "categories":["测试开发","测试开发工程师","自动化测试","软件测试"], "skills":["Python","SQL","自动化测试","Selenium"], "project":"使用Python编写自动化测试，使用SQL核对结果并整理可复现的缺陷记录", "keywords":["测试开发","自动化测试"]},
    {"name":"算法/LLM应用", "categories":["算法工程师","机器学习","自然语言处理","大模型算法"], "skills":["Python","PyTorch","Transformer","RAG"], "project":"使用Python与PyTorch训练文本分类模型，固定数据划分后比较验证结果，并尝试检索增强问答", "keywords":["算法","机器学习","大模型","自然语言"]},
]
PERSONAS = ["应届项目", "应届课程", "一年经验", "三年经验", "转岗", "资深", "信息残缺", "技能堆砌"]
STOP = set("熟悉 掌握 工作 要求 经验 项目 使用 负责 相关 进行 能够 以及 完成".split())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    with path.open("x",encoding="utf-8") as stream:
        for row in rows:stream.write(canonical(row)+"\n")
        stream.flush();os.fsync(stream.fileno())
    path.chmod(0o600)


def write_json(path, value):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    path.chmod(0o600)


def tokenize(text):
    """明确的中文二元字组＋英文技术词，保持无额外分词依赖与可复跑。"""
    words=[]
    for token in re.findall(r"[a-z][a-z0-9+#.]*|[\u4e00-\u9fff]+",text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+",token):
            words.extend(token[i:i+2] for i in range(len(token)-1) if token[i:i+2] not in STOP)
        elif len(token)>1:words.append(token)
    return words


class BM25:
    def __init__(self,jobs):
        self.jobs=jobs;self.postings=defaultdict(list);lengths=[]
        for index,job in enumerate(jobs):
            text=" ".join([job["title"],job["title"],job["category"]," ".join(job["skills"]),job["requirements"],job["description"]])
            frequencies=Counter(tokenize(text));lengths.append(sum(frequencies.values()))
            for word,count in frequencies.items():self.postings[word].append((index,count))
        self.lengths=lengths;self.average=sum(lengths)/max(len(lengths),1)
    def rank(self,text):
        scores=defaultdict(float);n=len(self.jobs)
        for word in sorted(set(tokenize(text))):
            posting=self.postings.get(word,[])
            idf=math.log1p((n-len(posting)+.5)/(len(posting)+.5))
            for index,frequency in posting:
                scores[index]+=idf*frequency*2.2/(frequency+1.2*(.25+.75*self.lengths[index]/max(self.average,1)))
        return [(self.jobs[index]["job_id"],score) for index,score in sorted(scores.items(),key=lambda item:(-item[1],self.jobs[item[0]]["job_id"]))]


def make_queries():
    queries=[]
    for direction_index,direction in enumerate(DIRECTIONS):
        dev_personas={direction_index%6,6+(direction_index%2)}
        for persona_index,persona in enumerate(PERSONAS):
            preferences={"city":"深圳","intent":direction["name"],"education":"本科","experience_years":0,"salary_min":8000,"district":""}
            if persona_index==0:
                text=f"本科应届，求职方向为{direction['name']}。课程项目：{direction['project']}。暂无全职工作经验；希望进一步积累工程实践。"
            elif persona_index==1:
                text=f"本科应届，方向为{direction['name']}。课程练习：{direction['project']}。目前主要是课堂与个人练习，了解{direction['skills'][-1]}，尚未负责线上生产系统。"
            elif persona_index==2:
                preferences.update(experience_years=1,salary_min=10000)
                text=f"本科，1年相关工作经验，目标是{direction['name']}。项目经历：{direction['project']}。在他人评审下完成模块交付，熟悉{direction['skills'][0]}与{direction['skills'][1]}。"
            elif persona_index==3:
                preferences.update(experience_years=3,salary_min=15000)
                text=f"本科，3年相关工作经验，方向为{direction['name']}。项目经历：{direction['project']}。负责模块设计、异常排查与协作交付；没有带团队或预算管理经历。"
            elif persona_index==4:
                text=f"本科，计划由技术支持转向{direction['name']}，暂无目标方向的正式工作经验。转岗练习：{direction['project']}。能够说明练习步骤，但尚未独立维护生产系统。"
            elif persona_index==5:
                preferences.update(experience_years=8,salary_min=25000)
                text=f"本科，8年相关工作经验，方向为{direction['name']}。项目经历：{direction['project']}。负责多个模块的设计评审、故障排查和新人指导，希望继续从事技术工作。"
            elif persona_index==6:
                preferences={"city":"深圳","intent":direction["name"],"education":None,"experience_years":None,"salary_min":None,"district":""}
                text=f"想在深圳找一份{direction['name']}相关工作，希望了解有哪些机会。"
            else:
                preferences.update(education=None,experience_years=None,salary_min=None)
                text="技能清单："+"、".join(direction["skills"]+["Linux","Docker","Git","分布式","机器学习"])+"。"
            split="dev" if persona_index in dev_personas else "test"
            query_id="q_"+digest([direction["name"],persona,text,preferences,split])[:20]
            queries.append({"query_id":query_id,"text":text,"preferences":preferences,"split":split,
                            "direction":direction["name"],"persona":persona,"scenario":f"{direction['name']} / {persona}",
                            "source":"programmatic_fiction_unreviewed","is_real_resume":False,"is_human_gold":False,
                            "annotation_status":"unreviewed","scenario_group":"clarification_stress" if persona_index>=6 else "recommendation_case",
                            "authoring_note":"完全虚构、独立模板编写，没有按测试JD反推；场景类型不是人工判定的正确动作。"})
    return queries


def http_json(url,payload=None):
    opener=build_opener(ProxyHandler({}))
    request=Request(url,data=None if payload is None else json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with opener.open(request,timeout=120) as response:return json.load(response)


def load_dense(features,jobs,jobs_sha,queries,url):
    if features is None:return None,{"used":False,"reason":"未提供冻结向量特征；本次不声称包含dense候选"}
    import numpy as np
    if urlparse(url).hostname not in {"127.0.0.1","localhost","::1"}:raise ValueError("查询编码仅允许本机服务。")
    metadata=json.loads(features.with_suffix(".manifest.json").read_text("utf-8"))
    if metadata.get("jobs_sha256")!=jobs_sha or metadata.get("features_sha256")!=sha256(features):
        raise ValueError("冻结向量缓存与内容版本或指纹不一致。")
    with np.load(features,allow_pickle=False) as archive:
        ids=archive["job_ids"].astype(str).tolist();matrix=archive["text_vectors"].copy()
    if len(ids)!=len(set(ids)) or set(ids)!={job['job_id'] for job in jobs}:raise ValueError("向量job_id集合与本次数据不一致。")
    model=metadata["model"];health=http_json(url.rstrip('/')+'/health')
    for key in ("model","revision","pooling","dimension","max_tokens"):
        if health.get(key)!=model.get(key):raise ValueError("查询编码器与冻结特征模型不一致。")
    if matrix.shape!=(len(ids),model['dimension']) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=.003):
        raise ValueError("岗位向量形状、有限性或归一化失败。")
    query_vectors=[]
    for offset in range(0,len(queries),8):
        value=http_json(url.rstrip('/')+'/encode',{"texts":[q['text'] for q in queries[offset:offset+8]],"query":True})
        if value.get('model')!=model['model'] or value.get('revision')!=model['revision']:raise ValueError("编码过程中模型版本变化。")
        query_vectors.extend(value['vectors'])
    query_vectors=np.asarray(query_vectors,dtype=np.float32)
    if query_vectors.shape!=(len(queries),model['dimension']) or not np.isfinite(query_vectors).all() or not np.allclose(np.linalg.norm(query_vectors,axis=1),1,atol=.003):
        raise ValueError("查询向量校验失败。")
    return ({"ids":ids,"matrix":matrix,"query_vectors":query_vectors},
            {"used":True,"features_sha256":metadata['features_sha256'],"features_jobs_sha256":jobs_sha,
             "model":model,"query_template":"编码服务官方query模板，输入为虚构简历正文","query_count":len(queries),
             "query_vectors_sha256":hashlib.sha256(query_vectors.tobytes()).hexdigest()})


def direction_match(query,job):
    direction=next(item for item in DIRECTIONS if item['name']==query['direction'])
    return job['category'] in direction['categories'] or any(word in (job['title']+' '+job['category']).lower() for word in direction['keywords'])


def rule_signals(query,job):
    signals=[];pref=query['preferences']
    match=re.search(r'(\d+)\s*(?:[-~至]\s*\d+)?年',job['requirements'])
    if match and pref.get('experience_years') is not None and int(match.group(1))>pref['experience_years']:
        signals.append('可能存在年资差，待人工核验')
    salary=job.get('salary',{})
    if salary.get('status')=='月薪' and pref.get('salary_min') is not None and salary['monthly_high']<pref['salary_min']:
        signals.append('广告月薪上界低于场景期望，待人工核验')
    return signals


def make_pools(jobs,queries,pool_size,seed,dense):
    representatives={}
    for job in sorted(jobs,key=lambda row:row['job_id']):representatives.setdefault((job['split'],job['job_family_id']),job)
    corpora={split:sorted([job for (group_split,_),job in representatives.items() if group_split==split],key=lambda row:row['job_id']) for split in ('dev','test')}
    engines={split:BM25(rows) for split,rows in corpora.items()}
    by_id={job['job_id']:job for job in jobs};pools=[];tasks=[]
    dense_lookup={job_id:index for index,job_id in enumerate(dense['ids'])} if dense else {}
    for query_index,query in enumerate(queries):
        corpus=corpora[query['split']]
        if len(corpus)<pool_size:raise ValueError('本split的不同岗位家族不足目标候选池大小。')
        lexical=engines[query['split']].rank(query['text']+' '+query['preferences']['intent'])
        routes={'bm25':lexical}
        if dense:
            import numpy as np
            indices=np.array([dense_lookup[job['job_id']] for job in corpus])
            scores=dense['matrix'][indices]@dense['query_vectors'][query_index]
            routes['frozen_bge']=sorted([(job['job_id'],float(score)) for job,score in zip(corpus,scores)],key=lambda row:(-row[1],row[0]))
        same=[job for job in corpus if direction_match(query,job)]
        same.sort(key=lambda job:(-len(rule_signals(query,job)),digest([seed,query['query_id'],job['job_id']])))
        routes['rule_contrast']=[(job['job_id'],None) for job in same]
        randomized=list(corpus);random.Random(f"{seed}:{query['query_id']}").shuffle(randomized)
        routes['random']=[(job['job_id'],None) for job in randomized]
        selected={}
        def add_route(name,limit):
            if limit<=0:return
            added=0
            for rank,(job_id,score) in enumerate(routes[name],1):
                if job_id not in selected and len(selected)>=pool_size:break
                if job_id not in selected:
                    selected[job_id]={'job_id':job_id,'job_family_id':by_id[job_id]['job_family_id'],'sources':[]};added+=1
                provenance={'source':name,'rank':rank,'score':score}
                if name=='rule_contrast':provenance['signals']=rule_signals(query,by_id[job_id])
                selected[job_id]['sources'].append(provenance)
                if added>=limit:break
        # 先覆盖两个检索器的前10个不同家族；重复条目合并来源，不制造未标注=0。
        add_route('bm25',min(10,pool_size))
        if dense:add_route('frozen_bge',min(10,pool_size-len(selected)))
        add_route('rule_contrast',min(5,pool_size-len(selected)))
        add_route('random',pool_size-len(selected))
        if len(selected)!=pool_size:raise ValueError('候选池补齐失败。')
        candidates=list(selected.values());random.Random(f"blind:{seed}:{query['query_id']}").shuffle(candidates)
        for candidate in candidates:
            task_id='rel_'+digest([query['query_id'],candidate['job_id']])[:20]
            tasks.append({'task_id':task_id,'kind':'relevance','job_id':candidate['job_id'],'query_id':query['query_id'],
                          'job_family_id':candidate['job_family_id'],'split':query['split'],'label':None,'annotation_status':'unlabeled',
                          'pool_sources':candidate['sources'],'source':'programmatic_candidate_pool_unreviewed','is_human_gold':False})
        pools.append({'query_id':query['query_id'],'split':query['split'],'candidates':candidates,'candidate_count':len(candidates)})
    return pools,tasks,{split:len(rows) for split,rows in corpora.items()}


def make_review_profiles(jobs,count,seed):
    families={}
    for job in sorted(jobs,key=lambda row:row['job_id']):
        if job['split']=='train' and job['description'] and job['skills']:
            families.setdefault(job['job_family_id'],job)
    seeds=sorted(families.values(),key=lambda job:digest([seed,job['job_family_id']]))
    if len(seeds)<count:raise ValueError('训练分区可用岗位家族不足审核模板种子数。')
    result=[]
    for job in seeds[:count]:
        evidence=[]
        for skill,item in sorted(job['skills'].items()):
            if item['level']=='否定':continue
            field,start,end=item['field'],item['start'],item['end']
            if job[field][start:end]!=item['quote']:raise ValueError('审核材料技能引用无法回溯。')
            evidence.append({'skill':skill,'job_id':job['job_id'],'field':field,'start':start,'end':end,'quote':item['quote']})
        for variant in ('项目证据待填写','能力缺口待核验'):
            profile_id='review_'+digest([job['job_id'],variant])[:20]
            source_excerpt=job['description'][:240]
            text=f"虚构审核模板：{variant}，不是任何求职者的真实履历。\n目标岗位：{job['title']}。\n岗位参考原文：{source_excerpt}\n请审核者依据真实或另行批准的虚构案例填写经历；不能把岗位要求直接改写成候选人已掌握的技能。"
            result.append({'profile_id':profile_id,'split':'train','seed_job_id':job['job_id'],'job_family_id':job['job_family_id'],
                           'source':'programmatic_unreviewed','not_for_training':True,'is_human_gold':False,'label':None,
                           'variant':variant,'text':text,'skills_to_review':[item['skill'] for item in evidence],
                           'jd_evidence':evidence+[{'job_id':job['job_id'],'field':'description','start':0,'end':len(source_excerpt),'quote':source_excerpt}],
                           'review_status':'pending','note':'这些是待填写画像模板，不是已生成的匹配正例。'})
    return result


def verify(jobs,queries,pools,tasks,review_profiles,pool_size):
    by_id={row['job_id']:row for row in jobs};q_by_id={row['query_id']:row for row in queries}
    assert len(queries)==48 and Counter(q['split'] for q in queries)==Counter(dev=12,test=36)
    assert len(tasks)==len(queries)*pool_size and all(row['label'] is None for row in tasks)
    for pool in pools:
        ids=[row['job_id'] for row in pool['candidates']]
        assert len(ids)==pool_size and len(set(ids))==pool_size
        assert len({by_id[job_id]['job_family_id'] for job_id in ids})==pool_size
        assert all(by_id[job_id]['split']==pool['split'] for job_id in ids)
    for task in tasks:assert task['split']==by_id[task['job_id']]['split']==q_by_id[task['query_id']]['split']
    for profile in review_profiles:
        assert by_id[profile['seed_job_id']]['split']=='train' and profile['not_for_training'] and profile['label'] is None
        for ev in profile['jd_evidence']:assert by_id[ev['job_id']][ev['field']][ev['start']:ev['end']]==ev['quote']
    return {'queries':len(queries),'query_split_counts':dict(Counter(q['split'] for q in queries)),
            'query_scenario_split_counts':dict(Counter(q['split']+'/'+q['scenario_group'] for q in queries)),
            'relevance_tasks':len(tasks),'pool_size':pool_size,'all_labels_null':True,'qrels_rows':0,
            'candidate_split_mismatch':0,'duplicate_families_per_pool':0,'review_profiles':len(review_profiles),
            'review_seed_families':len({row['job_family_id'] for row in review_profiles}),'review_profiles_train_only':True}


def build(dataset,repository,features=None,url='http://127.0.0.1:8091',seed=42,pool_size=30,review_seeds=500):
    if not dataset.is_absolute() or not repository.is_absolute():raise ValueError('数据和仓库路径必须绝对路径。')
    dataset,repository=dataset.resolve(),repository.resolve()
    if dataset.is_relative_to(repository):raise ValueError('评测产物必须位于仓库外。')
    if pool_size<20 or review_seeds<0:raise ValueError('候选池至少20；审核种子数非负。')
    jobs_path=dataset/'jobs.jsonl';jobs_sha=sha256(jobs_path)
    dataset_manifest=json.loads((dataset/'manifest.json').read_text('utf-8'))
    if dataset_manifest['files']['jobs.jsonl']['sha256']!=jobs_sha:raise ValueError('岗位数据与manifest不一致。')
    jobs=read_jsonl(jobs_path);queries=make_queries()
    dense,dense_info=load_dense(features,jobs,jobs_sha,queries,url)
    config={'schema':SCHEMA,'jobs_sha256':jobs_sha,'dataset_config_hash':dataset_manifest.get('config_hash'),
            'seed':seed,'pool_size':pool_size,'review_seed_families':review_seeds,'code_sha256':sha256(Path(__file__)),
            'dense':dense_info,'bm25_tokenizer':'zh_bigram_ascii_v1','bm25_k1':1.2,'bm25_b':.75,
            'pool_universe':'同split内按job_id排序选每个家族一个代表，所有路线共享该内容ID集合'}
    config_hash=digest(config);output_root=dataset/'benchmark'
    if output_root.is_symlink():raise ValueError('benchmark输出根目录不能是符号链接。')
    output_root.mkdir(mode=0o700,exist_ok=True)
    run=output_root/('benchmark-'+config_hash[:16])
    if run.is_symlink():raise ValueError('benchmark版本目录不能是符号链接。')
    if run.exists():
        if (run/'manifest.json').is_symlink():raise ValueError('benchmark manifest不能是符号链接。')
        manifest=json.loads((run/'manifest.json').read_text('utf-8'))
        if manifest.get('config_hash')!=config_hash:raise ValueError('既有benchmark版本不相容。')
        if set(manifest.get('files',{}))!={'queries.jsonl','tasks.jsonl','pools.jsonl','qrels.jsonl','review_profiles.jsonl'}:raise ValueError('既有benchmark文件清单不完整。')
        for name,info in manifest['files'].items():
            if Path(name).name!=name or (run/name).is_symlink() or sha256(run/name)!=info['sha256']:raise ValueError('既有benchmark被改动，拒绝覆盖。')
        return run,manifest
    staging=Path(tempfile.mkdtemp(prefix='.benchmark-',dir=output_root))
    try:
        pools,tasks,universes=make_pools(jobs,queries,pool_size,seed,dense)
        review=make_review_profiles(jobs,review_seeds,seed)
        checks=verify(jobs,queries,pools,tasks,review,pool_size)
        files={}
        for name,rows in (('queries.jsonl',queries),('tasks.jsonl',tasks),('pools.jsonl',pools),('qrels.jsonl',[]),('review_profiles.jsonl',review)):
            write_jsonl(staging/name,rows);files[name]={'sha256':sha256(staging/name),'rows':len(rows)}
        manifest={'schema':SCHEMA,'config_hash':config_hash,'config':config,'created_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  'dataset_dir':str(dataset),'files':files,'checks':checks,'candidate_universe_counts':universes,
                  'dense':dense_info,'is_human_gold':False,'labels_status':'全部待人工审阅，qrels为空，无法报告nDCG/Recall',
                  'reviewer_blinding':'人工界面应隐藏pool_sources中的方法、排名、分数；只展示同一岗位与虚构材料。',
                  'training_boundary':'评测queries仅dev/test；审核模板种子仅train、not_for_training=true，未经审阅不得训练。'}
        write_json(staging/'manifest.json',manifest)
        if sha256(jobs_path)!=jobs_sha:raise ValueError('构建期间岗位内容变化。')
        if run.exists():raise ValueError('发布目标已有内容，拒绝覆盖。')
        os.rename(staging,run)
        return run,manifest
    finally:
        if staging.exists():shutil.rmtree(staging)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--repo-root',type=Path,default=Path.cwd())
    parser.add_argument('--features',type=Path);parser.add_argument('--encoder-url',default='http://127.0.0.1:8091')
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--pool-size',type=int,default=30);parser.add_argument('--review-seeds',type=int,default=500)
    args=parser.parse_args()
    try:run,manifest=build(args.dataset,args.repo_root,args.features,args.encoder_url,args.seed,args.pool_size,args.review_seeds)
    except (ValueError,OSError,KeyError,AssertionError) as exc:parser.exit(2,f'评测材料生成失败：{exc}\n')
    print(json.dumps({'run_dir':str(run),'checks':manifest['checks'],'dense_used':manifest['dense']['used'],'files':manifest['files']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
