import json
import uuid
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import JOBS_FILE, API_KEY, MODEL, BASE_URL, TEMPERATURE, XGB_MODEL_PATH
from data_preprocess import JobDataLoader
from resume_extract import ResumeExtractor
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from search.matcher import JobMatcher
from models import ResumeProfile


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).resolve().parent), **kwargs)

    def log_message(self, format, *args):
        """重写日志方法，过滤Vite相关请求"""
        # 过滤掉Vite相关的404错误日志
        if len(args) > 0 and isinstance(args[0], str):
            if '/@vite/' in args[0] or '/@id/' in args[0]:
                return
        # 调用父类方法记录其他日志
        super().log_message(format, *args)

    def _json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path)
        # 静默处理Vite相关请求（浏览器缓存残留）
        if p.path.startswith('/@vite/') or p.path.startswith('/@id/'):
            self.send_error(404)
            return
        print(f"[HTTP] GET {p.path}")
        if p.path == '/api/jobs':
            try:
                loader = JobDataLoader(JOBS_FILE)
                data = loader.to_dict_list()
                self._json({'jobs': data})
            except Exception as e:
                self._json({'error': str(e)}, 500)
        else:
            super().do_GET()

    def do_POST(self):
        p = urlparse(self.path)
        print(f"[HTTP] POST {p.path}")
        length = int(self.headers.get('Content-Length') or '0')
        raw = self.rfile.read(length).decode('utf-8') if length > 0 else '{}'
        print(f"[DEBUG] 收到原始数据: {raw[:200]}")  # 打印前200字符
        try:
            payload = json.loads(raw)
        except Exception as e:
            print(f"[ERROR] JSON解析失败: {e}")
            payload = {}
        print(f"[DEBUG] 解析后的payload: {payload}")
        if p.path == '/api/chat_resume':
            try:
                msgs = payload.get('messages', [])
                resume_text = ''
                for m in msgs:
                    if m.get('role') == 'user':
                        resume_text = m.get('content', '')
                use_llm = bool(payload.get('use_llm'))
                print(f"[DEBUG] use_llm={use_llm}, payload.use_llm={payload.get('use_llm')}, type={type(payload.get('use_llm'))}")
                positions_file = Path(__file__).resolve().parents[1] / 'position_dictionary.txt'
                allowed = []
                if positions_file.exists():
                    import re
                    for line in positions_file.read_text('utf-8').splitlines():
                        s = line.strip()
                        if not s:
                            continue
                        if s.startswith('[') and s.endswith(']'):
                            continue
                        s = re.sub(r'^\s*\d+\s*→\s*', '', s)
                        allowed.append(s)
                # 从文件加载内置职位分类（三级分类）
                builtin_positions_file = Path(__file__).resolve().parents[1] / 'data' / 'builtin_positions.txt'
                if builtin_positions_file.exists():
                    for line in builtin_positions_file.read_text('utf-8').splitlines():
                        line = line.strip()
                        # 跳过空行和注释
                        if not line or line.startswith('#'):
                            continue
                        if line not in allowed:
                            allowed.append(line)
                    print(f'[INFO] 职位三级分类加载完成，共 {len(allowed)} 个')
                else:
                    print(f'[WARN] 职位分类文件不存在: {builtin_positions_file}')
                
                # 如果未启用LLM，返回友好提示
                if not use_llm:
                    self._json({
                        'assistant_reply': '💡 提示：请勾选"启用智能对话"开关以使用LLM智能对话功能。',
                        'resume_profile': None,
                        'weight_suggestion': None,
                    })
                    return
                
                # 检查API_KEY
                if not API_KEY:
                    self._json({'error': 'API密钥未配置，请在.env文件中设置API_KEY'}, 500)
                    return
                
                # LLM处理
                import asyncio
                
                # 检查是否需要提取简历（仅当有简历文本且明确需要提取时）
                need_extract = payload.get('extract_resume', False)
                profile = None
                
                if need_extract and resume_text:
                    try:
                        # 提取简历
                        print(f"[DEBUG] 开始提取简历，文本长度: {len(resume_text)}")
                        ext = ResumeExtractor(allowed)
                        profile = asyncio.run(ext.extract(resume_text))
                        print(f"[DEBUG] 简历提取完成: {profile is not None}")
                    except Exception as e:
                        print(f"[ERROR] 简历提取异常: {e}")
                        import traceback
                        traceback.print_exc()
                        self._json({'error': f'简历提取失败: {str(e)}'}, 500)
                        return
                else:
                    print(f"[DEBUG] 跳过简历提取 (need_extract={need_extract}, has_text={bool(resume_text)})")
                
                try:
                    # LLM对话
                    print(f"[DEBUG] 开始LLM对话，消息数: {len(msgs)}")
                    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=TEMPERATURE)
                    lm_msgs = [SystemMessage(content='你是专业的智能求职助手，请以自然中文与用户对话，围绕求职偏好、城市、薪资与职位意向给出建议或确认更新。')]
                    for m in msgs:
                        role = m.get('role')
                        content = m.get('content', '')
                        if role == 'user':
                            lm_msgs.append(HumanMessage(content=content))
                        elif role == 'assistant':
                            lm_msgs.append(AIMessage(content=content))
                    
                    print(f"[DEBUG] 调用LLM，总消息数: {len(lm_msgs)}")
                    ai_msg = llm.invoke(lm_msgs)
                    reply_msg = ai_msg.content
                    print(f"[DEBUG] LLM返回成功，回复长度: {len(reply_msg)}")
                except Exception as e:
                    print(f"[ERROR] LLM调用异常: {e}")
                    import traceback
                    traceback.print_exc()
                    self._json({'error': f'LLM调用失败: {str(e)}'}, 500)
                    return
                
                self._json({
                    'assistant_reply': reply_msg,
                    'resume_profile': profile.model_dump() if profile else None,
                    'weight_suggestion': None,
                })
                return
            except Exception as e:
                print(f"[ERROR] /api/chat_resume: {e}")
                self._json({'error': str(e)}, 500)
        elif p.path == '/api/recommend':
            try:
                use_llm = bool(payload.get('use_llm'))
                limit = int(payload.get('limit') or 10)
                resume_dict = payload.get('resume') or {}
                use_xgb = payload.get('use_xgb')
                min_score = float(payload.get('min_score') or 0.5)
                resume = None
                try:
                    resume = ResumeProfile.model_validate(resume_dict)
                except Exception:
                    resume = None
                missing = []
                if not resume or not str((resume.personal_info and resume.personal_info.current_city) or '').strip():
                    missing.append('所在城市')
                pt = (resume and resume.work_preferences and resume.work_preferences.position_type_name) or []
                if not pt:
                    missing.append('目标职位')
                sal = (resume and resume.work_preferences and resume.work_preferences.salary_expectation and resume.work_preferences.salary_expectation.min_annual_package)
                if not sal or float(sal) <= 0:
                    missing.append('期望年薪')
                if missing:
                    q = ''
                    for k in ['所在城市','目标职位','期望年薪']:
                        if k in missing:
                            q = '请告诉我所在城市（或心仪工作城市）' if k=='所在城市' else ('请告诉我目标职位，例如：Java开发、全栈工程师' if k=='目标职位' else '请告诉我期望年薪（例如30万）')
                            break
                    self._json({'jobs': [], 'is_complete': False, 'missing_fields': missing, 'assistant_reply': q})
                    return
                loader = JobDataLoader(JOBS_FILE)
                jobs = loader.to_dict_list()
                jm = JobMatcher(jobs)
                matched = jm.find_matches(resume, limit=limit, min_score=min_score, use_xgb=use_xgb) if resume else []
                def simplify(m):
                    return {
                        'company_name': m.get('company_name'),
                        'job_title': m.get('job_title'),
                        'position_type_name': m.get('position_type_name'),
                        'city': m.get('city'),
                        'salary': m.get('salary'),
                        'score': m.get('score'),
                        'reasons': m.get('reasons', []),
                        'job_raw': m.get('raw'),
                    }
                top_jobs = [simplify(x) for x in matched]
                reply_msg = None
                if use_llm:
                    if not API_KEY:
                        self._json({'error': 'API key missing'}, 500)
                        return
                    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=TEMPERATURE)
                    prompt = (
                        "你是一名智能求职推荐助手。根据我提供的结构化岗位列表，生成友好的中文推荐内容：\n"
                        "1) 先给出高匹配岗位推荐表（公司/岗位/匹配点/薪资/备注），\n"
                        "2) 再总结下一步行动建议。\n"
                        "表格请用 Markdown 表格格式。\n"
                    )
                    lm_msgs = [SystemMessage(content=prompt), HumanMessage(content=json.dumps({'jobs': top_jobs, 'limit': limit}, ensure_ascii=False))]
                    ai_msg = llm.invoke(lm_msgs)
                    reply_msg = ai_msg.content
                self._json({'jobs': top_jobs, 'assistant_reply': reply_msg})
                return
            except Exception as e:
                print(f"[ERROR] /api/recommend: {e}")
                self._json({'error': str(e)}, 500)
        elif p.path == '/api/resume_enhance':
            try:
                resume_text = (payload.get('resume_text') or '').strip()
                current_profile = payload.get('current_profile') or {}
                messages = payload.get('messages') or []
                
                if not resume_text:
                    self._json({'error': '简历文本不能为空'}, 400)
                    return
                
                # 使用专门的提示词来完善简历
                if not API_KEY:
                    self._json({'error': 'API key missing'}, 500)
                    return
                
                try:
                    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=TEMPERATURE)
                    
                    system_prompt = (
                        "你是一个简历助手，负责提取职位匹配相关的关键信息。"
                        "**重要**：不要询问姓名、联系电话、电子邮箱等个人隐私信息。"
                        "只关注：期望职位、工作城市、薪资期望、工作经验、技能等职业相关信息。"
                        "严格只输出JSON，且键使用如下结构："
                        "{"
                        "\"profile_update\": 对现有画像的增量更新对象，"
                        "\"missing_fields\": 缺失但需要补充的字段数组（不包括姓名/电话/邮箱），"
                        "\"is_complete\": 布尔值是否完整，"
                        "\"next_question\": 如果不完整，给出面向用户的下一条具体提问（仅关于职位匹配信息）"
                        "}"
                    )

                    user_prompt = (
                        f"当前简历文本:\n{resume_text}\n"
                        f"当前已有画像(JSON):\n{json.dumps(current_profile, ensure_ascii=False)}\n"
                        "请判断职位匹配相关的关键字段是否缺失（期望职位、城市、薪资、经验、技能等），并补充能从文本推断出的信息。"
                        "**不要**询问或提及姓名、电话、邮箱等个人联系方式。"
                        "如果职位匹配信息不完整，给出下一条提问并列出缺失字段；如果完整，设置is_complete为true并给出profile_update。"
                    )

                    lm_msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                    for m in messages:
                        try:
                            role = m.get('role'); content = m.get('content','')
                            if role == 'user': lm_msgs.append(HumanMessage(content=content))
                        except:
                            pass
                    response = llm.invoke(lm_msgs)
                    
                    try:
                        data = json.loads(response.content)
                        update = data.get('profile_update') or {}
                        missing = data.get('missing_fields') or []
                        is_complete = bool(data.get('is_complete'))
                        next_q = data.get('next_question') or ''
                        def _merge(dst, src):
                            if not isinstance(dst, dict) or not isinstance(src, dict):
                                return src or dst
                            out = dict(dst)
                            for k,v in src.items():
                                if isinstance(v, dict):
                                    out[k] = _merge(out.get(k, {}), v)
                                else:
                                    out[k] = v
                            return out
                        enhanced_profile = _merge(current_profile, update)
                        self._json({
                            'enhanced_profile': enhanced_profile,
                            'is_complete': is_complete,
                            'missing_fields': missing,
                            'assistant_reply': ("信息收集完成，可以开始推荐了" if is_complete else (next_q or "请补充缺失信息"))
                        })
                    except json.JSONDecodeError:
                        self._json({
                            'enhanced_text': response.content,
                            'is_complete': False,
                            'assistant_reply': '请完善简历信息'
                        })
                        
                except Exception as e:
                    print(f"[ERROR] 简历完善失败: {e}")
                    self._json({'error': f'简历完善失败: {str(e)}'}, 500)
                    
            except Exception as e:
                print(f"[ERROR] /api/resume_enhance: {e}")
                self._json({'error': str(e)}, 500)
                
        elif p.path == '/api/feedback':
            try:
                action = (payload.get('action') or '').strip()
                job = payload.get('job') or {}
                resume = payload.get('resume') or {}
                meta = {
                    'action': action,
                    'ts': datetime.now(timezone.utc).isoformat(),
                    'type': 'feedback',
                    'job': job,
                    'resume': resume
                }
                # 使用项目根目录的logs（不是web/logs）
                out = Path(__file__).parent.parent / 'logs'
                out.mkdir(parents=True, exist_ok=True)
                fp = out / 'recommend_events.jsonl'
                with fp.open('a', encoding='utf-8') as w:
                    w.write(json.dumps(meta, ensure_ascii=False) + '\n')
                self._json({'ok': True})
            except Exception as e:
                self._json({'error': str(e)}, 500)
        elif p.path == '/api/recommend_events':
            try:
                # 使用项目根目录的logs（不是web/logs）
                out = Path(__file__).parent.parent / 'logs'
                out.mkdir(parents=True, exist_ok=True)
                target = out / 'recommend_events.jsonl'
                op = (payload.get('op') or 'list').strip()
                def read_jsonl(fp: Path):
                    if not fp.exists():
                        return []
                    lines = fp.read_text('utf-8').splitlines()
                    data = []
                    for ln in lines:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            data.append(json.loads(ln))
                        except Exception:
                            pass  # 静默跳过解析失败的行
                    return data
                def write_jsonl(fp: Path, items):
                    with fp.open('w', encoding='utf-8') as w:
                        for it in items:
                            w.write(json.dumps(it, ensure_ascii=False) + '\n')
                def append_jsonl(fp: Path, item):
                    with fp.open('a', encoding='utf-8') as w:
                        w.write(json.dumps(item, ensure_ascii=False) + '\n')
                if op == 'list':
                    items = read_jsonl(target)
                    # 规范化数据格式
                    for idx, item in enumerate(items):
                        # 1. 确保有id
                        if 'id' not in item or not item.get('id'):
                            import hashlib
                            content_str = json.dumps({
                                'action': item.get('action'),
                                'job': item.get('job', {}).get('岗位名称'),
                                'company': item.get('job', {}).get('企业')
                            }, ensure_ascii=False, sort_keys=True)
                            item['id'] = hashlib.md5(content_str.encode()).hexdigest()[:16]
                        
                        # 2. 冷启动数据：label转action
                        if 'action' not in item and 'label' in item:
                            item['action'] = 'like' if item.get('label') == 1 else 'skip'
                            item['type'] = 'cold_start'
                        
                        # 3. 确保有type
                        if 'type' not in item:
                            item['type'] = 'feedback'
                        
                        # 4. 确保有ts（用于排序）
                        if 'ts' not in item:
                            item['ts'] = '2025-11-11T00:00:00Z'
                    
                    # 按时间倒序排列
                    items.sort(key=lambda x: x.get('ts', ''), reverse=True)
                    self._json({'items': items})
                    return
                if op == 'create':
                    ev = payload.get('event') or {}
                    ev['id'] = ev.get('id') or str(uuid.uuid4())
                    ev['ts'] = ev.get('ts') or datetime.now(timezone.utc).isoformat()
                    append_jsonl(target, ev)
                    self._json({'ok': True, 'id': ev['id']})
                    return
                if op == 'update':
                    ev = payload.get('event') or {}
                    eid = ev.get('id')
                    if not eid:
                        self._json({'error': 'missing id'}, 400)
                        return
                    items = read_jsonl(target)
                    new_items = []
                    found = False
                    for it in items:
                        if it.get('id') == eid:
                            merged = dict(it)
                            for k,v in ev.items():
                                merged[k] = v
                            new_items.append(merged)
                            found = True
                        else:
                            new_items.append(it)
                    if not found:
                        self._json({'error': 'not found'}, 404)
                        return
                    write_jsonl(target, new_items)
                    self._json({'ok': True})
                    return
                if op == 'delete':
                    eid = (payload.get('id') or '').strip()
                    print(f'[删除] 收到删除请求，ID: {eid}')
                    if not eid:
                        self._json({'error': 'missing id'}, 400)
                        return
                    items = read_jsonl(target)
                    print(f'[删除] 读取到 {len(items)} 条记录')
                    # 同样的id生成逻辑
                    import hashlib
                    new_items = []
                    deleted = False
                    for idx, it in enumerate(items):
                        item_id = it.get('id')
                        if not item_id:
                            # 生成临时id用于比较
                            content_str = json.dumps({
                                'action': it.get('action'),
                                'job': it.get('job', {}).get('岗位名称'),
                                'company': it.get('job', {}).get('企业')
                            }, ensure_ascii=False, sort_keys=True)
                            item_id = hashlib.md5(content_str.encode()).hexdigest()[:16]
                        
                        print(f'[删除] 第{idx}条 item_id={item_id}, 目标id={eid}, 匹配={item_id == eid}')
                        
                        if item_id != eid:
                            new_items.append(it)
                        else:
                            deleted = True
                            print(f'[删除] ✓ 找到并删除: {it.get("job", {}).get("岗位名称")}')
                    
                    if deleted:
                        write_jsonl(target, new_items)
                        print(f'[删除] ✓ 删除成功，剩余 {len(new_items)} 条记录')
                    else:
                        print(f'[删除] ✗ 未找到匹配的记录')
                    
                    self._json({'ok': True, 'deleted': deleted, 'remaining': len(new_items)})
                    return
                self._json({'error': 'unsupported op'}, 400)
            except Exception as e:
                self._json({'error': str(e)}, 500)
        elif p.path == '/api/xgb_ops':
            # XGBoost operation - 使用模块化API
            try:
                op = (payload.get('op') or 'export').strip()
                params = payload.get('params', {})
                
                # 导入模块化API
                from training.xgb_api import handle_xgb_ops
                
                # 调用统一接口
                result = handle_xgb_ops(op, params)
                
                # 返回结果
                if result.get('success'):
                    self._json(result, 200)
                else:
                    self._json(result, 400)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json({'error': str(e)}, 500)
        else:
            self._json({'error': 'Not Found'}, 404)


def run():
    server = HTTPServer(('127.0.0.1', 8002), Handler)
    print('Unified server listening on http://127.0.0.1:8002')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    run()
