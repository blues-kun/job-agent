import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, Any
import sys

# ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import JOBS_FILE, API_KEY
from data_preprocess import JobDataLoader
from resume_extract import ResumeExtractor, ResumeStorage


def json_response(handler: BaseHTTPRequestHandler, obj: Dict[str, Any], status: int = 200):
    body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.send_header('Access-Control-Allow-Headers', 'Content-Type')
    handler.send_header('Content-Length', str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/jobs':
            try:
                loader = JobDataLoader(JOBS_FILE)
                data = loader.to_dict_list()
                json_response(self, {'jobs': data})
            except Exception as e:
                json_response(self, {'error': str(e)}, status=500)
        else:
            json_response(self, {'error': 'Not Found'}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get('Content-Length') or '0')
        raw = self.rfile.read(length).decode('utf-8') if length > 0 else '{}'
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}

        if parsed.path == '/api/chat_resume':
            try:
                # messages: [{role: 'user'|'assistant'|'system', content: str}]
                messages = payload.get('messages', [])
                resume_text = ''
                for m in messages:
                    if m.get('role') == 'user':
                        resume_text = m.get('content', '')
                # Extract resume profile via LLM
                positions_file = Path('position_dictionary.txt')
                allowed_positions = []
                if positions_file.exists():
                    with open(positions_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not (line.startswith('[') and line.endswith(']')):
                                allowed_positions.append(line)
                profile = None
                weights = None
                if API_KEY and resume_text:
                    extractor = ResumeExtractor(allowed_positions)
                    import asyncio
                    profile = asyncio.get_event_loop().run_until_complete(extractor.extract(resume_text))
                else:
                    # Fallback: 无API密钥时返回提示与默认权重
                    weights = {
                        'POSITION_TYPE_WEIGHT': 1.5,
                        'LOCATION_WEIGHT': 0.8,
                        'SALARY_WEIGHT': 1.0,
                        'EXPERIENCE_WEIGHT': 0.6,
                        'EDUCATION_WEIGHT': 0.5,
                        'SCHOOL_LEVEL_WEIGHT': 0.4,
                        'TEXT_SIMILARITY_WEIGHT': 0.9,
                        'TITLE_INTENT_WEIGHT': 0.3,
                    }

                # Simple weight suggestion based on profile
                if profile:
                    weights = {
                        'POSITION_TYPE_WEIGHT': 1.5,
                        'LOCATION_WEIGHT': 0.8 if not profile.personal_info.willingness_to_relocate else 0.2,
                        'SALARY_WEIGHT': 1.1,
                        'EXPERIENCE_WEIGHT': 0.6,
                        'EDUCATION_WEIGHT': 0.5,
                        'SCHOOL_LEVEL_WEIGHT': 0.4,
                        'TEXT_SIMILARITY_WEIGHT': 0.9,
                        'TITLE_INTENT_WEIGHT': 0.3,
                    }

                assistant_msg = '已分析你的简历信息，生成结构化档案与推荐的匹配权重。如需调整，请继续描述你的偏好，例如城市、期望薪资或职位类别。'
                if not API_KEY:
                    assistant_msg = '当前未配置模型密钥，已为你提供默认权重。你可以继续描述偏好（城市、薪资、职位）或在 .env 中配置 API_KEY 后再试。'
                reply = {
                    'assistant_reply': assistant_msg,
                    'resume_profile': profile.model_dump() if profile else None,
                    'weight_suggestion': weights,
                }
                json_response(self, reply)
            except Exception as e:
                json_response(self, {'error': str(e)}, status=500)
        else:
            json_response(self, {'error': 'Not Found'}, status=404)


def run(host='127.0.0.1', port=8001):
    server = HTTPServer((host, port), Handler)
    print(f"API server listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    run()
