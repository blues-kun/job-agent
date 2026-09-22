#!/usr/bin/env python3
"""共用编码约定的本机BGE/Qwen服务；可选adapter，模型变化使索引失效。"""
from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--pooling", choices=["auto", "cls", "last_token"], default="auto")
    args = parser.parse_args()
    if not 64 <= args.max_tokens <= 32768:
        raise ValueError("编码长度必须在64至32768之间")
    from job_agent.encoder import LocalEncoder
    encoder = LocalEncoder(args.model_dir, args.device, args.adapter, args.max_tokens, args.pooling)
    info = encoder.info

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def response(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.response(info if self.path == "/health" else {"error": "未知路径"}, 200 if self.path == "/health" else 404)

        def do_POST(self):
            if self.path != "/encode":
                return self.response({"error": "未知路径"}, 404)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2000000:
                    return self.response({"error": "输入大小不合法"}, 413)
                payload = json.loads(self.rfile.read(size))
                texts = payload["texts"]
                if not isinstance(texts, list) or not 1 <= len(texts) <= 64 or any(not isinstance(text,str) or len(text)>24000 for text in texts):
                    return self.response({"error": "单批最多64条文本，每条不超过24,000字符"}, 422)
                vectors = encoder.encode(texts, query=bool(payload.get("query"))).tolist()
                self.response({"vectors": vectors, **{key:info[key] for key in ["model","revision","encoder_contract_hash"]}})
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                self.response({"error": "无效请求"}, 422)
            except Exception:
                # 错误响应和日志不包含简历文本。
                self.response({"error": "编码失败，请检查模型与算力状态"}, 500)

    print(json.dumps(info, ensure_ascii=False), flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
