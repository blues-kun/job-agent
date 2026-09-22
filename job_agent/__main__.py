"""运行：python -m job_agent --port 8090。"""
import argparse
import uvicorn
from .api import create_app


def main():
    parser = argparse.ArgumentParser(description="启动本地岗位需求与简历工作台")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
