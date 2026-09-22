"""生成完全手写虚构的岗位 Excel，供公开仓库演示；不读取真实招聘数据。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from openpyxl import Workbook


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "完全手写虚构的演示岗位；不代表真实企业、招聘需求、薪资行情或当前在招职位。"
HEADERS = ["岗位名称", "企业", "岗位薪资", "岗位要求", "岗位职责", "岗位地址", "职位类型名称", "二级分类", "数据性质"]

# 这些记录由项目维护者手写；不来自招聘网站、用户简历或私有工作簿。
# 元组顺序：岗位名、薪资、要求、职责、区域、职类、大类。
DEMO_JOBS = [
    ("Python后端开发（应届演示）", "8-12K·13薪", "本科；接受应届生；熟悉Python和SQL，有接口开发项目经验。", "使用Python开发订单接口，编写SQL查询，完成单元测试和接口文档。", "南山区", "Python开发", "后端开发"),
    ("Python后端开发（中级演示）", "15-22K·14薪", "本科；要求3年以上后端开发经验；熟悉Python、MySQL和Redis。", "负责订单服务与缓存设计，排查慢SQL，维护自动化测试和监控。", "福田区", "Python开发", "后端开发"),
    ("Python平台架构师（资深演示）", "30-45K·16薪", "本科；要求7年以上开发经验；精通Python，熟悉Kubernetes和分布式系统。", "设计多租户任务调度平台，评审服务架构，负责容量分析和故障复盘。", "南山区", "架构师", "后端开发"),
    ("Java后端开发（应届演示）", "9-13K·13薪", "本科；接受应届生；熟悉Java和SQL；了解Spring Boot优先。", "使用Java实现库存接口，编写SQL和单元测试，维护接口文档。", "龙华区", "Java开发", "后端开发"),
    ("Java交易服务开发（演示）", "18-28K·14薪", "本科；要求3年以上开发经验；熟悉Java和Spring Boot，掌握MySQL或PostgreSQL。", "负责订单交易服务，设计数据库表和幂等接口，开展代码评审。", "南山区", "Java开发", "后端开发"),
    ("多语言后端开发（OR演示）", "12-18K·13薪", "本科；要求1年以上开发经验；熟悉Python或Java；掌握SQL。", "参与业务接口开发，根据现有服务选择Python或Java，编写SQL和接口测试。", "福田区", "后端开发", "后端开发"),
    ("云端业务开发（嵌套要求演示）", "16-24K·14薪", "本科；要求3年以上开发经验；熟悉（Python或Java）和（MySQL或PostgreSQL）；了解Docker优先。", "开发订单接口和数据库查询，维护容器构建流程，编写测试文档。", "宝安区", "后端开发", "后端开发"),
    ("数据分析师（应届演示）", "8-12K·13薪", "本科；接受应届生；掌握SQL和Excel；了解Python优先。", "整理业务数据，编写SQL报表，使用Excel核对指标，解释活动转化结果。", "福田区", "数据分析", "数据与算法"),
    ("营销数据分析师（演示）", "14-20K·14薪", "本科；要求3年以上数据分析经验；熟悉SQL和Python；有统计分析经验。", "分析营销活动转化，使用Python构建统计报表，评估指标变化并说明局限。", "南山区", "数据分析", "数据与算法"),
    ("数据工程师（演示）", "18-26K·14薪", "本科；要求3年以上开发经验；熟悉SQL以及Spark或Flink；了解Python优先。", "维护数据处理任务，核对数据质量，排查计算延迟，编写数据口径文档。", "龙岗区", "数据开发", "数据与算法"),
    ("推荐算法工程师（演示）", "25-38K·15薪", "硕士；要求3年以上算法经验；熟悉Python和PyTorch；有推荐系统评测经验。", "构建召回与排序实验，分析样本偏差，实施离线评测和消融实验。", "南山区", "推荐算法", "数据与算法"),
    ("前端开发（应届演示）", "8-13K·13薪", "本科；接受应届生；熟悉JavaScript、HTML和CSS；了解Vue或React。", "开发职位列表和筛选表单，维护组件交互，编写页面测试和使用说明。", "宝安区", "前端开发", "前端与测试"),
    ("React前端开发（演示）", "17-25K·14薪", "本科；要求3年以上前端开发经验；熟悉TypeScript和React。", "开发管理工作台，维护表格与图表组件，改进页面可访问性和加载性能。", "福田区", "前端开发", "前端与测试"),
    ("软件测试实习生（演示）", "180-220元/天", "大专及以上；在校生可申请；了解Python或Java；愿意学习软件测试。", "整理测试用例，复现接口缺陷，记录测试步骤，协助维护自动化脚本。", "龙华区", "测试开发", "前端与测试"),
    ("技术项目协调（演示）", "面议", "本科；要求2年以上项目协调经验；具备需求沟通和进度管理能力；不要求Python开发经验。", "协调产品与研发排期，跟进风险和验收记录，为Python开发团队整理需求文档。", "罗湖区", "技术项目管理", "产品与支持"),
    ("技术支持工程师（演示）", "10-16K·13薪", "大专及以上；要求1年以上技术支持经验；熟悉Linux和SQL；了解网络排障优先。", "收集故障现象，使用Linux日志和SQL定位问题，整理用户操作指南。", "龙岗区", "技术支持", "产品与支持"),
]


def checked_output(output: Path) -> Path:
    """拒绝相对路径、仓库、已有文件和路径中任意符号链接。"""
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("--output 必须为仓库外的绝对路径。")
    if ".." in output.parts:
        raise ValueError("输出路径不能包含 ..。")
    if output.suffix.lower() != ".xlsx":
        raise ValueError("输出文件必须使用 .xlsx 扩展名。")
    for item in (output, *output.parents):
        if item.is_symlink():
            raise ValueError("输出文件及其所有父目录都不能是符号链接。")
    resolved = output.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("演示文件必须放在仓库外。")
    if any((parent / ".git").exists() for parent in resolved.parents):
        raise ValueError("演示文件不能写入任何 Git 仓库。")
    if output.exists():
        raise FileExistsError("输出目标已经存在，禁止覆盖；请选择新的文件名。")
    return output


def create_demo_data(output: Path) -> dict:
    output = checked_output(output)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output = checked_output(output)
    book = Workbook()
    sheet = book.active
    sheet.title = "完全虚构演示岗位"
    sheet.append(HEADERS)
    for index, (title, salary, requirements, description, district, category, family) in enumerate(DEMO_JOBS, 1):
        sheet.append([title, f"虚构演示企业{index:02d}（不存在）", salary, requirements,
                      description, f"深圳市{district}（虚构地址，无实际办公地点）", category, family, DISCLAIMER])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            book.save(stream)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    finally:
        book.close()
    return {"output": str(output), "jobs": len(DEMO_JOBS), "note": DISCLAIMER}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="尚不存在的仓库外绝对 .xlsx 路径；父目录可自动创建")
    args = parser.parse_args()
    try:
        result = create_demo_data(args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"已生成 {result['jobs']} 条虚构岗位：{result['output']}")
    print(result["note"])
    print("启动平台前，将 JOB_AGENT_DATA 设为上面的绝对路径。")


if __name__ == "__main__":
    main()
