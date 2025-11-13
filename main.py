"""
智能求职 Agent 主程序
"""
import asyncio
import json
import time
from pathlib import Path
from typing import List

from config import BASE_DIR, JOBS_FILE, RESUME_FILE, POSITION_DICT_FILE, SYSTEM_PROMPT
from data_preprocess import JobDataLoader
from resume_extract import ResumeExtractor, ResumeStorage
from search import JobMatcher
from agent import JobAgent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


def load_position_dictionary(file_path: Path) -> List[str]:
    """加载职位类型字典"""
    positions = []
    
    if not file_path.exists():
        print(f"警告: 职位字典文件不存在: {file_path}")
        return positions
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和二级分类标题（[...]格式）
            if line and not (line.startswith('[') and line.endswith(']')):
                positions.append(line)
    
    print(f"[OK] 已加载 {len(positions)} 个职位类型")
    return positions


def read_resume_input() -> str:
    """读取简历输入"""
    print("\n" + "="*80)
    print("简历输入")
    print("="*80)
    print("请选择输入方式:")
    print("1. 提供简历文件路径")
    print("2. 直接粘贴简历文本")
    
    choice = input("\n选择 (1/2): ").strip()
    
    if choice == '1':
        path = input("请输入简历文件路径: ").strip()
        if Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print("文件不存在，请直接粘贴简历文本")
    
    print("\n请粘贴简历文本，输入 /done 结束：")
    lines = []
    while True:
        line = input()
        if line.strip() == '/done':
            break
        lines.append(line)
    
    return '\n'.join(lines)


async def initialize_system():
    """初始化系统"""
    print("\n" + "="*80)
    print("智能求职 Agent 系统初始化")
    print("="*80)
    
    # 1. 加载职位字典
    print("\n[1/3] 加载职位字典...")
    allowed_positions = load_position_dictionary(POSITION_DICT_FILE)
    
    # 2. 加载岗位数据
    print("\n[2/3] 加载岗位数据...")
    loader = JobDataLoader(JOBS_FILE)
    jobs_data = loader.to_dict_list()
    
    # 3. 创建组件
    print("\n[3/3] 初始化组件...")
    resume_storage = ResumeStorage(RESUME_FILE)
    job_matcher = JobMatcher(jobs_data)
    
    print("\n[OK] 系统初始化完成")
    
    return allowed_positions, resume_storage, job_matcher


async def extract_resume_if_needed(
    resume_storage: ResumeStorage,
    allowed_positions: List[str]
):
    """如果需要，提取简历"""
    if resume_storage.exists():
        print("\n[OK] 已存在简历文件")
        return
    
    print("\n" + "="*80)
    print("简历提取")
    print("="*80)
    
    # 读取简历文本
    resume_text = read_resume_input()
    
    if not resume_text.strip():
        print("错误: 简历内容为空")
        return
    
    # 提取结构化信息
    print("\n正在提取简历结构化信息...")
    extractor = ResumeExtractor(allowed_positions)
    resume_profile = await extractor.extract(resume_text)
    
    # 保存简历
    resume_storage.save(resume_profile)
    
    print("\n[OK] 简历已保存")
    print(f"  - 职位类型: {', '.join(resume_profile.work_preferences.position_type_name)}")
    print(f"  - 当前城市: {resume_profile.personal_info.current_city}")
    print(f"  - 期望薪资: {resume_profile.work_preferences.salary_expectation.min_annual_package//10000}万+")


async def run_conversation(agent: JobAgent):
    """运行对话循环（参考 hello-uv 的非流式方式）"""
    print("\n" + "="*80)
    print("开始对话")
    print("="*80)
    print("提示：")
    print("  - 输入 /get_resume 查看简历")
    print("  - 输入 /quit 退出")
    print("  - 直接提问或请求推荐岗位")
    print("="*80)
    
    # 构建消息历史
    history = []
    history.append(SystemMessage(content=SYSTEM_PROMPT))
    
    # 添加当前简历作为上下文
    resume = agent.resume_storage.get()
    if resume:
        history.append(HumanMessage(content=f"当前简历（供参考）：\n{json.dumps(resume.model_dump(), ensure_ascii=False)}"))
        history.append(AIMessage(content="收到，我会根据需要调用工具进行岗位匹配。"))
    
    conversation_log = []
    
    # 使用 bind_tools 而不是 create_agent（参考 hello-uv）
    llm_with_tools = agent.model.bind_tools(agent.tools)
    
    while True:
        user_input = input("\n你: ").strip()
        
        if not user_input:
            continue
        
        # 特殊命令
        if user_input == '/quit':
            print("\n[系统] 正在退出...")
            log_file = BASE_DIR / f"conversation_{int(time.time())}.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_log, f, ensure_ascii=False, indent=2)
            print(f"[系统] 对话已保存: {log_file}")
            break
        
        if user_input == '/get_resume':
            if resume:
                print(json.dumps(resume.model_dump(), ensure_ascii=False, indent=2))
            else:
                print("[系统] 简历未加载")
            continue
        
        # 记录用户输入
        history.append(HumanMessage(content=user_input))
        conversation_log.append({
            "role": "user",
            "content": user_input,
            "timestamp": int(time.time())
        })
        
        # 手动工具调用循环（参考 hello-uv/job_agent.py）
        try:
            ai_msg = await llm_with_tools.ainvoke(history)
            
            # 工具调用循环
            while getattr(ai_msg, "tool_calls", None):
                history.append(ai_msg)
                conversation_log.append({
                    "role": "assistant",
                    "content": ai_msg.content or "",
                    "tool_calls": str(ai_msg.tool_calls)[:200],
                    "timestamp": int(time.time())
                })
                
                # 执行每个工具调用
                for call in ai_msg.tool_calls:
                    name = call.get("name")
                    args = call.get("args", {})
                    result_text = ""
                    
                    try:
                        # 找到对应的工具并执行
                        tool_func = None
                        for t in agent.tools:
                            if t.name == name:
                                tool_func = t
                                break
                        
                        if tool_func:
                            result_text = tool_func.invoke(args)
                        else:
                            result_text = f"[unknown tool] {name}"
                    except Exception as e:
                        result_text = f"[tool error] {name}: {e}"
                    
                    history.append(ToolMessage(content=result_text, tool_call_id=call.get("id", name)))
                    conversation_log.append({
                        "role": "tool",
                        "name": name,
                        "content": result_text[:500] if len(result_text) > 500 else result_text,
                        "timestamp": int(time.time())
                    })
                
                # 再次调用模型处理工具结果
                ai_msg = await llm_with_tools.ainvoke(history)
            
            # 无更多工具调用，输出最终回复
            print(f"\n助理: {ai_msg.content}")
            history.append(ai_msg)
            conversation_log.append({
                "role": "assistant",
                "content": ai_msg.content,
                "timestamp": int(time.time())
            })
            
        except Exception as e:
            print(f"\n[错误] {e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数"""
    try:
        # 初始化系统
        allowed_positions, resume_storage, job_matcher = await initialize_system()
        
        # 提取简历（如果需要）
        await extract_resume_if_needed(resume_storage, allowed_positions)
        
        # 创建 Agent
        print("\n正在创建 Agent...")
        agent = JobAgent(resume_storage, job_matcher)
        print("[OK] Agent 已就绪")
        
        # 运行对话
        await run_conversation(agent)
        
    except KeyboardInterrupt:
        print("\n\n[系统] 用户中断")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

