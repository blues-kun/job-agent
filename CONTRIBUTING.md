# 贡献指南

感谢您对本项目的关注！

## 如何贡献

### 报告 Bug
- 请在 Issues 中详细描述问题
- 提供复现步骤和环境信息（Python 版本、操作系统等）
- 如有可能，附上截图或错误日志

### 提出新功能
- 在 Issues 中描述您的想法
- 说明该功能的使用场景和预期效果

### 提交代码
1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 代码规范
- 遵循 PEP 8 Python 代码风格
- 添加必要的注释和文档字符串
- 保持代码简洁易读

## 开发环境配置
```bash
# 克隆仓库
git clone https://github.com/blues-kun/job_agent.git
cd job_agent

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp env_template.txt .env
# 编辑 .env 填写你的 API_KEY

# 启动服务
cd web
python unified_server.py
```

## 联系方式
如有任何问题，欢迎通过 blues924@outlook.com 联系我们。

---
版权所有 © 深圳大学 徐琨博

