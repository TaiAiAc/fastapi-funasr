# Web Service

## 项目介绍

Web Service 是一个基于 FastAPI 的 Web 服务框架，集成了语音识别(ASR)和语音端点检测(VAD)功能，使用 FunASR 作为核心语音处理引擎。

## 环境要求

- Python >= 3.10
- PDM 依赖管理工具

## 安装依赖

### 1. 安装 PDM

```bash
# 安装 PDM
pip install pdm

# 验证安装
pdm --version
```

### 2. 安装项目依赖

在项目根目录下执行以下命令安装所有依赖：

```bash
# 使用 PDM 安装项目依赖
pdm install
```

## 启动服务

### 开发模式启动

在项目根目录下执行以下命令启动服务：

```bash
# 使用 PDM 启动（推荐）
pdm run start

# 或直接使用 Python 运行
python main.py
```

启动后，服务将在 0.0.0.0:8000 上运行，控制台会显示以下信息：
- 局域网访问地址
- 本机访问地址
- API 文档地址

### 生产环境启动

对于生产环境，建议使用以下方式启动：

```bash
# 使用 uvicorn 直接启动（生产模式）
uvicorn src:app --host 0.0.0.0 --port 8000 --workers 4
```

## 服务访问

服务启动后，可以通过以下地址访问：
- 根路径：http://localhost:8000/
- API 文档：http://localhost:8000/docs
- 备选 API 文档：http://localhost:8000/redoc

## 项目结构

```
├── src/
│   ├── __init__.py      # 应用主入口
│   ├── middleware/      # 中间件目录
│   ├── routes/          # 路由目录
│   ├── services/        # 服务层目录
│   └── utils/           # 工具函数
├── main.py              # 启动文件
├── pyproject.toml       # 项目配置和依赖
└── .env                 # 环境变量配置文件
```

## 配置说明

**注意**：修改 `.env` 文件后，需要重启服务以使配置生效。

## 常见问题

### 依赖安装失败

如果遇到依赖安装失败，可以尝试更新 pip 或使用镜像源：

```bash
# 更新 pip
pip install --upgrade pip

# 使用PDM安装依赖（使用镜像源）
pdm install -i https://pypi.tuna.tsinghua.edu.cn/simple
```