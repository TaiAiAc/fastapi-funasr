# FunASR Web Service

## 项目介绍

FunASR Web Service 是一个基于 FastAPI 和 FunASR 的语音识别 Web 服务，提供了高效、便捷的语音识别接口。

## 环境准备

### 系统要求
- Python >= 3.10
- 推荐使用 PDM 进行依赖管理

### 安装 PDM

```bash
# Windows 安装 PDM
pip install pdm

# 验证安装
pdm --version
```

## 项目依赖

主要依赖包及版本要求：
- funasr>=1.1.5
- modelscope>=1.10.0
- torch>=2.0.0
- torchaudio>=2.0.0
- fastapi>=0.119.0
- uvicorn[standard]>=0.37.0 (包含 websockets 和高性能依赖)
- python-multipart (用于解析 form-data 上传文件)
- numpy

## 安装依赖

在项目根目录下执行以下命令安装依赖：

```bash
# 使用 PDM 安装依赖
pdm install

## 启动服务

### 开发模式启动

在项目根目录下执行以下命令启动服务：

```bash
# 使用 Python 直接运行主文件
python main.py

# 或使用 PDM 运行
pdm run start
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

## API 接口说明

### 基础接口

- `GET /`：欢迎页面，返回欢迎信息
- `GET /hello`：测试接口，返回问候信息

### 语音识别接口

- `POST /recognition/asr`：语音识别接口，支持文件上传
- `WS /recognition/stream_asr`：流式语音识别 WebSocket 接口
- `WS /asr`：另一个语音识别 WebSocket 接口

## 项目结构

```
├── src/
│   ├── __init__.py      # 应用主入口
│   ├── middleware/      # 中间件目录
│   │   ├── __init__.py
│   │   ├── authentication.py    # 认证中间件
│   │   ├── error_handling.py    # 错误处理中间件
│   │   ├── ip_whitelist.py      # IP白名单中间件
│   │   ├── logging.py           # 日志中间件
│   │   └── middleware_manager.py # 中间件管理器
│   ├── routes/          # 路由目录
│   │   ├── __init__.py
│   │   ├── funasr.py            # FunASR相关路由
│   │   └── recognition.py       # 语音识别路由
│   └── utils/           # 工具函数
│       ├── __init__.py
│       └── logger.py            # 日志工具
├── main.py              # 启动文件
├── pyproject.toml       # 项目配置和依赖
└── static/              # 静态文件目录
    └── index.html
```

## 中间件功能

项目使用了多个中间件来增强服务功能：

1. **日志中间件**：记录请求和响应信息
2. **认证中间件**：处理请求认证逻辑
3. **错误处理中间件**：统一异常处理
4. **IP白名单中间件**：限制访问IP（默认开放所有IP）

## 配置说明

### IP白名单配置

如需配置IP白名单，请修改 `src/middleware/middleware_manager.py` 中的 `register_middlewares` 方法，设置 `whitelist` 参数：

```python
# 示例：只允许特定IP访问
whitelist = ["127.0.0.1", "192.168.1.100"]
# 如需开放所有IP，设置为 None
whitelist = None
```

## 常见问题

### 端口被占用

如果 8000 端口被占用，可以修改 `main.py` 中的 `port` 变量来更改端口：

```python
port = 8080  # 修改为可用端口
```

### 依赖安装失败

如果遇到依赖安装失败，可以尝试更新 pip 或使用镜像源：

```bash
# 更新 pip
pip install --upgrade pip

# 使用镜像源安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 开发指南

1. 添加新路由：在 `src/routes/` 目录下创建新的路由文件，并在 `src/routes/__init__.py` 中的 `register_routers` 函数中注册
2. 添加新中间件：在 `src/middleware/` 目录下创建新的中间件文件，并在 `src/middleware/middleware_manager.py` 中的 `register_middlewares` 方法中注册
3. 开发模式下，修改代码后服务会自动重启（通过 uvicorn 的 reload 功能）
