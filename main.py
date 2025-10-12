# run_server.py
import uvicorn
import socket

def get_lan_ip():
    """
    获取本机局域网IP地址
    """
    try:
        # 创建一个UDP套接字，但不实际连接任何地址
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接到一个外部地址，这不会实际发送数据
        s.connect(('8.8.8.8', 80))
        # 获取套接字的IP地址
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # 如果获取失败，返回回环地址
        return '127.0.0.1'


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8000
    lan_ip = get_lan_ip()

    # 打印服务信息
    print("=" * 60)
    print(f"FunASR Web Service 启动信息:")
    print(f"局域网访问地址: http://{lan_ip}:{port}/")
    print(f"本机访问地址: http://127.0.0.1:{port}/")
    print(f"API文档地址: http://{lan_ip}:{port}/docs")
    print(f"可选API文档: http://{lan_ip}:{port}/redoc")
    print("=" * 60)

    uvicorn.run(
        "src:app",  # 使用导入字符串格式
        host=host,
        port=port,
        reload=True,  # 开发模式自动重载
        log_level="info"
    )
