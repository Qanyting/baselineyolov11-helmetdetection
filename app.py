import os

from src import create_app

app = create_app()

if __name__ == "__main__":
    # 禁用自动重载，避免开发服务器父/子进程重复初始化模型。
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)
