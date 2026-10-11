# Helmet Lab · YOLO11 安全帽检测平台

一个可本机运行的 Flask Web 平台：用户注册和登录后，上传图片或视频，
查看带检测框、类别和置信度的结果，下载文件，并管理自己的检测历史。
采用单体架构，Web 路由、数据库、推理服务和文件操作分别维护。

Web 功能已通过自动测试和浏览器检查。当前已接入本机训练的 SHWD 安全帽模型，
独立测试集 mAP50 为 85.22%；用户截图在阈值 0.50 下正确识别 2 顶安全帽与 2 颗裸头，
0.25 下仍存在底部文字区域误检。详见 [模型评估记录](docs/trained-model-evaluation.md)。
SHWD 数据准备与训练步骤见 [训练指南](docs/training.md)，
旧模型问题见 [截图排查](docs/accuracy-investigation.md)。功能测试不代表模型准确率验收。
详见 [验收记录](docs/verification.md)。

## 已实现功能

- 注册、登录、退出，30 天“记住我”；密码采用 Werkzeug scrypt 哈希。
- 图片点击/拖拽上传与预览，支持 JPG、JPEG、PNG、WEBP。
- 视频上传与预览，支持 MP4、MOV、AVI、MKV；逐帧检测并输出 H.264 MP4。
- 原始图片/视频、检测结果和 SQLite 历史记录自动保存。
- 结果页展示类别数量、耗时、置信度、媒体信息；图片展示每个目标的坐标。
- 工作台统计、最近检测、历史筛选、分页、详情、下载与删除。
- 每个用户只能访问自己的记录和文件；媒体不放入公开 static 目录。
- 文件大小/格式/解码校验、CSRF 防护、推理失败清理、日志和友好的错误页面。
- 每个应用进程只加载一个 YOLO 实例，多请求通过锁安全复用。

当前不实现实时摄像头、RTSP、Tracking、行为识别或管理员后台。

## 技术栈与环境

Python 3.11、Flask、Flask-SQLAlchemy、Flask-Login、Flask-WTF、SQLite、
Ultralytics、OpenCV、Pillow、Jinja2、原生 HTML/CSS/JavaScript。
视频功能需要系统安装 **ffmpeg（包含 libx264 编码器）**。

建议使用独立 Conda 环境 `helmet_platform`。
开发环境是 macOS Apple Silicon；设备自动选择优先 MPS，然后 CUDA，然后 CPU。
也可以在 `.env` 中显式设置 `YOLO_DEVICE=cpu`、`mps` 或 `cuda:0`。

## 克隆并启动

安装 Python 3.11 和 ffmpeg 后，在终端执行：

```bash
git clone https://github.com/Qanyting/baselineyolov11-helmetdetection.git
cd baselineyolov11-helmetdetection
conda create -n helmet_platform python=3.11 -y
conda activate helmet_platform
pip install -r requirements.txt
cp .env.example .env
python app.py
```

浏览器打开 [http://127.0.0.1:5000](http://127.0.0.1:5000)。创建自己的账户后登录。
仓库已包含约 5.2 MB 的 SHWD 训练权重 `models/best.pt` 及匹配说明。
账户、数据库、密钥和上传目录会在本机生成。视频功能需要可用的 ffmpeg。
macOS 可使用 `brew install ffmpeg` 安装；其他系统请通过系统包管理器安装。
已经安装依赖的本地项目只需激活环境后运行 `python app.py`。

## 训练工地安全帽模型（M1 / 8GB）

本机验收时已准备 SHWD 数据和通用预训练权重；这些训练资源不随仓库上传。
新克隆的项目按照 [训练指南](docs/training.md) 下载数据和通用预训练权重后，
关闭运行中的 Web 服务并执行：

```bash
conda activate helmet_platform
python scripts/train_helmet.py --check
python scripts/train_helmet.py
```

默认训练 2,000 张图片、验证 400 张，50 轮，`imgsz=512`、`batch=2`、`device=mps`。
训练由你启动，脚本不会自动替换平台旧模型。训练结果保存到
`runs/helmet_shwd_m1/weights/best.pt`；评估和安装步骤见 [训练指南](docs/training.md)。

## 在其他机器从零安装

```bash
conda create -n helmet_platform python=3.11 -y
conda activate helmet_platform
pip install -r requirements.txt
cp .env.example .env
```

也可在项目根目录运行 `conda env create -f environment.yml` 创建环境。
macOS 安装 ffmpeg：

```bash
brew install ffmpeg
ffmpeg -version
```

本仓库已包含可运行的安全帽模型，安装依赖后执行：

```bash
python app.py
```

数据库目录和表会自动创建，无需手动操作 SQLite。
`requirements.txt` 仅列实际直接依赖；`requirements-lock.txt` 保存本次 macOS/Python 3.11
验收环境的全部精确版本。需要复现该环境时，可运行 `pip install -r requirements-lock.txt`。

## 项目结构

```text
.
├── app.py                    # 本地启动入口（不启用自动重载）
├── config.py                 # 统一配置、路径和 .env
├── requirements.txt          # 运行依赖
├── requirements-dev.txt      # 测试依赖
├── requirements-lock.txt     # 已验证版本快照
├── environment.yml           # 全新 Conda 环境定义
├── .env.example
├── models/
│   ├── best.pt               # 仓库包含的 SHWD 二分类权重
│   ├── best.json             # 权重指纹、训练场景与评估说明
│   └── README.md
├── database/
│   ├── app.db                # 自动创建，私有数据；不提交 Git
│   └── .secret_key           # 本地开发随机密钥；不提交 Git
├── src/
│   ├── __init__.py           # Flask 应用工厂、初始化和错误处理
│   ├── extensions.py         # 数据库、登录和 CSRF 扩展
│   ├── models/               # User 与 DetectionRecord
│   ├── routes/
│   │   ├── auth.py           # 注册、登录、退出
│   │   ├── dashboard.py      # 首页与统计
│   │   ├── detection.py      # 图片/视频 Web 上传入口
│   │   └── history.py        # 历史、详情、媒体鉴权、删除
│   ├── services/
│   │   ├── detector.py       # 唯一 YOLO 实例、推理、标注
│   │   ├── image_service.py
│   │   ├── video_service.py  # 视频逐帧检测及 ffmpeg 转码
│   │   └── errors.py
│   └── utils/                # 文件路径、上传和媒体验证
├── templates/                # 登录、注册、工作台、上传、结果、历史
├── static/css/main.css
├── static/js/main.js
├── storage/
│   ├── uploads/images/       # 原图，UUID 命名
│   ├── uploads/videos/       # 原视频，UUID 命名
│   ├── results/images/       # 标注 JPEG
│   └── results/videos/       # 标注 MP4 与原视频 MP4 预览
├── logs/app.log              # 轮转日志；不提交 Git
├── docs/                     # 设计说明和验收结果
└── tests/                    # 权限、上传、失败清理及真实模型验收
```

## 配置

编辑 `.env` 后需要重启。路径相对项目根目录，也支持绝对路径。

| 配置 | 默认值 | 说明 |
|---|---|---|
| `SECRET_KEY` | 首次随机生成 | 留空时自动保存在 `database/.secret_key`，保证重启后 Cookie 仍有效 |
| `DATABASE_URI` | `sqlite:///database/app.db` | SQLite 数据库路径 |
| `MODEL_PATH` | `models/best.pt` | 训练权重 |
| `UPLOAD_FOLDER` / `RESULT_FOLDER` | `storage/uploads` / `storage/results` | 私有媒体存储根目录 |
| `MAX_UPLOAD_MB` | `50` | 单个 HTTP 上传请求的总大小上限（含 multipart 开销） |
| `MAX_IMAGE_PIXELS` | `25000000` | 图片/视频分辨率像素上限 |
| `MAX_VIDEO_SECONDS` | `600` | 视频时长上限 |
| `MAX_VIDEO_FRAMES` | `18000` | 帧数上限，与时长限制同时生效 |
| `YOLO_CONF` / `YOLO_IOU` | `0.25` / `0.7` | 默认置信度、NMS IoU |
| `YOLO_IMGSZ` | `640` | 模型输入尺寸 |
| `YOLO_DEVICE` | `auto` | 自动或显式推理设备 |
| `FFMPEG_BINARY` | `ffmpeg` | ffmpeg 命令或完整路径 |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | 本地监听地址和端口 |
| `COOKIE_SECURE` | `false` | HTTPS 部署时改为 `true` |

本地随机密钥不是固定生产密钥。部署到 HTTPS 时设置独立 `SECRET_KEY` 与
`COOKIE_SECURE=true`，并使用 WSGI 服务器。`python app.py` 用于本地学习与开发。

## 使用说明

1. 注册一个 3–32 字符的用户名和 8–128 字符的密码，登录时按需勾选“记住我”。
2. 从工作台或侧栏进入图片/视频检测，拖入文件或点击选择。
3. 确认预览与文件大小，按需调整置信度阈值，点击“开始检测”。
4. 等待完成后自动进入结果页，查看原文件、检测结果及数量统计。
5. 在历史中按类型筛选，打开详情、下载或删除自己的记录。

退出登录使用 POST 并校验 CSRF；密码不存入 Cookie、session 或 localStorage。
未登录访问检测、历史或媒体地址时会跳转登录。

### 图片检测

服务器检查扩展名与真实图片格式是否一致，解码前验证尺寸和完整性；
按 EXIF 修正方向后推理，保存带框线、类别、置信度的 JPEG。
原上传字节单独保留。各类别从 `model.names` 动态读取。
没有检测目标也会保存成功记录，页面给出调整阈值的建议。

### 视频检测

OpenCV 逐帧读取，复用同一个模型，统计各帧数量，并尽量保留原始分辨率和 FPS。
结果转为 H.264 / yuv420p / MP4，写入 faststart；原视频另生成兼容的 H.264 预览，
原上传文件仍完整保留用于下载。有音轨时，预览和结果尽量保留并编码为 AAC。
奇数尺寸视频会在编码时补齐到偶数尺寸；可变帧率输入按 OpenCV 返回的 FPS 输出为恒定帧率。

**视频数量是逐帧目标数量的累计，不是唯一人员数量。** 同一目标在多个帧中会被重复计数。
当前没有 Tracking 或去重。视频处理采用同步请求，长视频请保持页面打开；
有等待提示和防重复提交，但不显示实时百分比，不使用 Redis/Celery/WebSocket。

## 存储与失败处理

媒体通过 `/media/<记录ID>/<original|result|preview>` 鉴权后访问，支持视频字节范围请求。
路径由服务器生成 UUID，原始文件名只用于展示；所有权在数据库查询中限制。
上传失败、解码失败、推理异常、转码失败或数据库保存失败，会清理本次生成的文件。
删除时先重命名暂存文件，数据库操作失败则恢复；成功后同时移除记录和文件。
详细异常仅写日志，不在普通页面显示 Traceback。

数据库保存完整的动态类别数量、图片检测框、推理/处理耗时和视频元数据。
时间以 UTC 存储，在界面中按 Asia/Shanghai 显示。

## 自动测试

```bash
conda activate helmet_platform
pip install -r requirements-dev.txt
python -m pytest -q
```

普通测试使用隔离的临时 SQLite/存储和确定性的推理替身验证 Web 行为，
视频管线仍调用真实 OpenCV 和 ffmpeg。真实 YOLO 全流程需提供本地图片：

```bash
HELMET_TEST_IMAGE=/absolute/path/to/helmet-image.jpg python -m pytest -q
```

存在 `models/best.pt` 且设置该变量后，会实际检测图片并由同一图片生成短视频，
通过 Web 上传、推理、结果保存、数据库记录和页面读取验收。
所有自动测试使用临时数据目录，不覆盖实际用户数据。
无变量时真实推理测试会明确跳过，其他测试仍执行。

## 常见问题

**提示 `YOLO model not found`**：将自己的训练模型放在 `models/best.pt` 或更新 `MODEL_PATH`，重启。
没有权重时 Web 页面仍可打开，检测按钮不可用；不会偷偷下载通用模型替代安全帽模型。

**`ModuleNotFoundError`**：确认 `conda activate helmet_platform`，重新安装 `requirements.txt`。
可通过 `python -c "import sys; print(sys.executable)"` 确认解释器路径。

**视频无法处理**：执行 `ffmpeg -version` 和 `ffmpeg -encoders`，确认存在 libx264。
检查视频未损坏，符合 50 MB、600 秒和 18000 帧限制。某些受 DRM 保护的文件无法解码。

**视频上传前无法预览**：某些 AVI/MKV 编码不受浏览器支持，可以正常上传；
检测后会生成 MP4 原视频预览和结果视频。若结果仍报错，查看 `logs/app.log`。

**MPS 推理报错或资源不足**：设置 `YOLO_DEVICE=cpu` 并重启，或缩小图片/视频分辨率。
首次推理包含设备初始化耗时，通常比后续推理慢。

**5000 端口占用**：在 `.env` 设置 `PORT=5001`，然后访问对应端口。
macOS 的系统服务也可能占用 5000 端口。

**页面验证过期**：刷新页面后重新选择文件或再次提交表单。

**删除后无法找回文件**：删除会同时清除记录与对应媒体，界面先弹出确认框；需要长期留存时先下载。

**备份与迁移**：停止应用后一起备份 `database/` 和 `storage/`，并保留 `.env` 中的密钥配置。
不要只复制数据库而丢失媒体目录。当前自动创建表适用于全新数据库，后续修改表结构需迁移。

## 扩展入口

新推理任务可以继续复用 `HelmetDetector.detect_frame()` 返回的结构化结果；
摄像头、行为识别或 Tracking 适合另建服务和路由。当前不包含这些扩展。

界面按照任务书并使用 [frontend-design Skill](https://github.com/anthropics/skills/tree/main/skills/frontend-design)
设计，不依赖前端框架、远程字体或 UI CDN。
技术接口参考 [Flask 文件上传文档](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/) 和
[Ultralytics Predict 文档](https://docs.ultralytics.com/modes/predict/)。
