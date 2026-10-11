# 在这台 M1 / 8GB Mac 上手动训练

数据准备时没有代用户启动训练。用户现已在终端完成 50 轮训练，
并已评估与接入新模型，见 [训练后评估记录](trained-model-evaluation.md)。
使用此前为本项目新建的 Conda 环境 `helmet_platform`，Python 3.11。
模型从 COCO 通用预训练 YOLO11n 开始，不使用旧的 BikesHelmets 微调权重继续训练。

## 数据与目录

来源是 [官方 SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)，
原始数据 7,581 张，包含安全帽和未戴帽头部。官方负样本中有来自 SCUT-HEAD 的普通人群，
所以它并非全部为工地照片；新场景的准确性仍需用测试集和你的图片检查。

原始 Pascal VOC 类别 `hat` → `0: helmet`；`person` → `1: no_helmet`。
此数据集的 `person` 是裸头框，不能当作完整人体框。
脚本验证图片解码、尺寸、类别和非零框，将越界框裁到图像边界，并记录修正数量。
官方测试集优先保留，精确相同的解码图像去重；训练、验证、测试之间没有精确重复图。
没有做近似重复图或人物身份去重，这仍是数据评估的局限。
全量类别审计发现 `000377.xml` 中 3 个 `dog` 标签实际上框住了安全帽区域。
已核对原图并剔除整张拼图，避免保留未标注目标或自行猜测修改标签；原始 ZIP 未改变。

训练/验证从官方 trainval 中固定 seed=42 划分；M1 配置再各选取约一半含安全帽图、
一半只有裸头的图。这是按图平衡，密集人群使裸头框数量仍多于安全帽框。
`m1.yaml` 和 `full.yaml` 共用官方测试集，测试图不会用于模型拟合或挑选最佳轮次。
你的 `IMG_2377.PNG` 不加入任何数据划分。

```text
datasets/
├── raw/VOC2028.zip            # 官方原始压缩包
├── raw/extracted/VOC2028/     # 原始 VOC 图片、XML、划分
└── shwd/
    ├── images/               # 去重后的图片
    ├── labels/               # YOLO 二分类标签
    ├── m1.yaml               # 默认：2,000 train、400 val、官方 test
    ├── full.yaml             # 完整划分，可另开训练
    ├── m1_{train,val,test}.txt
    ├── full_{train,val,test}.txt
    └── manifest.json         # 数据源、SHA-256、划分、数量、精确重复记录
models/pretrained/yolo11n.pt   # 约 2.6M 参数、COCO 80 类通用预训练权重
```

大型数据、缓存、训练输出和权重已排除 Git。
新克隆的项目先准备训练依赖和通用预训练权重：

```bash
pip install -r requirements-training.txt
python -c "from pathlib import Path; from ultralytics import YOLO; Path('models/pretrained').mkdir(parents=True, exist_ok=True); YOLO('models/pretrained/yolo11n.pt')"
python scripts/prepare_shwd.py
```

这会从 Ultralytics 下载通用预训练权重，并从官方 SHWD 来源下载和转换训练数据。
已准备数据和权重的本机可以直接使用后续命令。
重新准备数据可运行 `python scripts/prepare_shwd.py`，已有有效 ZIP 会复用。
本机实测划分数量和检查结果见本文末尾的准备记录。

## 启动训练

关闭运行中的 `python app.py`，避免它与训练共用内存。
在终端执行以下命令；`--check` 只加载权重、校验参数并扫描三个划分，不训练：

```bash
cd baselineyolov11-helmetdetection
conda activate helmet_platform
python scripts/train_helmet.py --check
python scripts/train_helmet.py
```

若终端提示 `conda activate` 未初始化，可以先执行 `source "$(conda info --base)/etc/profile.d/conda.sh"`。

默认参数是根据 M1 / 8GB 选择的保守起点：

| 参数 | 值 | 作用 |
|---|---|---|
| 模型 | YOLO11n | 较小的参数规模 |
| epochs | 50 | 最多 50 轮，验证指标连续 12 轮无改善会提前停止 |
| imgsz | 512 | 降低训练内存需求；小头部辨识可能弱于 640 |
| batch | 2 | 每批 2 张，避免较大的内存占用 |
| device | mps | 使用 Apple Silicon GPU |
| workers | 0 | 单进程读取数据 |
| cache / amp | False / False | 不将整套图片缓存到内存，不使用混合精度 |
| optimizer / lr0 | AdamW / 0.001 | 明确的迁移学习优化设置 |

脚本在导入 PyTorch 前设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`。
实际训练速度和峰值内存要等你开始训练后才能测量；**没有承诺的准确率或完成时间**。
观察前两轮耗时，再估算剩余时间。插电运行，避免电脑自动休眠。
需要防止空闲休眠时，可用 `caffeinate -i python scripts/train_helmet.py` 替代最后一条命令。

输出目录：

```text
runs/helmet_shwd_m1/
├── weights/best.pt           # 按验证指标选出的最佳权重
├── weights/last.pt           # 最近保存的权重和训练状态
├── args.yaml
├── results.csv
└── results.png               # 训练损失与验证指标曲线
```

同名目录存在时新训练会明确拒绝覆盖；可使用 `--name helmet_shwd_m1_v2` 开新一轮。
每轮结束保存 last，另外每 10 轮保存一次阶段权重。

## 内存不足与中断恢复

如果 batch=2 仍报 MPS out of memory，降低为 1，并开新目录：

```bash
python scripts/train_helmet.py --batch 1 --name helmet_shwd_m1_b1
```

已完成至少一轮并保存 `last.pt` 后，中断可以恢复：

```bash
python scripts/train_helmet.py --resume runs/helmet_shwd_m1/weights/last.pt --batch 1
```

恢复会读取 checkpoint 里的原数据、轮数和输出目录。当前批次的进度不会保存，
恢复从已完成的轮次继续。训练完成后不能用 resume 当作任意追加轮数。
MPS 持续报不支持的算子时，可以恢复到 CPU：

```bash
python scripts/train_helmet.py --resume runs/helmet_shwd_m1/weights/last.pt --device cpu --batch 1
```

完整数据适合在小规模流程跑通后尝试；会增加训练时间，并不自动保证更高准确率：

```bash
python scripts/train_helmet.py --data datasets/shwd/full.yaml --name helmet_shwd_full --batch 1
```

如果更改训练 run 名称，后续评估与安装的 `--model` 也必须使用该 run 的路径。

## 训练后评估

训练结束后先在独立的官方测试集上计算精确率、召回率、mAP50 和 mAP50–95，
并生成你的截图的可视化结果：

```bash
python scripts/evaluate_helmet.py \
  --model runs/helmet_shwd_m1/weights/best.pt \
  --image /path/to/your/image.png
```

终端会打印评估报告的绝对路径，默认首次在 `runs/helmet_test/evaluation.json`。
再次评估会用 `helmet_test2` 等新目录。报告保存总指标、分类指标和权重 SHA-256；
标注截图位于该评估目录下的 `sample/`。截图按 640 输入、阈值 0.25 推理，
便于与平台默认设置对比；测试集指标按训练的 512 输入计算。

人工对照图片里的 2 顶安全帽和 2 颗未戴帽头部，检查位置、类别和误检。
一张图片通过仍不能证明所有工地场景可靠，原始截图中的界面文字也可能造成干扰。
评估脚本不会改动图片原文件或平台模型。

## 安装新模型到平台

确认测试指标和截图表现达到你的使用要求后，停止 Web 服务，运行：

```bash
python scripts/install_helmet_model.py \
  --model runs/helmet_shwd_m1/weights/best.pt \
  --report runs/helmet_test/evaluation.json
python app.py
```

如果评估结果在 `helmet_test2`，将 `--report` 改成终端打印的实际路径。
安装脚本校验两类别和评估报告的权重指纹，先把原来的模型和说明备份到
`models/backups/<时间>/`，再替换 `models/best.pt` 并生成匹配的 `models/best.json`。
平台每个进程只加载一次模型，所以替换文件后必须重启。
历史记录中的旧检测结果保留，不会用新模型自动重算。

技术依据：[YOLO11 官方模型说明](https://docs.ultralytics.com/models/yolo11/)、
[Ultralytics 训练、MPS 与恢复文档](https://docs.ultralytics.com/modes/train/)、
[YOLO 数据格式](https://docs.ultralytics.com/datasets/detect/)。

## 本机准备记录（2026-10-06）

原始 ZIP 1,117,970,537 字节，SHA-256：

```text
2f6c5c347b778271bc8b27c2d61b731fc5fb8cdb4f45fce583dd2f82e6a46401
```

检查了全部 7,581 张图片，剔除 39 张精确重复图和 1 张错误标注图，保留 7,541 张。
全部图片的 XML 尺寸与实际尺寸一致，无需裁剪越界框。
64 张 JPEG 缺少结束标记，已在 YOLO 副本中以 quality=100、subsampling=0 重存修复；
原始 ZIP 与原始解压文件未修改。修复记录、剔除记录和去重记录保存于
`datasets/shwd/manifest.json`，精简摘要另存 [shwd-summary.json](shwd-summary.json)。
原项目 README 和 LICENSE 已保存在 `datasets/raw/SHWD_README.md` 与 `SHWD_LICENSE`。

| 配置 | 划分 | 图片 | helmet 框 | no_helmet 框 |
|---|---|---:|---:|---:|
| M1 默认 | train | 2,000 | 2,729 | 25,646 |
| M1 默认 | val | 400 | 557 | 4,818 |
| 完整 | train | 4,822 | 5,717 | 71,063 |
| 完整 | val | 1,205 | 1,389 | 17,680 |
| 两配置共用 | test | 1,514 | 1,876 | 22,538 |

已检查所有 YOLO 标签的五列格式、有效类别、有限数值、归一化范围与正面积。
两个配置各自的 train/val/test 图像哈希没有交集。
默认配置通过当前 Ultralytics 的真实数据扫描，读取 2,000 / 400 / 1,514 张，
均无损坏图片或标签；`--check` 明确输出 `No training has been started`。
完整配置也已通过真实扫描，读取 4,822 / 1,205 / 1,514 张，无损坏图片或标签。
标注抽样预览保存在 `test-artifacts/shwd-label-preview.jpg`：绿框为安全帽，红框为裸头。
抽样观察确认包含工地安全帽及正常头部，而非骑行头盔图片。

自动测试 40 项通过，Python 与 JavaScript 语法检查、`pip check` 通过。
为腾出内存，已停止由 Codex 启动的 Web 预览服务，用户数据保留。
训练、模型测试集指标及安装新模型均等待你在终端按本文命令运行。
