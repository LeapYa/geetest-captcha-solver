# 旧版资料（2026-05 ~ 2026-10）

> 这里是**改版前**的模型和文档，保留只为可追溯/回滚，**当前管线不依赖它们**。
>
> 目录内容：
> - `models/ocr_tiny.onnx` —— 旧 OCR（48×320，彩色描边字形，单字 50%）
> - `models/order_lm.pkl` —— 旧 3-gram 语言模型（35.6MB，现用 `order_lm_small.pkl`）
> - `models/inference.pdiparams|json|yml` —— paddle 推理模型（供对照，运行时不用）
> - `models/ocr_config.yml` —— paddle 推理配置
> - `models/ppocrv6_dict.txt` —— 18708 字旧字符表（**与 tiny 模型不配套，用错全乱码**）
> - `models/detector.pt` —— 旧 torch 检测器（现用 `detector_v26.onnx`）
> - `models/geetest.pkl` —— 旧模板库
>
> 下面的文档是当时的 README 原文，其中「实测成绩」等数字是 3 字时代的数据。

---

# geetest-solver

极验 v4 点选验证码求解库。支持 `word` / `icon` / `phrase` 三种题型。

## 实测成绩

全部数据来自真实人工标注（500 张验证码）：

| 题型 | 指标 | 数值 | 单张耗时 | 方案 |
| --- | --- | --- | --- | --- |
| **word** | 3 点全对 | **99~100%** | 0.1s | YOLO26n(ONNX) + 形状匹配 |
| **icon** | 3 点全对 | **99%** | 0.1s | YOLO26n(ONNX) + 形状匹配 |
| **phrase 3 字** | 词序全对 | **90.1%** | 1.0s | 微调 PP-OCRv6-tiny + n-gram/分类器融合 |
| **phrase 4 字** | 整句全对 | **85.0%** | 1.0s | **微调 OCR v3（48×96）+ 谓宾结构语序** |
| **phrase 4 字** | 单字 | **95.0%** | 1.3 ms/字 | 同上（留出 240 字） |
| **word / icon** | 点在框上 | 真实站点均验证通过（2026-11） | 0.1s | 掩膜+边缘 chamfer 形状匹配 |

> 2026-10 极验把 phrase 从 3 字改成 4 字、字形换成白粗描边，旧模型掉到 52%。
> 本仓库已随 302 张人工标注重训（`models/ocr_tiny_w96.onnx`），并新增 4 字语序
> 模块 `geetest_solver/order_pos.py`（谓宾结构先验，80.8%）。

> word/icon 的指标是「3 个点全部落在人工标注框内」的比例。
> phrase 是端到端（识别 + 语序全对）。

## 安装

```bash
pip install paddlepaddle torch torchvision ultralytics opencv-python numpy pyyaml scipy
pip install -e .
```

`phrase` 型需要 PaddleOCR 的模型定义和 CTC 解码器：

```bash
git clone -b v3.7.0 https://github.com/PaddlePaddle/PaddleOCR.git
export PYTHONPATH=$PYTHONPATH:$PWD/PaddleOCR
```

模型文件用 Git LFS 管理，克隆后需 `git lfs pull`。

## 用法

```python
import geetest_solver

solver = geetest_solver.GeetestSolver()

# word / icon 型（有 ques 小图）
result = solver.classification(bg_bytes, [q1_bytes, q2_bytes, q3_bytes])

# phrase 型（无 ques）
result = solver.classification(bg_bytes)

print(result["type"])     # "word" / "icon" / "phrase"
print(result["points"])   # [(x, y), ...] 点击坐标，按点击顺序
print(result["word"])     # phrase 型识别出的词
```

输入支持 `bytes`、`numpy.ndarray`、文件路径。

### 配合极验 load 接口

```python
import json
import requests
import geetest_solver

sess = requests.Session()
r = sess.get("https://gcaptcha4.geetest.com/load", params={
    "captcha_id": "8b417c4b2871c1335ea780f4b562fd64",
    "challenge": challenge,
    "client_type": "web",
    "lang": "zh-cn",
})
data = json.loads(r.text[r.text.index("(") + 1:r.text.rindex(")")])["data"]

BASE = "https://static.geetest.com/"
bg = sess.get(BASE + data["imgs"]).content
ques = [sess.get(BASE + q).content for q in (data.get("ques") or [])]

solver = geetest_solver.GeetestSolver()
result = solver.classification(bg, ques or None)
# result["points"] 就是点击坐标
```

## 三种题型

实测 45 次抽样（`captcha_id=8b417c4b2871c1335ea780f4b562fd64`）：

| 类型 | 抽样 | ques 小图 | 任务本质 |
| --- | --- | --- | --- |
| `phrase` | 19 | **无** | 认字 + 排语序 |
| `word` | 15 | 3 张 | 找出指定 3 个字 |
| `icon` | 11 | 3 张 | 找出指定 3 个图标 |

## 实现原理

### word / icon：形状匹配

```
背景图 + ques
    ↓
YOLO26n 检测所有框（重叠目标也能出框）
    ↓
每个框裁出 → 归一化成二值形状 64×64
    ↓
与 ques 形状做多角度 IoU（18 个角度取最大）
    ↓
匈牙利算法分配（一框一目标）
```

**关键：不识别文字。**

ques 是固定模板（word 203 个 / icon 39 个），背景里是艺术字。
归一化成二值形状后做 IoU，比 OCR 转文字再比对稳得多
（OCR 在 word 上只有 42%，错误会层层传递）。

形状归一化：

```python
def shape_of(img):
    if img.shape[2] == 4:          # ques：用 alpha
        m = (img[:,:,3] > 20)
    else:                           # 背景：高饱和 + 高亮度
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = ((s > 100) & (v > 60)) | (v > 170)
    # 裁紧 + 等比缩放 + 居中到 64×64
```

### word / icon：掩膜 + 边缘 chamfer（2026-11 换新）

站点把题面图换成 24×24 黑白小图、面板字形换成「粗彩色描边 + 3D」，
旧的二值 IoU 在真实题面上相似度矩阵几乎是平的（0.35~0.47 → 等于随机点）。

现在用 `geetest_solver/shape_chamfer.py`：

```python
# 1) 面板：白描边/高饱和彩色描边 → 掩膜 → 背景压 0
# 2) 两边 Canny 取边缘，归一化 64×64
# 3) 对称 chamfer 距离（双向最近边缘距离）+ 旋转搜索
# 4) 匈牙利分配 → 第 i 个题面图 → 第 j 个框
```

真实站点实拍（都只截通过那一段）：

| 题型 | 题面 | 视频 |
| --- | --- | --- |
| phrase | 擦净餐桌 | `docs/video/lianjia_pass.mp4`（10.5s） |
| word | 康师傅 | `docs/video/lianjia_pass_word.mp4`（8.7s） |
| icon | 手势图标 | `docs/video/lianjia_pass_icon.mp4`（7.8s） |

⚠️ 题型判断有个坑：面板文案是「请在下图依次点击,看不清可刷新」，
`看不清可刷新` 不能当题面词解析，否则 icon 会被误判成 word（本项目踩过）。

### phrase 4 字：语序用「谓宾结构先验」

4 字题的短语都是 `动词 + (量词) + 宾语`（穿过路口 / 拎袋水果 / 拌盘黄瓜）。
旧的 n-gram + 六选一分类器会把宾谓倒装排前面（真实题面只有 27.6%），
`order_pos.py` 直接对 24 个排列做句式解析：

```
+1.0  首字能作谓语      +1.0  第二字是量词      +1.0  尾部是双字名词
+0.8  前两字是双字动词  +2.6  状语+动+宾        +2.8  动+补+宾
-0.4  名词开头（宾谓倒装）   -0.9  尾部不是词
```

真实 245 条题面上 **80.8%**（n-gram 34.3%、分类器 5.5%、旧融合 27.6%）。

### phrase：OCR + 语序

```
背景图
    ↓ YOLO26n 检测          字数正确 98.7%
字框
    ↓ 微调 PP-OCRv6          单字识别 95.4%
字符
    ↓ 字符 3-gram            语序 90.1%
点击顺序
```

**预处理必须与训练严格一致**（错了会掉 49 个百分点）：

```python
# 等比缩放到高 48，右侧补 0 到宽 320，归一化到 [-1, 1]
h, w = crop.shape[:2]
rw = min(320, int(math.ceil(48 * w / h)))
r = cv2.resize(crop, (rw, 48))
r = r.astype("float32").transpose((2, 0, 1)) / 255.0
r -= 0.5
r /= 0.5
pad = np.zeros((3, 48, 320), dtype=np.float32)
pad[:, :, :rw] = r
```

## 模型文件

全部在 `models/`（LFS）：

| 文件 | 体积 | 用途 |
| --- | --- | --- |
| `detector_v26.pt` | **24.8 MB** | YOLO26n 字框检测器 |
| `ocr_tiny_w96.onnx` | **4.4 MB** | **当前推荐**：4 字新字形微调版（48×96） |
| `ocr_tiny.onnx` | **4.3 MB** | 旧版（48×320，识别旧字形） |
| `ocr_ppocrv6_tiny.pdparams` | 68.2 MB | 动态图权重（训练用） |
| `ppocrv6_tiny_dict.txt` | 30 KB | tiny 字符表（6,906 字） |
| `ocr_tiny_config.yml` | 3 KB | tiny 配置 |
| `ocr_ppocrv6.pdparams` | 221.9 MB | 微调 PP-OCRv6 medium（备用） |
| `ocr_config.yml` | 3 KB | PP-OCRv6 模型配置 |
| `ppocrv6_dict.txt` | 45 KB | 字符表（18708 字） |
| `order_lm_small.pkl` | **10.8 MB** | 字符 3-gram（压缩版，推荐） |
| `order_ranker_int8.pt` | **5.9 MB** | 六选一分类器（int8，推荐） |
| `order_lm.pkl` | 35.6 MB | 字符 3-gram（完整版，备用） |
| `order_ranker.pt` | 16.5 MB | 六选一分类器（fp32，备用） |
| `query_lib.pkl` | **1.0 MB** | 形状模板库（word 203 + icon 39） |
| `jieba_dict.txt` | 4.8 MB | 词典（语序命中奖励） |
| `detector.pt` | 6.2 MB | 旧版 v8n 检测器（备用） |

## YOLO26n vs YOLOv8n

| 指标 | v8n | **v26n** |
| --- | --- | --- |
| word 字数正确 | 87% | **95%** |
| word 3 点全对 | 95% | **100%** |
| icon 3 点全对 | 99% | 99% |
| 参数量 | 3.16M | **2.57M** |
| 体积 | 30.3MB | **24.8MB** |

v26n 参数更少、体积更小，但精度更高。

## 训练方法

模型全部自训，完整流程见数据集仓库：

- **字框检测**：YOLO26n，500 张人工标注（1500 框），100 epoch，imgsz 320
- **单字识别**：微调 PP-OCRv6-medium，5436 个单字，lr 0.0002，20 epoch
- **语序模型**：字符 3-gram，中文维基词条 17MB + jieba 词典 349K 词

## 性能

| 阶段 | 耗时 |
| --- | --- |
| YOLO 检测 | 30 ms |
| 形状匹配（3 目标） | 40 ms |
| OCR 识别（3 字，仅 phrase） | 150 ms |
| 语序判定 | <5 ms |

| 题型 | 单张总耗时 |
| --- | --- |
| word / icon | ~0.1s |
| phrase | ~1.0s |

首次调用需 15~20 秒加载模型（含预热）。建议常驻进程复用 solver 实例。

内存约 400 MB，可在 2 核 2G 服务器运行。

## 语序模型：n-gram + 分类器融合

### 实测成绩

在 47 个「字符集完全未出现在训练集」的 phrase 词上（**零泄漏**）：

| 方案 | Top-1 | Top-3 |
| --- | --- | --- |
| 字符 3-gram | 85.1% | 95.7% |
| 六选一分类器 | 80.9% | 97.9% |
| **融合（权重 0.4）** | **89.4%** | **100.0%** |

**端到端（292 张真实验证码）：77.1% → 81.5%。**

### 为什么融合

两个模型互补：

| 模型 | 优势 | 来源 |
| --- | --- | --- |
| n-gram | 常见搭配 | 183MB 语料统计的 133 万条 3-gram |
| 分类器 | 整体构词合理性 | 2.4 万词的排列规律 |

单独用分类器（80.9%）不如 n-gram（85.1%），融合后 89.4%。

```python
分数 = 0.6 × n-gram归一化 + 0.4 × 分类器归一化
```

两个分数量纲不同，先各自 min-max 归一化到 [0,1] 再加权。

### 分类器结构

```
输入：3 个字的 6 种排列（顺序随机，位置无关）
编码：共享的 6 层 Transformer backbone（dim=192，词表 8192）
池化：每个候选取最后有效位置的隐状态
输出：6 个 logits → 交叉熵
```

训练数据：23,783 个词（词表来自 LCCC 851 万行语料验证）。
模型 685 万参数，16.5MB。

### 数据泄漏的教训

第一次测出融合 **96.0%**，但发现 **85% 的测试词在训练集里**。
在分类器从未见过的词上重测，融合反而低于 n-gram。

修正方法：**按字符集划分**，同一组三个字的全部排列必须全在同一侧。

### 词表质量的关键作用

| 词表来源 | 词数 | 分类器 Top-1 |
| --- | --- | --- |
| jieba 词频≥100 | 6,419 | 65.5% |
| **LCCC 语料验证过** | **28,765** | **73.1%** |

jieba 的 13 万三字词里，只有 21.5%（28,249 个）在真实语料中出现过。
其余是「一万两」「一二九」这类数字组合。

**教训：词频不等于词的真实性，要在真实语料里验证。**

### 失败的尝试

| 手段 | 结果 |
| --- | --- |
| 冻结 backbone（只训分类头） | 60.8%，更差 |
| 微调 IME 语言模型（4.25M） | 61.0%，不如 n-gram |
| 降低学习率 + 余弦退火 | 无改善 |

复现代码见 `geetest_solver/train_ranker.py`、`order.py`、`ime_model.py`。


## OCR 档位选择

PP-OCRv6 提供三档，实测（真实人工框，876 个字）：

| 档位 | 单字准确率 | 速度 | 体积 | 字符表 |
| --- | --- | --- | --- | --- |
| **tiny** | **97.8%** | **21 ms/字** | **68.2 MB** | 6,906 字 |
| small | 97.5% | 66 ms/字 | 119.1 MB | 18,708 字 |
| medium | 99.1% | 219 ms/字 | 221.9 MB | 18,708 字 |

**推荐 tiny**：单字准确率只低 1.3 个百分点，但**端到端从 81.8% 涨到 90.8%**。

原因是 tiny 的 6,906 字精简字符表候选空间小得多。
medium 的 18,708 字里含大量形近字（「这」→「过」、「散」→「敏」），tiny 没有这些候选。

### 两个容易踩的坑

1. **输出层键名不同**：tiny 是 `head.ctc_head.fc2.bias`（两级），
   small/medium 是 `head.ctc_head.fc.bias`
2. **字符表不同**：tiny 必须用 `ppocrv6_tiny_dict.txt`（6,906 字），
   用 18,708 字的表会全错（实测准确率 0%）

### 静态图导出：68.2MB → 4.4MB

动态图的 `.pdparams` 含优化器状态，导出成静态推理图后只剩推理所需权重：

```python
m = paddle.jit.to_static(model, input_spec=[paddle.static.InputSpec([None,3,48,320])])
paddle.jit.save(m, "models/ocr_tiny_static/model")
# model.json       0.14 MB
# model.pdiparams  4.28 MB
```

| 指标 | 动态图 | **静态图** |
| --- | --- | --- |
| 模型文件 | 68.2 MB | **4.4 MB**（-94%） |
| OCR 内存增量 | ~150 MB | **14 MB**（-91%） |
| 单字准确率 | 97.8% | 97.8% |
| 速度 | 21 ms/字 | 26 ms/字 |

solver 自动优先加载 `models/ocr_tiny_static/`，没有则退回动态图。

### 内存实际情况

整体运行约 900MB，瓶颈不在 OCR 权重：

```
paddle 运行时          ~270 MB
静态图 predictor 增量     14 MB
YOLO26n (torch)        ~300 MB
numpy + opencv         ~120 MB
```

所以继续量化 OCR 权重对总内存影响不到 2%，静态图已是最优解。

## 语序模块轻量化

原始版本语序模块占 52.1MB，压到 **16.7MB（-68%）**，准确率不降反升。

| 版本 | n-gram | 分类器 | 合计 | 端到端 |
| --- | --- | --- | --- | --- |
| 原版 | 35.6 MB | 16.5 MB | 52.1 MB | 81.5% |
| **轻量版** | **10.8 MB** | **5.9 MB** | **16.7 MB** | **81.8%** |

### n-gram：提高计数阈值

3-gram 表里有 138 万项，其中大量只出现 1~2 次（噪声）。

```python
# 只保留 count >= 5 的项
tri = {ctx: {k: v for k, v in cnt.items() if v >= 5}
       for ctx, cnt in tri.items()}
```

| min_count | 项数 | 体积 | 纯 n-gram Top-1 | 融合 Top-1 |
| --- | --- | --- | --- | --- |
| 2（原版） | 1,380,694 | 23.0 MB | 87.2% | 89.4% |
| **5（推荐）** | **404,701** | **10.8 MB** | **87.2%** | **91.5%** |
| 10 | 181,437 | 8.0 MB | 83.0% | 87.2% |
| 20 | 84,518 | 6.7 MB | 83.0% | 85.1% |

**min_count=5 是最优平衡点**：体积减 53%，融合准确率反而从 89.4% 升到 91.5%。
低频 3-gram 本就是噪声，去掉后模型更稳。

### 分类器：int8 量化

二维权重按输出行对称量化，LayerNorm/bias 保持 fp32：

```
真实值[行, 列] = int8值[行, 列] × scale[行]
scale[行] = max(abs(权重[行])) / 127
```

| 版本 | 体积 | 分类器 Top-1 | 融合 Top-1 | 融合 Top-3 |
| --- | --- | --- | --- | --- |
| fp32 | 16.5 MB | 80.9% | 89.4% | 100.0% |
| **int8** | **5.9 MB** | **83.0%** | **89.4%** | **100.0%** |

**准确率零损失**（分类器还略高，量化有正则化效果）。

### 为什么不换更小的模型

试过换小架构，全部失败：

| 架构 | 参数 | Top-1 | 原因 |
| --- | --- | --- | --- |
| 6层192（保留） | 4.25M | 73.1% | 能加载官方预训练权重 |
| 4层128 | 1.85M | 55.6% | 维度不匹配，从零训练 |
| 2层128 | 1.45M | 60.4% | 同上 |

**官方预训练权重是关键。** 换架构就用不了它，从零训练在 2.4 万词上必然过拟合。

所以保留架构 + 量化，比换小架构效果好得多。

## 只依赖 onnxruntime + numpy + opencv

整套方案**不需要 torch，也不需要 paddle**。

| 组件 | 实现 | 体积 | 速度 |
| --- | --- | --- | --- |
| 字框检测 | YOLO26n → ONNX | 9.3 MB | 10 ms |
| 文字识别 | PP-OCRv6-tiny → ONNX | 4.3 MB | 3 ms/字 |
| 语序 n-gram | pickle（压缩版） | 10.8 MB | <1 ms |
| 语序分类器 | numpy 重写推理 | 5.1 MB | 15 ms |
| **合计** | | **35 MB** | **65 ms/张** |

峰值内存 464MB。

### 依赖

```bash
pip install numpy opencv-python onnxruntime
# 不需要 torch、ultralytics、paddlepaddle、paddleocr
```

### paddle2onnx 的 Windows 坑

导出 OCR 模型时在 Windows 上卡了很久：`paddle2onnx` 的 wheel 缺依赖 DLL，
2.1.0 / 2.0.2rc1 全部 `DLL load failed`，1.3.1 能加载但不认 Paddle 3.x 的 PIR 格式。

**在 WSL 里导出**（Linux 下正常）：

```bash
python3 -m pip install --user --break-system-packages paddlepaddle==3.3.1 paddle2onnx==2.1.0
python3 -c "import paddle2onnx; paddle2onnx.export(model_filename=..., params_filename=..., save_file=...)"
```

导出的 ONNX 单字准确率 97.8%，与 paddle 完全一致，速度快 8.7 倍。

### CTC 解码

PP-OCR 用 CTC 训练，输出 `(1, 40, 6906)`。解码规则：

1. 每帧取 argmax
2. 合并连续相同索引
3. 去掉 blank（索引 0）

字符表 `ppocrv6_tiny_dict.txt` 共 6905 行，加 blank 和 space 共 6906 项。
**用错字符表会全部乱码但不会报错**（实测：`马德里` → `挲佀早`）。

## 历史：零 torch 依赖

整套方案只需要 `numpy` + `opencv` + `onnxruntime` + `paddle`，**不需要 torch/ultralytics**。

| 组件 | 实现 | 体积 | 运行时内存 |
| --- | --- | --- | --- |
| 字框检测 | YOLO26n → ONNX | 9.3 MB | ~38 MB |
| OCR | PP-OCRv6 tiny 静态图 | 4.4 MB | ~14 MB |
| 语序 n-gram | pickle（压缩版） | 10.8 MB | ~40 MB |
| 语序分类器 | numpy 重写推理 | 5.1 MB | ~17 MB |

对比 torch 版：体积 47MB → 35MB，内存 828MB → 464MB，速度 130ms → 65ms。

### YOLO 的 NMS 阈值要调

YOLO26 是端到端设计，导出 ONNX 后输出里有**大量重复框**（同一目标 5~8 个）。
默认 NMS 阈值 0.45 时 word 字数正确率只有 91%，调到 **0.15** 升到 98%。

```python
CONF_THRES = 0.40
IOU_THRES = 0.15    # 不是常见的 0.45
```

### 语序分类器的 numpy 实现

6 层 Pre-LN Transformer（dim=192）用 numpy 重写，15ms/次。
权重存 npz 保留 int8，加载时反量化：

```python
np.savez_compressed("order_ranker.npz",
                    **{k.replace(".", "__"): v for k, v in sd.items()})
```

必须和官方实现对齐的三点：
1. Pre-LN（先 LayerNorm 再注意力）
2. 因果掩码
3. **GELU 用精确 erf 形式**，不是 tanh 近似

## 局限

1. **phrase 语序**：3-gram 对部分词判断失误（如「无意中」→「中意无」），
   端到端失败中约 60% 来自这里
2. **字符集**：OCR 字符表 18708 字，超出范围必错
3. **模板库**：word 203 / icon 39 个模板，新模板需重建库
4. **icon 单例失败**：95 张里 1 张只命中 1/3

## 许可

仅供研究使用。
