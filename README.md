# 极验 v4 点选验证码识别库

极验 v4 点选验证码识别库。支持 `word` / `icon` / `phrase` 三种题型，
三种题型都已在真实站点（链家/贝壳）验证通过。

纯推理只依赖 **onnxruntime + numpy + opencv**，不需要 torch / paddle。

---

## 实测成绩

数据来自真实站点人工标注（302 张 4 字题面 + 500 张旧数据）：

| 题型 | 指标 | 数值 | 耗时 | 方案 |
| --- | --- | --- | --- | --- |
| **word** | 点中 3 个框 | 真机验证通过 | ~0.1s | 掩膜 + 边缘 chamfer 形状匹配 |
| **icon** | 点中 3 个图标 | 真机验证通过 | ~0.1s | 同上 |
| **phrase（3 字）** | 词序全对 | 90.1%（263/292） | ~1s | 微调 OCR + n-gram/分类器融合 |
| **phrase（4 字）** | 单字 | **95.0%**（留出 240 字） | 1.3 ms/字 | 微调 PP-OCRv6-tiny（48×96） |
| **phrase（4 字）** | 整句 4 字全对 | **85.0%**（留出 60 张） | ~1s | 上面 + 谓宾结构语序（80.8%） |

真机实录（只截通过那一段，`docs/video/`）：

| 题型 | 题面 | 文件 |
| --- | --- | --- |
| phrase | 擦净餐桌 | `lianjia_pass.mp4` |
| word | 康师傅 | `lianjia_pass_word.mp4` |
| icon | 手势图标 | `lianjia_pass_icon.mp4` |

> 2026-10 极验改版：phrase 从 3 字变 4 字、字形换成「白色粗描边 + 饱和填充」，
> 旧模型单字从 97.8% 掉到 50%。本版本用 302 张新标注重训后回到 95%，
> 并重写了 4 字语序（谓宾结构先验）。旧版资料见 [`geetest_solver_old/`](geetest_solver_old/)。

---

## 安装

```bash
pip install onnxruntime opencv-python numpy scipy
pip install -e .
```

可选：`jieba`（仅训练语序模型时需要）。

---

## 用法

```python
import geetest_solver

solver = geetest_solver.GeetestSolver()

# word / icon 型（有 ques 小图）
result = solver.classification(bg_bytes, [q1, q2, q3], captcha_type="word")

# phrase 型（无 ques；4 字自动走 OCR + 谓宾语序）
result = solver.classification(bg_bytes)

print(result["type"])     # word / icon / phrase
print(result["points"])   # [(x, y), ...] 点击坐标（按点击顺序）
print(result["word"])     # phrase 型识别出的词
```

`captcha_type` 也可留 `"auto"`：无 ques 图 → phrase，有 ques → word/icon
（word 与 icon 走同一套形状匹配，区别只在题面图上是字还是图标）。

### 配合极验 load 接口

```python
import json, requests

data = requests.get("https://api.geetest.com/get.php", params={...}).json()
bg = requests.get("https://" + data["bg"]).content
ques = [requests.get("https://" + q).content for q in data["ques"]]

out = solver.classification(bg, ques)
# 用 out["points"] 依次点击 → 提交
```

---

## 三种题型

| 类型 | 题面 | 任务 |
| --- | --- | --- |
| `phrase` | 4 个艺术字，无题面图 | 认字 + 排语序（谓宾结构） |
| `word` | 3 张小字图 + 3 个 3D 字 | 找出题面里的字，按顺序点 |
| `icon` | 3 张小图标 + 3 个彩色图标 | 找出同样的图标，按顺序点 |

题型判断（实测文案）：

```
请按语序依次点击                  → phrase
请在下图依次点击,看不清可刷新       → icon
请在下图依次点击,看不清可刷新 康师傅  → word
```

⚠️ 坑：`看不清可刷新` 是提示语，不是题面词。正则一把抓会把 icon 误判成 word。

---

## 实现原理

### word / icon：掩膜 + 边缘 chamfer

站点把题面图换成 24×24 黑白小图、面板字形换成「粗彩色描边 + 3D」，
旧的二值 IoU 相似度矩阵几乎是平的（0.35~0.47，等于随机点）。现在：

```
1) 面板：白描边 / 高饱和彩色描边 → 掩膜 → 背景压 0
2) 两边 Canny 取边缘，归一化到 64×64
3) 对称 chamfer 距离（双向最近边缘距离）+ 旋转搜索
4) 匈牙利分配：第 i 个题面图 → 第 j 个框
```

代码：`geetest_solver/shape_chamfer.py`（`solver._solve_match` 首选它，
失败自动回退旧的二值 IoU）。

### phrase 4 字：谓宾结构语序

4 字题的短语都是 `动词 + (量词) + 宾语`（穿过路口 / 拎袋水果 / 拌盘黄瓜）。
旧方案（字符 n-gram + 六选一分类器）会把宾谓倒装排前面，真实题面只有 27.6%。
`order_pos.py` 直接对 24 个排列做句式解析：

```
+1.0  首字能作谓语        +1.0  第二字是量词       +1.0  尾部是双字名词
+0.8  前两字是双字动词    +2.6  状语+动+宾         +2.8  动+补+宾
-0.4  名词开头（宾谓倒装） -0.9  尾部不是词
```

真实 245 条题面上 **80.8%**（n-gram 34.3%、分类器 5.5%、旧融合 27.6%）。

### phrase 3 字：n-gram + 分类器融合

保留旧方案：字符 n-gram（`order_lm_small.pkl`，10.8MB）+ 六选一分类器
（`order_ranker.npz`，numpy 推理），融合权重 0.4，实测 89~90%。

---

## 模型文件

只列当前版本用到的（旧的都在 `geetest_solver_old/models/`）：

| 文件 | 体积 | 用途 |
| --- | --- | --- |
| `detector_v26.onnx` | 9.3 MB | YOLO26n 字框检测（NMS IoU 必须 0.15） |
| `ocr_tiny_w96.onnx` | 4.2 MB | 4 字新字形微调 OCR（48×96，单字 95%） |
| `ppocrv6_tiny_dict.txt` | 30 KB | tiny 字符表（6906 字，**必须配套**） |
| `order_lm_small.pkl` | 10.8 MB | 3 字字符 n-gram |
| `order_ranker.npz` | 5.1 MB | 3 字六选一分类器（numpy） |
| `jieba_dict.txt` | 4.8 MB | 谓宾先验的词表（词性 + 词频） |
| `query_lib.pkl` | 1.0 MB | 旧形状模板库（word/icon 回退用） |

两个历史坑：

1. **字符表必须配套**：tiny 用 6906 字表，用 18708 字表会全乱码（且不报错）；
2. **窄宽训练必须窄宽推理**：48×96 训的模型按 320 宽跑，96.1% → 88.9%。

---

## 性能

```
检测 ~6 ms + OCR 1.3 ms/字 + 语序 ~10 ms
单张 phrase 4 字：~60 ms（含模型加载后热身）
内存峰值：~460 MB（onnxruntime 主导）
```

---

## 局限

- 4 字语序 80.8%：剩下的错误多是标注本身有歧义（`好排队伍`/`排好队伍`），
  或双字名词内部顺序（`花茶`/`茶花`）——规则判不了，需要短语级语料统计；
- word / icon 单次成功率约 50~70%（形状匹配），失败可刷新重试；
- 训练/标注流程不在本仓库，见 `docs/TECHNICAL_REPORT_zh.md`。

---

## 许可

仅用于学习与研究。请遵守目标站点的服务条款。
