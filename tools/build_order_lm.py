"""从三字词表构建字符 n-gram 概率表，产出 models/order_lm_small.pkl。

为什么单独一个脚本
------------------
`order_lm_small.pkl` 是一张统计表，不是训练出来的神经网络：把语料里每个词
按字符切开，统计 uni / bi / tri 的出现次数，推理时按加一平滑算对数概率。
表是死的，所以体积只有 10.8MB，重跑一遍几十秒。

语料格式
--------
一行一个词，UTF-8，不需要词频：

    拌盘黄瓜
    拎袋水果
    ...

数据集仓库的 `text_corpus/three_char_words.txt` 就是这个格式。
可以传多个 --corpus，按顺序累加统计。仓库里那张表是用 LCCC 中文语料
加上 text_corpus 的三字词一起统计出来的，字符覆盖 11706 个；
只拿三字词重建也能跑，字符 3483 个，覆盖面窄一些。

用法
----
    python tools/build_order_lm.py \
        --corpus ../geetest-captcha-dataset/text_corpus/three_char_words.txt \
        --out models/order_lm_small.pkl

只保留出现次数不小于 --min-count 的 bi / tri，压体积；一元表全留。
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def build(corpora: list[Path], order: int = 3, min_count: int = 2) -> dict:
    uni: Counter = Counter()
    bi: dict[str, Counter] = defaultdict(Counter)
    tri: dict[tuple[str, str], Counter] = defaultdict(Counter)
    vocab: set[str] = set()

    n_lines = 0
    for corpus in corpora:
        for line in corpus.read_text(encoding="utf-8").splitlines():
            word = line.strip()
            if not word or word.startswith("#"):
                continue
            n_lines += 1
            padded = "^" + word + "$"
            for i in range(1, len(padded)):
                ch = padded[i]
                uni[ch] += 1
                vocab.add(ch)
                bi[padded[i - 1]][ch] += 1
                if i >= 2:
                    tri[(padded[i - 2], padded[i - 1])][ch] += 1

    # 低频的上下文直接丢掉，加一平滑本来就给了它们退路
    bi = {k: dict(v) for k, v in bi.items() if sum(v.values()) >= min_count}
    tri = {k: dict(v) for k, v in tri.items() if sum(v.values()) >= min_count}

    print(f"语料 {n_lines} 条，字符 {len(vocab)} 个，"
          f"bi {len(bi)} 组，tri {len(tri)} 组")
    return {"order": order, "min_count": min_count,
            "uni": dict(uni), "vocab": sorted(vocab), "bi": bi, "tri": tri}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, nargs="+",
                    help="一个或多个语料文件，按行读")
    ap.add_argument("--out", default="models/order_lm_small.pkl")
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()

    blob = build([Path(c) for c in args.corpus], min_count=args.min_count)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(blob, f, protocol=4)
    print(f"写出 {out}（{out.stat().st_size / 1048576:.1f}MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
