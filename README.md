# music2gtp8score

输入一段音频，输出一份符合**你自己排版标准**的鼓谱 —— 直接生成 Guitar Pro 7/8 的 `.gp` 文件。

```
音频 ─► MSST 53轨分离 ─► kick/snare/hh/toms 各轨 ─► 逐轨 onset 检测 ─► 网格量化
     ─► 两声部记谱(你的排版规则) ─► alphaTab Gp7Exporter ─► 输出 .gp
```

## 各环节

| 环节 | 实现 |
|---|---|
| 分离 | [MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training)（MIT）+ mvsep 53 轨 BS-Roformer 权重；底鼓/军鼓/踩镲/通鼓**各自独立成轨**，转谱退化为 onset 检测 |
| 转谱 | 纯信号基线：librosa onset 检测 + 力度归一；通鼓按频谱峰值细分音高；接口可插拔 |
| 量化 | librosa 动态速度估计（八度归一化 70–150 BPM），吸附 16 分音符网格 |
| 记谱 | 构建 alphaTab 模型 JSON，应用 `config/layout.*.yaml` 排版规则 |
| 写谱 | [alphaTab](https://alphatab.net)（MPL-2.0）`Gp7Exporter` 导出 `.gp`（GP7/8 直接打开） |

## 排版标准

默认值从真实鼓谱实证归纳（见 [`config/layout.default.yaml`](config/layout.default.yaml)）：
纯五线谱无 TAB、镲/鼓两声部、每行 4 小节、普通小节线、隐藏力度记号、
全部普通音符（不判鬼音/滚奏）。复制一份改成你的标准，`--config` 指定。

## 安装

```bash
# Python 侧（转谱/量化/记谱）
pip install -e .[audio,dev]
# Node 侧（GP7 写出桥）
npm install --prefix tools
# 分离引擎（MSST 代码，模型权重自备）
git clone --depth 1 https://github.com/ZFTurbo/Music-Source-Separation-Training vendor/Music-Source-Separation-Training
```

分离需要：带 torch+CUDA 的 Python（如 MSST-GUI 的 `runtime/python.exe`）+
53 轨 ckpt/yaml。路径通过环境变量覆盖：`M2G_MSST_PYTHON` / `M2G_MSST_REPO` /
`M2G_MSST_YAML` / `M2G_MSST_CKPT`。

```bash
music2gtp8score --check        # 一键自检所有依赖
```

## 使用

```bash
music2gtp8score song.mp3 -o out/song --title 歌名
music2gtp8score --stems-dir out/song/stems/song -o out/song   # 复用已分离的轨
```

输出：`out/song/歌名.gp` + `out/song/stems/`（53 轨 WAV）。

## 已知边界（攻关中）

- 53 轨模型没有 crash/ride 独立轨 → v1 不写镲片（踩镲除外）；计划从 drums 残轨检测
- 恒速 tempo map，变速歌曲待做多段落
- 下拍对齐简化处理（以首个事件为小节起点）
- 技法全部按普通音符写（用户决策 v1）

## 评测

开发期用「标准答案」对拍：已有成谱的歌跑管线后对比小节数/速度/音符命中。

## 许可证

MIT（本仓库代码）。MSST 为 MIT，alphaTab 为 MPL-2.0；模型权重与输入音频的
权利由使用者自行负责。
