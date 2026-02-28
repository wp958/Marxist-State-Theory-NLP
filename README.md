# Marxist-State-Theory-NLP
# 🏛️ 基于词汇共现网络的马克思主义国家学说历时性演进研究
> **Digital Humanities Project**: A Text-Mining Analysis of Marxist State Theory 
> **作者**：王潘 | **哲学 + 数字人文交叉研究**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Jieba-green.svg)
![NetworkX](https://img.shields.io/badge/Network_Analysis-NetworkX-orange.svg)

## 📖 项目简介
本项目运用**自然语言处理（NLP）**和**词汇共现网络（Co-occurrence Network）**，对马克思、恩格斯及列宁在四个历史阶段的海量经典著作进行大规模文本挖掘。通过定量分析“国家”这一核心概念的演变，直观揭示了马克思主义国家学说从“哲学思辨”走向“政治经济学解剖”，再到“革命政权建构”的范式转移。

## 🎯 核心技术栈
- **文本预处理**：`Jieba` 分词 + 领域自定义实体词典 + 停用词降噪
- **核心算法**：基于 `±15` 动态对称滑动窗口提取词汇共现矩阵
- **复杂网络与可视化**：`NetworkX` 拓扑计算 + `Matplotlib/Seaborn` 数据可视化

---

## 📊 核心数据图谱展示

### 1. 概念网络演进：从“哲学批判”到“实体解剖”
> 早期批判阶段（市民社会/异化） ➡️ 成熟阶段（资产阶级/国家机器） ➡️ 实践阶段（公社/打碎） ➡️ 继承阶段（苏维埃/专政）

<img width="1400" height="1200" alt="早期批判阶段_network_window15" src="https://github.com/user-attachments/assets/79807a91-c5b9-4972-97ce-72a05488fb2d" />

<img width="1400" height="1200" alt="成熟阶段_network_window15" src="https://github.com/user-attachments/assets/e8f54800-7fad-4905-a075-8f6ee8565a18" />

<img width="1400" height="1200" alt="实践应用阶段_network_window15" src="https://github.com/user-attachments/assets/9f189483-acff-46cc-8015-bb41f8d1b798" />

<img width="1400" height="1200" alt="继承和发展阶段_network_window15" src="https://github.com/user-attachments/assets/5de9051b-89fc-42ad-8298-977fdc782592" />

### 2. 宏观趋势：“阶级性”与“革命性”的恒定锚点
> 数据证实：“阶级”与“革命”在四个阶段始终保持最高频次；而“民主”一词在后期急剧攀升，展现了无产阶级对新型政体（苏维埃）的积极建构。
<img width="1300" height="800" alt="overall_lines_top8_window15" src="https://github.com/user-attachments/assets/c9b3717d-ef4b-4c2a-b83b-5c9ce2cfe419" />


### 3. 六大理论维度的范式转移
> “异化”维度在第一阶段后迅速衰减，而“暴力性”与“民主性”在最终阶段显著增强。
<img width="3600" height="2100" alt="theory_dimension_evolution_window15" src="https://github.com/user-attachments/assets/382dc44e-867f-42ac-84f0-107de6e40ec2" />

---

## 💡 总结
本项目通过可验证的 Python 代码，从底层词汇结构为宏观思想史演进提供了实证依据。完整的分析代码与原始论文已上传至本仓库。
