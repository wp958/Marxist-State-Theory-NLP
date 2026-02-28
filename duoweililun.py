import os
import pandas as pd
import matplotlib.pyplot as plt

# === 解决matplotlib中文显示问题 ===
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# === 配置 ===
RESULTS_ROOT_DIR = r"D:\zhuo mian\PythonProject2\cooccurrence_analysis_final"
TARGET_KEYWORD = "国家"
WINDOW_SIZE = 15

# 阶段名称
STAGES = [
    "早期批判阶段",
    "成熟阶段",
    "实践应用阶段",
    "继承和发展阶段"
]

# 理论维度关键词组
THEORY_DIMENSIONS = {
    "阶级性": [
        "阶级", "阶级斗争", "统治阶级", "资产阶级", "无产阶级", "上层建筑", "生产关系", "市民社会", "官僚机构", "官僚制度"
    ],
    "暴力性": [
        "暴力", "镇压", "国家暴力", "国家机器", "军队", "警察", "官僚机构", "官僚制度", "波拿巴主义"
    ],
    "革命性": [
        "革命", "打碎", "无产阶级专政", "政治解放", "波拿巴主义", "公社"
    ],
    "民主性": [
        "民主", "苏维埃", "委员会", "公共权力", "民主集中制", "人民", "廉价政府"
    ],
    "消亡论": [
        "国家消亡", "过渡时期", "半国家", "“半国家”", "暂时性", "不可调和的产物", "打碎"
    ],
    "异化/虚幻性": [
        "异化", "虚幻形式", "共同体", "市民社会", "意识形态", "（统治阶级的）共同利益的虚幻形式"
    ]
}

# 结果输出目录
OUTPUT_DIR = os.path.join(RESULTS_ROOT_DIR, "theory_dimension_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_stage_cooc_df(results_root, target_keyword, window_size, stage):
    stage_dir = os.path.join(
        results_root,
        f"target_{target_keyword}_window_{window_size}",
        stage
    )
    expected_filename = f"{stage}_cooccurrence_window{window_size}.csv"
    file_path = os.path.join(stage_dir, expected_filename)
    if os.path.isfile(file_path):
        print(f"读取文件：{file_path}")
        return pd.read_csv(file_path, encoding='utf-8-sig')
    else:
        print(f"未找到文件：{file_path}")
        return pd.DataFrame()

def calculate_theory_dimension_strengths(stage_cooc_dfs, theory_dimensions):
    result = pd.DataFrame(index=STAGES, columns=theory_dimensions.keys()).fillna(0)
    for stage in STAGES:
        df = stage_cooc_dfs.get(stage, pd.DataFrame())
        if not df.empty:
            for dim, kws in theory_dimensions.items():
                result.loc[stage, dim] = df[df['关联预设词'].isin(kws)]['共现频次'].sum()
    return result.astype(int)

def plot_theory_dimension_evolution(strength_df, output_dir, window_size):
    plt.figure(figsize=(12, 7))
    for col in strength_df.columns:
        plt.plot(strength_df.index, strength_df[col], marker='o', linewidth=2, label=col)
    plt.title(f"“{TARGET_KEYWORD}”相关理论维度历时性演变（窗口半径：{window_size}）", fontsize=16)
    plt.xlabel("发展阶段", fontsize=13)
    plt.ylabel("共现频次", fontsize=13)
    plt.xticks(rotation=25, fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(title="理论维度", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, f"theory_dimension_evolution_window{window_size}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"理论维度演变折线图已保存到: {path}")

def main():
    stage_cooc_dfs = {}
    for stage in STAGES:
        df = load_stage_cooc_df(RESULTS_ROOT_DIR, TARGET_KEYWORD, WINDOW_SIZE, stage)
        if not df.empty:
            print(f"{stage} 共现数据加载成功，{len(df)}条。")
        else:
            print(f"{stage} 共现数据未找到或为空。")
        stage_cooc_dfs[stage] = df

    theory_strength_df = calculate_theory_dimension_strengths(stage_cooc_dfs, THEORY_DIMENSIONS)
    print("\n各阶段理论维度共现频次：")
    print(theory_strength_df)

    plot_theory_dimension_evolution(theory_strength_df, OUTPUT_DIR, WINDOW_SIZE)
    theory_strength_df.to_csv(
        os.path.join(OUTPUT_DIR, f"theory_dimension_strengths_window{WINDOW_SIZE}.csv"),
        encoding='utf-8-sig'
    )
    print(f"理论维度共现频次数据已保存到: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()