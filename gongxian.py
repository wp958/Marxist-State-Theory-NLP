import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# --- 配置 ---
SEGMENTED_TEXTS_ROOT_DIR = r"D:\zhuo mian\PythonProject2\processed_texts"
TARGET_KEYWORD = "国家"
RESULTS_ROOT_DIR = r"D:\zhuo mian\PythonProject2\cooccurrence_analysis_final"
WINDOW_SIZE = 15  # “国家”前后各取多少个词

# 可选：停用词文件路径，设为None则不使用
STOPWORDS_FILE_PATH = r"D:\zhuo mian\PythonProject2\stopwords\cn_stopwords.txt"
# STOPWORDS_FILE_PATH = None

PRESET_KEYWORDS_BY_STAGE = {
    "早期批判阶段": [ # 年份已去除
        "国家", "阶级", "阶级斗争", "统治阶级", "政治权力", "生产关系", "上层建筑", "革命", "无产阶级", "资产阶级", "暴力",
        "市民社会", "官僚机构/官僚制度", "异化", "政治解放", "共同体", "意识形态", "（统治阶级的）共同利益的虚幻形式"
    ],
    "成熟阶段": [ # 年份已去除
        "国家", "阶级", "阶级斗争", "统治阶级", "政治权力", "生产关系", "上层建筑", "革命", "无产阶级", "资产阶级", "暴力",
        "委员会", "无产阶级专政", "行政权力", "国家机器", "波拿巴主义", "立法", "国家暴力"
    ],
    "实践应用阶段": [ # 年份已去除
        "国家", "阶级", "阶级斗争", "统治阶级", "政治权力", "生产关系", "上层建筑", "革命", "无产阶级", "资产阶级", "暴力",
        "公社", "打碎", "廉价政府", "过渡时期", "国家消亡", "（阶级矛盾）不可调和（的产物）", "公共权力"
    ],
    "继承和发展阶段": [ # 年份已去除
        "国家", "阶级", "阶级斗争", "统治阶级", "政治权力", "生产关系", "上层建筑", "革命", "无产阶级", "资产阶级", "暴力",
        "（国家是阶级统治的）机关/工具", "民主", "镇压", "苏维埃", "“半国家”", "民主集中制"
    ]
}
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 辅助函数 ---
def load_stopwords(filepath):
    stopwords = set()
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f: stopwords.add(line.strip())
            if stopwords: print(f"停用词加载成功，共 {len(stopwords)} 个。")
        except Exception as e:
            print(f"加载停用词文件 '{filepath}' 时出错: {e}。")
    else:
        print("未进行停用词过滤。")
    return stopwords


def filter_words_in_context(words_list, relevant_keywords_set, stopwords_set):
    if not stopwords_set:
        return [word for word in words_list if word in relevant_keywords_set and word.strip()]
    return [word for word in words_list if word in relevant_keywords_set and word not in stopwords_set and word.strip()]


# --- 核心分析函数 ---
def get_target_focused_cooccurrence(stage_name, stage_text_dir, current_stage_preset_kws, target_keyword, stopwords_set,
                                    window_radius):
    target_cooc_edges = Counter()
    if not os.path.isdir(stage_text_dir):
        print(f"警告：目录 '{stage_text_dir}' 不存在，跳过阶段 '{stage_name}'。")
        return [], current_stage_preset_kws

    preset_kws_set = set(current_stage_preset_kws)
    other_preset_kws = preset_kws_set - {target_keyword}
    if not other_preset_kws:
        print(f"警告：阶段 '{stage_name}' 中除了目标词 '{target_keyword}' 外没有其他预设词，无法计算共现。")
        return [], current_stage_preset_kws

    all_words_in_files = []
    for filename in os.listdir(stage_text_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(stage_text_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f: all_words_in_files.extend(line.strip().split())
            except Exception as e:
                print(f"读取文件 {filepath} 时出错: {e}")

    if not all_words_in_files:
        print(f"警告：阶段 '{stage_name}' 的文本文件中未读取到任何词语。")
        return [], current_stage_preset_kws

    filtered_words_stream = filter_words_in_context(all_words_in_files, preset_kws_set, stopwords_set)

    if not filtered_words_stream:
        print(f"警告：阶段 '{stage_name}' 经过滤后无有效预设词。")
        return [], current_stage_preset_kws

    for i, word in enumerate(filtered_words_stream):
        if word == target_keyword:
            start_index = max(0, i - window_radius)
            end_index = min(len(filtered_words_stream), i + window_radius + 1)
            window_other_kws_in_order = []
            for k_idx in range(start_index, i):
                if filtered_words_stream[k_idx] in other_preset_kws:
                    window_other_kws_in_order.append(filtered_words_stream[k_idx])
            for k_idx in range(i + 1, end_index):
                if filtered_words_stream[k_idx] in other_preset_kws:
                    window_other_kws_in_order.append(filtered_words_stream[k_idx])
            unique_window_kws = set(window_other_kws_in_order)
            for pk_word in unique_window_kws:
                target_cooc_edges[(target_keyword, pk_word)] += 1

    if not target_cooc_edges:
        print(f"警告：阶段 '{stage_name}' 未找到 '{target_keyword}' 与其他预设词的共现。")
        return [], current_stage_preset_kws

    edges = [(u, v, weight) for (u, v), weight in target_cooc_edges.items()]
    df_cooc = pd.DataFrame(edges, columns=['目标词', '关联预设词', '共现频次']).sort_values(by='共现频次',
                                                                                            ascending=False).reset_index(
        drop=True)
    print(f"\n阶段: {stage_name} (窗口半径 {window_radius}) - '{target_keyword}' 与其他预设词共现Top10:")
    print(df_cooc.head(10))
    return edges, current_stage_preset_kws


# --- 可视化函数 ---
def plot_stage_network(edges, nodes_list, stage_name, target_keyword, window_radius, output_path):
    if not nodes_list:
        print(f"警告：阶段 '{stage_name}' 无节点可绘制网络图。")
        return

    G = nx.Graph()
    all_graph_nodes = set(nodes_list)
    if edges:
        for u, v, w in edges:
            all_graph_nodes.add(u)
            all_graph_nodes.add(v)

    fixed_pos = {target_keyword: (0, 0)} if target_keyword in all_graph_nodes else {}

    for node in all_graph_nodes:
        G.add_node(node, size=(3000 if node == target_keyword else 1500),
                   color=('orangered' if node == target_keyword else 'skyblue'), label=node)

    if edges:
        G.add_weighted_edges_from(edges)

    pos = None
    if G.number_of_nodes() == 0:
        print(f"警告：阶段 '{stage_name}' 图中无节点，无法绘制。")
        return
    elif G.number_of_nodes() == 1:
        pos = {list(G.nodes())[0]: (0, 0)}
    elif G.number_of_nodes() < 15 and G.number_of_edges() > 0:
        pos = nx.circular_layout(G)
    elif G.number_of_nodes() > 0 :
        try:
            k_val = 1.5 # 你之前确认这个k值效果好
            pos = nx.spring_layout(G, pos=fixed_pos if fixed_pos and target_keyword in G else None,
                                   fixed=list(fixed_pos.keys()) if fixed_pos and target_keyword in G else None,
                                   k=k_val, iterations=100, seed=42, weight='weight', scale=2) # 增加了iterations和scale
        except Exception as e_layout:
            print(f"布局警告 (阶段 {stage_name}): {e_layout}, 使用默认 spring_layout。")
            pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42, weight='weight')
    else:
        pos = nx.random_layout(G, seed=42)

    plt.figure(figsize=(14, 12))
    node_sizes = [d['size'] for n, d in G.nodes(data=True)]
    node_colors = [d['color'] for n, d in G.nodes(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)

    if G.edges():
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        if weights:
            min_w, max_w = min(weights), max(weights)
            edge_widths = [1 + 4 * (w - min_w) / (max_w - min_w) if max_w > min_w else 2 for w in weights]
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='dimgray')
            if len(G.edges()) < 25 and len(G.edges()) > 0:
                edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='darkred')

    labels_to_draw = {n: d['label'] for n, d in G.nodes(data=True) if n in G}
    nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=10, font_weight='normal')
    plt.title(f"阶段: {stage_name} - '{target_keyword}'核心关联网络 (窗口半径: {window_radius})", fontsize=16)
    plt.axis('off'); plt.tight_layout()
    plt.savefig(output_path); plt.close()
    print(f"网络图已保存到: {output_path}")


def plot_overall_heatmap(all_dfs, target_keyword, output_dir, window_radius, top_n_per_stage=5):
    if not all_dfs: print("无数据绘制热力图。"); return
    valid_dfs = {s: df for s, df in all_dfs.items() if isinstance(df, pd.DataFrame) and not df.empty and '关联预设词' in df.columns and '共现频次' in df.columns}
    if not valid_dfs: print("无有效数据绘制热力图。"); return

    top_words_union = set()
    for stage_name, df in valid_dfs.items():
        top_words_union.update(
            df.sort_values(by='共现频次', ascending=False).head(top_n_per_stage)['关联预设词'].tolist())

    heatmap_rows = sorted([w for w in list(top_words_union) if w != target_keyword])
    if not heatmap_rows: print("未能筛选出热力图行（排除目标词后为空）。"); return

    stages_ordered = list(PRESET_KEYWORDS_BY_STAGE.keys())
    heatmap_df = pd.DataFrame(0, index=heatmap_rows, columns=stages_ordered)

    for stage_col_name in stages_ordered:
        if stage_col_name in valid_dfs:
            df_current_stage = valid_dfs[stage_col_name]
            for word_row_name in heatmap_rows:
                matching_rows = df_current_stage[df_current_stage['关联预设词'] == word_row_name]
                if not matching_rows.empty:
                    heatmap_df.loc[word_row_name, stage_col_name] = matching_rows['共现频次'].sum()

    if heatmap_df.empty: print("生成的热力图数据为空。"); return
    plt.figure(figsize=(12, max(6, len(heatmap_rows) * 0.5)))
    sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5, cbar=True)
    title = f"'{target_keyword}' 与阶段性核心词共现演变 (各阶段Top{top_n_per_stage}词并集)\n(窗口半径: {window_radius})"
    plt.title(title, fontsize=14, y=1.03)
    plt.xlabel("发展阶段", fontsize=12); plt.ylabel("核心关联词", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=10); plt.yticks(fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(output_dir, f"overall_heatmap_stage_top{top_n_per_stage}_window{window_radius}.png")
    plt.savefig(path); plt.close()
    print(f"总体演变热力图已保存到: {path}")


def plot_overall_lines(all_dfs, target_keyword, output_dir, window_radius, num_lines=7):
    if not all_dfs: print("无数据绘制折线图。"); return
    valid_dfs = {s: df for s, df in all_dfs.items() if isinstance(df, pd.DataFrame) and not df.empty and '关联预设词' in df.columns and '共现频次' in df.columns}
    if not valid_dfs: print("无有效数据绘制折线图。"); return

    overall_counts = Counter()
    for df in valid_dfs.values():
        for _, r in df.iterrows(): overall_counts[r['关联预设词']] += r['共现频次']

    words_to_plot = [w for w, c in overall_counts.most_common(num_lines + 5) if w != target_keyword and c > 0][:num_lines]
    if not words_to_plot: print("未能确定折线图追踪词（排除目标词或无共现）。"); return

    stages_ordered = list(PRESET_KEYWORDS_BY_STAGE.keys())
    lines_df = pd.DataFrame(0, index=stages_ordered, columns=words_to_plot)

    for stage_col_name in stages_ordered:
        if stage_col_name in valid_dfs:
            df_current_stage = valid_dfs[stage_col_name]
            for word_line_name in words_to_plot:
                matching_rows = df_current_stage[df_current_stage['关联预设词'] == word_line_name]
                if not matching_rows.empty:
                    lines_df.loc[stage_col_name, word_line_name] = matching_rows['共现频次'].sum()

    if lines_df.empty: print("生成的折线图数据为空。"); return
    plt.figure(figsize=(13, 8))
    for word in words_to_plot:
        plt.plot(lines_df.index, lines_df[word], marker='o', linestyle='-', linewidth=2, markersize=6, label=word)
    plt.title(f"'{target_keyword}' 与Top{len(words_to_plot)}核心词共现频次演变 (窗口半径: {window_radius})", fontsize=16)
    plt.xlabel("发展阶段", fontsize=12); plt.ylabel(f"与'{target_keyword}'共现频次", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=10); plt.yticks(fontsize=10)
    plt.legend(title="核心关联词", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, title_fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(rect=[0, 0, 0.88, 1])
    path = os.path.join(output_dir, f"overall_lines_top{len(words_to_plot)}_window{window_radius}.png")
    plt.savefig(path); plt.close()
    print(f"总体演变折线图已保存到: {path}")


# 修改 plot_target_degree_evolution 函数，增加词汇标注
def plot_target_degree_evolution_enhanced(degrees_data, stage_words_data, target_keyword, output_dir,
                                          window_radius_info=""):
    """
    绘制增强版的连接度演变图，包含具体的连接词信息

    Args:
        degrees_data: 各阶段的连接度数据
        stage_words_data: 各阶段的具体连接词及其频次
        target_keyword: 目标关键词
        output_dir: 输出目录
        window_radius_info: 窗口半径信息
    """
    if not degrees_data:
        print("无连接度数据可绘制。")
        return

    stages = list(PRESET_KEYWORDS_BY_STAGE.keys())
    degrees = [degrees_data.get(s, 0) for s in stages]

    # 创建图形，增大尺寸以容纳文字
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 2])

    # 上部分：柱状图
    bars = ax1.bar(stages, degrees, color='cornflowerblue', width=0.55)
    ax1.set_ylabel(f"'{target_keyword}' 的连接度 (连接的预设词数量)", fontsize=12)
    title = f"'{target_keyword}' 节点连接度及核心词演变"
    if window_radius_info:
        title += f" (窗口半径: {window_radius_info})"
    ax1.set_title(title, fontsize=16, pad=20)
    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stages, rotation=25, ha="right", fontsize=11)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 在柱子上方标注数值
    max_degree_val = max(degrees) if degrees else 1
    for i, (bar, stage) in enumerate(zip(bars, stages)):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02 * max_degree_val,
                 int(yval), ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 在柱子内部或旁边显示主要连接词（前3个）
        if stage in stage_words_data and stage_words_data[stage]:
            top_words = stage_words_data[stage][:3]  # 取前3个高频词
            words_text = '\n'.join([f"{w[0]}" for w in top_words])
            # 根据柱子高度决定文字位置
            if yval > max_degree_val * 0.5:
                ax1.text(bar.get_x() + bar.get_width() / 2.0, yval * 0.5,
                         words_text, ha='center', va='center', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # 下部分：详细的词汇演变表格
    ax2.axis('tight')
    ax2.axis('off')

    # 准备表格数据
    table_data = []
    for stage in stages:
        if stage in stage_words_data and stage_words_data[stage]:
            # 获取前5个高频词及其频次
            top_words_with_freq = stage_words_data[stage][:5]
            words_str = '、'.join([f"{w[0]}({w[1]})" for w in top_words_with_freq])
            table_data.append([stage, words_str])
        else:
            table_data.append([stage, "无数据"])

    # 创建表格
    table = ax2.table(cellText=table_data,
                      colLabels=['发展阶段', f'与"{target_keyword}"共现的主要词汇（频次）'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.2, 0.8])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表格样式
    for i in range(len(table_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white')
            else:
                if j == 0:  # 阶段列
                    cell.set_facecolor('#E7E6E6')
                cell.set_text_props(wrap=True)

    plt.tight_layout()
    filename = f"{target_keyword}_degree_evolution_enhanced_window{window_radius_info}.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"增强版连接度演变图已保存到: {path}")


# 新增：创建词汇流动桑基图
def plot_keywords_flow_sankey(all_stages_cooc_dfs, target_keyword, output_dir, window_radius, top_n=5):
    """
    创建桑基图展示核心词在各阶段的流动和演变
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("需要安装plotly库来生成桑基图：pip install plotly")
        return

    if not all_stages_cooc_dfs:
        print("无数据生成桑基图。")
        return

    stages = list(PRESET_KEYWORDS_BY_STAGE.keys())

    # 收集所有阶段的top词汇
    all_keywords = set()
    stage_top_words = {}

    for stage in stages:
        if stage in all_stages_cooc_dfs and not all_stages_cooc_dfs[stage].empty:
            df = all_stages_cooc_dfs[stage]
            top_words = df.nlargest(top_n, '共现频次')[['关联预设词', '共现频次']].values.tolist()
            stage_top_words[stage] = top_words
            all_keywords.update([w[0] for w in top_words])

    # 创建节点列表
    nodes = []
    node_dict = {}
    node_idx = 0

    # 添加阶段节点
    for i, stage in enumerate(stages):
        nodes.append(stage)
        node_dict[f"{stage}_stage"] = node_idx
        node_idx += 1

    # 添加关键词节点（为每个阶段创建独立的关键词节点）
    for stage in stages:
        if stage in stage_top_words:
            for word, _ in stage_top_words[stage]:
                node_name = f"{word}_{stage}"
                nodes.append(word)
                node_dict[node_name] = node_idx
                node_idx += 1

    # 创建连接
    sources = []
    targets = []
    values = []
    colors = []

    # 阶段到关键词的连接
    color_map = {
        stages[0]: 'rgba(255, 127, 14, 0.4)',
        stages[1]: 'rgba(44, 160, 44, 0.4)',
        stages[2]: 'rgba(214, 39, 40, 0.4)',
        stages[3]: 'rgba(148, 103, 189, 0.4)'
    }

    for stage in stages:
        if stage in stage_top_words:
            stage_idx = node_dict[f"{stage}_stage"]
            for word, freq in stage_top_words[stage]:
                word_idx = node_dict[f"{word}_{stage}"]
                sources.append(stage_idx)
                targets.append(word_idx)
                values.append(freq)
                colors.append(color_map.get(stage, 'rgba(128, 128, 128, 0.4)'))

    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="blue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    )])

    fig.update_layout(
        title_text=f"'{target_keyword}'关联词汇在各发展阶段的分布（窗口半径: {window_radius}）",
        font_size=12,
        width=1200,
        height=800
    )

    path = os.path.join(output_dir, f"keywords_flow_sankey_window{window_radius}.html")
    fig.write_html(path)
    print(f"词汇流动桑基图已保存到: {path}")


# 修改主程序 main() 函数的相关部分
def main():
    print(f"开始分析 '{TARGET_KEYWORD}' (窗口半径: {WINDOW_SIZE})...")
    stopwords = load_stopwords(STOPWORDS_FILE_PATH)

    if not os.path.isdir(SEGMENTED_TEXTS_ROOT_DIR):
        print(f"错误：文本目录 '{SEGMENTED_TEXTS_ROOT_DIR}' 不存在。脚本将终止。")
        return

    current_run_output_dir = os.path.join(RESULTS_ROOT_DIR, f"target_{TARGET_KEYWORD}_window_{WINDOW_SIZE}")
    os.makedirs(current_run_output_dir, exist_ok=True)
    print(f"结果将保存到: {current_run_output_dir}")

    all_stages_cooc_dfs = {}
    all_stages_target_degrees = {}
    all_stages_top_words = {}  # 新增：存储每个阶段的主要连接词

    stages_in_order = list(PRESET_KEYWORDS_BY_STAGE.keys())

    for stage_name in stages_in_order:
        preset_kws = PRESET_KEYWORDS_BY_STAGE[stage_name]
        print(f"\n--- 处理阶段: {stage_name} ---")
        clean_stage_name_for_path = stage_name.replace("（", "_").replace("）", "").replace("/", "_").replace(" ", "_")
        stage_phase_output_dir_for_run = os.path.join(current_run_output_dir, clean_stage_name_for_path)
        os.makedirs(stage_phase_output_dir_for_run, exist_ok=True)

        stage_text_path = os.path.join(SEGMENTED_TEXTS_ROOT_DIR, stage_name)
        current_kws_for_stage = list(set(preset_kws))
        if TARGET_KEYWORD not in current_kws_for_stage:
            current_kws_for_stage.append(TARGET_KEYWORD)

        edges, nodes_for_plotting = get_target_focused_cooccurrence(
            stage_name, stage_text_path, current_kws_for_stage, TARGET_KEYWORD, stopwords, WINDOW_SIZE
        )

        # 计算连接度和收集主要连接词
        target_degree = 0
        stage_connected_words = []

        if edges:
            # 统计每个关联词的频次
            word_freq = {}
            for _, word, freq in edges:
                if word != TARGET_KEYWORD:
                    word_freq[word] = word_freq.get(word, 0) + freq

            # 按频次排序
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            stage_connected_words = sorted_words
            target_degree = len(word_freq)

        all_stages_target_degrees[stage_name] = target_degree
        all_stages_top_words[stage_name] = stage_connected_words
        print(f"阶段 '{stage_name}': '{TARGET_KEYWORD}' 的连接度为: {target_degree}")
        if stage_connected_words:
            print(f"主要连接词: {', '.join([f'{w[0]}({w[1]})' for w in stage_connected_words[:5]])}")

        df_cooc_current_stage = pd.DataFrame()
        if edges:
            df_cooc_current_stage = pd.DataFrame(edges, columns=['目标词', '关联预设词', '共现频次']).sort_values(
                by='共现频次', ascending=False).reset_index(drop=True)
            csv_path = os.path.join(stage_phase_output_dir_for_run,
                                    f"{clean_stage_name_for_path}_cooc_win{WINDOW_SIZE}.csv")
            df_cooc_current_stage.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"阶段性共现数据已保存到: {csv_path}")
            network_plot_path = os.path.join(stage_phase_output_dir_for_run,
                                             f"{clean_stage_name_for_path}_network_win{WINDOW_SIZE}.png")
            plot_stage_network(edges, nodes_for_plotting, stage_name, TARGET_KEYWORD, WINDOW_SIZE, network_plot_path)
        else:
            print(f"阶段 '{stage_name}' 无共现边，不生成CSV和网络图。")
        all_stages_cooc_dfs[stage_name] = df_cooc_current_stage

    if all_stages_cooc_dfs or all_stages_target_degrees:
        print("\n--- 生成总体演变图 ---")
        if all_stages_cooc_dfs:
            plot_overall_heatmap(all_stages_cooc_dfs, TARGET_KEYWORD, current_run_output_dir, WINDOW_SIZE,
                                 top_n_per_stage=7)
            plot_overall_lines(all_stages_cooc_dfs, TARGET_KEYWORD, current_run_output_dir, WINDOW_SIZE, num_lines=8)
            # 新增：生成桑基图
            plot_keywords_flow_sankey(all_stages_cooc_dfs, TARGET_KEYWORD, current_run_output_dir, WINDOW_SIZE)

        if all_stages_target_degrees:
            # 使用增强版的连接度演变图
            plot_target_degree_evolution_enhanced(
                all_stages_target_degrees,
                all_stages_top_words,
                TARGET_KEYWORD,
                current_run_output_dir,
                window_radius_info=str(WINDOW_SIZE)
            )
    else:
        print("所有阶段均未产生有效的共现或连接度数据，无法生成总体演变图。")

    print(f"\n--- 分析完成 ---")
    print(f"所有结果已保存在目录: {current_run_output_dir}")


if __name__ == '__main__':
    main()