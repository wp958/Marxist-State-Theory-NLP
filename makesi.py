import os
import re
import jieba

# --- 配置 ---
# 保持不变，但 ORIGINAL_TEXTS_DIR 现在是包含阶段子文件夹的根目录
ORIGINAL_TEXTS_DIR = r"D:\zhuo mian\PythonProject2\文本\马克思"
PROCESSED_SENTENCES_DIR = r"D:\zhuo mian\PythonProject2\processed_texts" # 处理后按句子输出的根目录
CUSTOM_DICT_PATH = r"D:\zhuo mian\PythonProject2\zidingyicidian.txt"
STOPWORDS_PATH = r"D:\zhuo mian\PythonProject2\tingyongci.txt"

# --- 加载Jieba和停用词 (您的代码，保持不变) ---
try:
    jieba.load_userdict(CUSTOM_DICT_PATH)
    print(f"自定义词典 '{CUSTOM_DICT_PATH}' 加载成功。")
except Exception as e:
    print(f"加载自定义词典失败: {e}")

def load_stopwords(filepath):
    stopwords = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
        print(f"停用词词典 '{filepath}' 加载成功，共 {len(stopwords)} 个停用词。")
    except Exception as e:
        print(f"加载停用词失败: {e}")
    return stopwords

stopwords_set = load_stopwords(STOPWORDS_PATH)

# --- 创建根输出目录 (如果不存在) ---
if not os.path.exists(PROCESSED_SENTENCES_DIR):
    os.makedirs(PROCESSED_SENTENCES_DIR)
    print(f"创建根输出目录: {PROCESSED_SENTENCES_DIR}")

# --- 句子分割和处理函数 (您的代码，保持不变) ---
def split_text_into_sentences(text_content):
    sentences = re.split(r'([。？！\n\r]+)', text_content)
    result_sentences = []
    current_sentence_parts = []
    for part in sentences:
        if not part.strip():
            continue
        current_sentence_parts.append(part)
        if any(term in part for term in ['。', '？', '！', '\n', '\r']):
            sentence = "".join(current_sentence_parts).strip()
            if sentence:
                result_sentences.append(sentence)
            current_sentence_parts = []
    if current_sentence_parts:
        sentence = "".join(current_sentence_parts).strip()
        if sentence:
            result_sentences.append(sentence)
    final_sentences = []
    for s in result_sentences:
        s_cleaned = re.sub(r'[\n\r]+', ' ', s).strip()
        if s_cleaned:
            final_sentences.append(s_cleaned)
    return final_sentences

def process_sentence(sentence_text, stopwords):
    cleaned_sentence = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', sentence_text)
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
    if not cleaned_sentence:
        return []
    words = jieba.cut(cleaned_sentence, cut_all=False, HMM=True)
    filtered_words = [
        word for word in words
        if word not in stopwords and len(word.strip()) > 1 # 保留了去除单字词的逻辑
    ]
    return filtered_words

# --- 主处理逻辑 (修改后) ---
# 遍历 ORIGINAL_TEXTS_DIR 下的每个阶段子文件夹
for stage_folder_name in sorted(os.listdir(ORIGINAL_TEXTS_DIR)): # sorted() 保证阶段顺序
    stage_folder_path = os.path.join(ORIGINAL_TEXTS_DIR, stage_folder_name)

    if os.path.isdir(stage_folder_path): # 确保是文件夹
        print(f"\n--- 开始处理阶段: {stage_folder_name} ---")

        # 为每个阶段在 PROCESSED_SENTENCES_DIR 下创建对应的子文件夹
        processed_stage_dir = os.path.join(PROCESSED_SENTENCES_DIR, stage_folder_name)
        if not os.path.exists(processed_stage_dir):
            os.makedirs(processed_stage_dir)
            print(f"  创建阶段输出目录: {processed_stage_dir}")

        # 遍历该阶段文件夹下的所有文本文件
        for filename in sorted(os.listdir(stage_folder_path)): # sorted() 保证文件顺序
            if filename.endswith(".txt"):  # 或其他您的原始文件格式
                original_filepath = os.path.join(stage_folder_path, filename)
                # 输出文件名保持与原文件名对应，但放入新的阶段子目录
                processed_filepath = os.path.join(processed_stage_dir, f"sentences_{filename}")

                print(f"  正在处理原始文件: {original_filepath}...")
                try:
                    with open(original_filepath, 'r', encoding='utf-8') as f_orig:
                        original_content = f_orig.read()

                    sentences = split_text_into_sentences(original_content)
                    # print(f"    文件被分割成 {len(sentences)} 个句子。") # 可以取消注释用于调试

                    with open(processed_filepath, 'w', encoding='utf-8') as f_proc:
                        processed_sentence_count = 0
                        for i, sentence_str in enumerate(sentences):
                            processed_tokens = process_sentence(sentence_str, stopwords_set)
                            if processed_tokens:  # 只写入包含有效词语的句子
                                f_proc.write(" ".join(processed_tokens) + "\n") # 每行一个处理后的句子
                                processed_sentence_count += 1
                    print(f"    处理完成，{processed_sentence_count} 个有效句子已保存到: {processed_filepath}")

                except Exception as e:
                    print(f"    处理文件 {original_filepath} 失败: {e}")
        print(f"--- 阶段 {stage_folder_name} 处理完毕 ---")

print("\n--- 所有原始文本预处理完成 (按阶段和句子输出) ---")