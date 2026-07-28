import pandas as pd
from sentence_transformers import SentenceTransformer, util
from IPython.display import Audio
from numpy import sin, pi, arange
from tqdm import tqdm, trange
from openpyxl import load_workbook, Workbook
import warnings

videoGameLength = 66
vicinitycoreLength = 58
demCareLength = 107
saref4envLength = 58
SWOLength = 89
SAREFLength = 72
OneM2MLength = 59

# Suppress the UserWarning related to tensorflow and torch-xla
warnings.filterwarnings("ignore", category=UserWarning, message="tensorflow")


def beep(frequency=440, duration=1, sampling_rate=44100):
    t = arange(sampling_rate * duration)
    waveform = sin(2 * pi * frequency * t / sampling_rate)
    return Audio(waveform, rate=sampling_rate, autoplay=True)

# def main(input_file, output_file):
#     df = pd.read_excel(input_file, header=None, names=['Sentence1', 'Sentence2'])  # The column 'Sentence1' contains the benchmark CQs, and the column 'Sentence2' contains the generated CQs
#     print("Data read from Excel:", df.head())  # Debugging line

#     num_rows_first_column = df['Sentence1'].count()
#     print("Number of rows in the first column:", num_rows_first_column)  # Debugging line

#     sentence1_list = df['Sentence1'].astype(str).tolist()
#     sentence2_list = df['Sentence2'].astype(str).tolist()
#     model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
#     print("Model loaded:", model)  # Debugging line

#     output_rows = []

#     for i, sentence1 in enumerate(sentence1_list):
#         if i >= num_rows_first_column:
#             break

#         embedding1 = model.encode(sentence1, convert_to_tensor=True)
#         matched_sentences = []

#         for sentence2 in sentence2_list:
#             embedding2 = model.encode(sentence2, convert_to_tensor=True)
#             similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
#             print(f"Similarity between '{sentence1}' and '{sentence2}': {similarity_score}")  # Debugging line

#             if similarity_score >= 0.7:# set your similarity threshold
#                 matched_sentences.append(sentence2)

#         decision = ', '.join(matched_sentences) if matched_sentences else "No match"
#         output_rows.append([sentence1, decision])

#     output_df = pd.DataFrame(output_rows, columns=['Sentence1', 'Decision'])
#     print("Output DataFrame:", output_df.head())  # Debugging line

#     output_df.to_excel(output_file, index=False)
#     print("Output file generated successfully!")

def main(input_file, output_file):
    # 读取输入文件
    df = pd.read_excel(input_file, header=None, names=['Sentence1', 'Sentence2'])  # The column 'Sentence1' contains the benchmark CQs, and the column 'Sentence2' contains the generated CQs
    print("Data read from Excel:", df.head())  # Debugging line

    # 获取基准句子列表
    sentence1_list = df['Sentence1'].astype(str).tolist()

    # 将 Sentence2 列中每个单元格的多行问题拆分为单独的句子
    sentence2_list = []
    for cell in df['Sentence2'].astype(str):
        sentence2_list.extend(cell.splitlines())  # 按行拆分问题

    # 加载 SentenceTransformer 模型
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    print("Model loaded:", model)  # Debugging line

    # 存储输出结果
    output_rows = []
    i = 0
    # 遍历基准句子
    for sentence1 in sentence1_list:
        i = i+1
        if i == OneM2MLength+1:  # 检查 sentence1 是否为 NaN 或空字符串
            print("Encountered empty or NaN sentence1. Exiting loop.")
            break

        embedding1 = model.encode(sentence1, convert_to_tensor=True)
        matched_sentences = []

        # 遍历拆分后的生成句子
        for sentence2 in sentence2_list:
            embedding2 = model.encode(sentence2, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
            print(f"Similarity between '{sentence1}' and '{sentence2}': {similarity_score}")  # Debugging line

            # 如果相似度超过阈值，记录匹配的句子
            if similarity_score >= 0.6:  # set your similarity threshold
                matched_sentences.append(sentence2)
                sentence2_list.remove(sentence2)

        # 记录匹配结果
        decision = ', '.join(matched_sentences) if matched_sentences else "No match"
        output_rows.append([sentence1, decision])

    # 将结果保存到 DataFrame
    output_df = pd.DataFrame(output_rows, columns=['Sentence1', 'Decision'])
    print("Output DataFrame:", output_df.head())  # Debugging line

    # 保存结果到 Excel 文件
    output_df.to_excel(output_file, index=False)
    print("Output file generated successfully!")

# Call the main function
main('../Implementation/MyResults/base_ontology_llm.xlsx',
     '../Implementation/MyResults/base_ontology_llm_out.xlsx')
beep()
