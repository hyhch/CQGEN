import pandas as pd
import re
def calculate_different_questions(file_a):
    # 加载文件 A
    df_a = pd.read_excel(file_a)
    
    # 精确率的分子：Decision 列中不是 "No match" 且不是空的单元格中的句子总数
    simple_numerator = 0
    simple_total = 0
    complex_numerator = 0
    complex_total = 0
    for idx, row in df_a.iloc[0:].iterrows():  # 从第2行开始
        decision = row['Decision']
        label = row['label']
        print("decision",decision)
        print("label",label)
        if pd.isna(label):
            continue
        if label.lower() == 'simple':
            simple_total = simple_total + 1 
            if pd.notna(decision) and decision != "No match":
                    simple_numerator = simple_numerator + 1
        if label.lower() == "complex":
            complex_total = complex_total + 1 
            if pd.notna(decision) and decision != "No match":
                    complex_numerator = complex_numerator + 1
    
    
    # 打印结果
    print(f"simple Numerator: {simple_numerator}")
    print(f"simple_total: {simple_total}")
    print(f"complex Numerator: {complex_numerator}")
    print(f"complex_total: {complex_total}")
    
    # 返回结果
    return {
        "simple_numerator": simple_numerator,
        "simple_total": simple_total,
        "complex_numerator": complex_numerator,
        "complex_total": complex_total
    }


def calculate_metrics(file_a, file_b):
    # 加载文件 A
    df_a = pd.read_excel(file_a)
    
    # 精确率的分子：Decision 列中不是 "No match" 且不是空的单元格中的句子总数
    precision_numerator = 0
    for decision in df_a['Decision'].dropna():  # 忽略空值
        if decision != "No match":
            sentences = re.findall(r',', decision)
            precision_numerator += len(sentences)+1
    
    # 召回率的分子：Decision 列中不是 "No match" 且不是空的单元格数量
    recall_numerator = df_a['Decision'].iloc[1:].dropna().apply(lambda x: x != "No match").sum()
    
    # 召回率的分母：Sentence1 列的非空单元格数量
    recall_denominator = df_a['Sentence1'].iloc[1:].dropna().shape[0]
    
    # 加载文件 B
    df_b = pd.read_excel(file_b, usecols=['Sentence2'])  # 只加载 E 列
    
    # 精确率的分母：E 列从第 2 行开始的所有句子总数
    precision_denominator = 0
    for cell in df_b['Sentence2'].iloc[1:].dropna():  # 忽略空值，从第 2 行开始
        sentences = cell.splitlines()  # 按行分割句子
        precision_denominator += len(sentences)
    
    # 返回精确率、召回率和F1分数
    precision = precision_numerator / precision_denominator if precision_denominator > 0 else 0
    recall = recall_numerator / recall_denominator if recall_denominator > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

# 调用函数并传入文件路径
file_a = '../Implementation/MyResults/base_ontology_llm_out.xlsx'  # 替换为文件 A 的路径
file_b = '../Implementation/MyResults/base_ontology_llm.xlsx'  # 替换为文件 B 的路径



# metrics = calculate_metrics(file_a, file_b)



# 如果要算简单/复杂问题把以下的注释取消
metrics = calculate_different_questions('../Implementation/MyResults/base_ontology_qwen_out.xlsx')

# 计算精确率和召回率
# precision = metrics["precision_numerator"] / metrics["precision_denominator"] if metrics["precision_denominator"] > 0 else 0
# recall = metrics["recall_numerator"] / metrics["recall_denominator"] if metrics["recall_denominator"] > 0 else 0

# 计算 F1 分数
# f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

simple = metrics["simple_numerator"] / metrics["simple_total"] if metrics["simple_total"] > 0 else 0
complex = metrics["complex_numerator"] / metrics["complex_total"] if metrics["complex_total"] > 0 else 0

print(f"simple: {simple:.4f}")
print(f"complex: {complex:.4f}")