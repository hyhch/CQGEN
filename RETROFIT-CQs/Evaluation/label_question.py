import pandas as pd
import openai

openai.api_key = "YOUR_API_KEY_HERE"
openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 定义分类函数
def classify_question(question):
    """调用大语言模型对问题分类为简单问题或复杂问题。"""
    if(question == 'Sentence1'):
        return "Unknown"
    try:
        # 调用大语言模型
        completion = openai.ChatCompletion.create(
            model="qwen-max",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': f'''Simple CQs (Simple CQs)
                features:
                Single hop query: Only one ontology class or attribute needs to be accessed, without the need for cross relationship inference.
                Direct retrieval: The answer can be obtained by directly matching attributes or instances in the ontology.

                No computation or aggregation: does not involve operations such as statistics, sorting, conditional filtering, etc.

                Example:
                What is the username of the player?
                (Directly retrieve the player's username attribute)

                What is the genre of the game?
                (Directly retrieve the type attributes of the game)

                Which devices measure temperature?
                (Directly match sensor classes and their measurement attributes)

                What is a building?
                (Directly retrieve the class definition of the ontology)

                Complex CQs (Complex Problems)
                features:
                Multi hop query: requires crossing multiple ontology classes, attributes, or relationships (topological distance ≥ 2).
                Inference or computation: involving conditional filtering, statistics, aggregation (such as "maximum", "average", "probability"), path analysis, etc.

                Dynamic or cross domain: may rely on time series, spatial relationships, or cross ontology associations.

                Example:
                Who are the friends that play other games with this player?
                (Need to associate the "player → friend → game" multi hop relationship)

                What is the likelihood that a player who purchased in-app items in one game will do so in another?
                (Probability Reasoning and Cross Game Behavior Analysis)

                Which roads connect two towns via the optimum path?
                (Spatial Path Planning and Multi condition Filtering)

                How many players clicked an in-game advertisement and then started another game?
                (Behavior Sequence Analysis and Statistics)

                What are the most traded items in the game’s marketplace?
                (Aggregation statistics and sorting)
                 Classify the following competency question as "Simple" or "Complex": {question}
                Please do not reply any other information.
            '''},
            ],
        )
        # 从返回内容中提取模型的回答
        response = completion.choices[0].message['content'].strip().lower()
        print(question)
        print(response)
        # 简化为 "Simple" 或 "Complex"
        if "simple" in response:
            return "Simple"
        elif "complex" in response:
            return "Complex"
        else:
            return "Unknown"  # 如果无法分类
    except Exception as e:
        print(f"Error classifying question: {question}. Error: {e}")
        return "Error"

# 读取Excel文件
file_path = "../Implementation/MyResults/SAREF_qwen_out.xlsx"
df = pd.read_excel(file_path)

# 检查是否存在 "Competency Questions" 列
if "Sentence1" not in df.columns:
    raise ValueError("The column 'Competency Questions' does not exist in the Excel file.")

# 对每一行的 Competency Questions 进行分类并添加到新列
df['label'] = df['Sentence1'].apply(classify_question)

# 保存结果到同一个文件或另存为新文件
output_file_path = "SAREF_qwen_out_labeled.xlsx"
df.to_excel(output_file_path, index=False)

print(f"Classification completed. Labeled file saved to: {output_file_path}")