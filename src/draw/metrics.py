import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

markdown_data = """
| Name                        |   Precision |   Recall |     F1 |
|:----------------------------|------------:|---------:|-------:|
| text-embedding-ada-002      |      38     |   60.083 | 44.648 |
| text-similarity-davinci-001 |      30.333 |   46.583 | 35.148 |
| text-similarity-ada-001     |      34     |   51.917 | 39.41  |
| text-similarity-curie-001   |      37.333 |   58.833 | 43.705 |
| text-similarity-babbage-001 |      36.333 |   56.917 | 42.481 |
| text2vec-base-chinese       |      32     |   50     | 37.557 |
| cross-bert                  |      48.167 |   71.583 | 55.229 |
| siamese-bert                |      53.333 |   78.333 | 61.086 |
"""

# 提取表格数据，以适应Pandas的read_csv函数
table_data = [line.strip().split("|")[1:-1] for line in markdown_data.strip().split("\n")[2:]]

# 创建Pandas DataFrame
columns = ["Name", "Precision", "Recall", "F1"]
df = pd.DataFrame(table_data, columns=columns)
df = df.apply(lambda x: x.str.strip())  # 去除数据中的额外空格

# 将字符串数据转换为数字（除了Name列）
numeric_columns = ["Precision", "Recall", "F1"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 设置Seaborn样式为"deep"
sns.set(style="whitegrid", palette="deep")

# 重新整理数据为长格式以适应Seaborn
df_long = df.melt(id_vars="Name", var_name="Metrics", value_name="Score")

# 绘制柱状图
plt.figure(figsize=(16, 6))
ax = sns.barplot(data=df_long, x="Name", y="Score", hue="Metrics")
ax.set_title("API Retrieve Metrics Comparison")
ax.set_xlabel("Model")
ax.set_ylabel("Score")

# 设置横刻度标签为水平
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('metrics.png', dpi=400, bbox_inches='tight')
plt.show()
