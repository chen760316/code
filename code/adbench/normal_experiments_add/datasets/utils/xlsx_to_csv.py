import pandas as pd

# 读取Excel文件
file_path = "../multi_class/drybean.xlsx"
df = pd.read_excel(file_path)

# 将数据保存为CSV文件
csv_file_path = file_path.replace(".xlsx", ".csv")
df.to_csv(csv_file_path, index=False)

print(f"文件已保存为 CSV: {csv_file_path}")