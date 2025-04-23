import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

# 读取原始数据
df_inpatient = pd.read_csv("train_inpatient.csv", low_memory=False)
df_outpatient = pd.read_csv("train_outpatient.csv", low_memory=False)
df_beneficiary = pd.read_csv("train_beneficiary.csv", low_memory=False)

# 定义字段
physician_columns = ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']
diagnosis_code_columns = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
procedure_code_columns = [f'ClmProcedureCode_{i}' for i in range(1, 7)]

# 创建组合字段
for df in [df_inpatient, df_outpatient]:
    for col in physician_columns + diagnosis_code_columns + procedure_code_columns:
        df[col] = df[col].fillna("Unknown").astype(str)
    df['Physician_group_String'] = df[physician_columns].apply(lambda row: '-'.join(row), axis=1)
    df['DiagnosisCode_group_String'] = df[diagnosis_code_columns].apply(lambda row: '-'.join(row), axis=1)
    df['procedureCode_group_String'] = df[procedure_code_columns].apply(lambda row: '-'.join(row), axis=1)

# 合并 beneficiary
df_beneficiary['DOB'] = pd.to_datetime(df_beneficiary['DOB'], errors='coerce')
reference_date = pd.to_datetime('2019-01-01')
df_beneficiary['Age'] = (reference_date - df_beneficiary['DOB']).dt.days // 365
df_inpatient = df_inpatient.merge(df_beneficiary, on='BeneID', how='left')
df_outpatient = df_outpatient.merge(df_beneficiary, on='BeneID', how='left')

# 合并 inpatient 和 outpatient
combined_df = pd.concat([df_inpatient, df_outpatient], ignore_index=True)

# 要编码的列
label_cols = ['Gender', 'Race', 'RenalDiseaseIndicator', 'State', 'County']
category_columns = physician_columns + diagnosis_code_columns + procedure_code_columns + [
    'Physician_group_String', 'DiagnosisCode_group_String', 'procedureCode_group_String', 'BeneID'
]
all_cat_cols = list(set(label_cols + category_columns))

# 创建映射字典
label_mapping_dict = {}

for col in all_cat_cols:
    combined_df[col] = combined_df[col].fillna("Unknown").astype(str)
    le = LabelEncoder()
    le.fit(combined_df[col])
    # 使用 list(le.classes_) 将 numpy 类型转换为 JSON 可序列化类型
    label_mapping_dict[col] = list(le.classes_)

# 保存为 JSON
with open("final_integrated_model_label_encoders.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping_dict, f, ensure_ascii=False, indent=2)

print("✅ LabelEncoder 映射已保存为 label_encoders.json")
