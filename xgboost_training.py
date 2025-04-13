import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier


df_inpatient = pd.read_csv("train_inpatient.csv", low_memory=False)
df_outpatient = pd.read_csv("train_outpatient.csv", low_memory=False)
df_beneficiary = pd.read_csv("train_beneficiary.csv", low_memory=False)
df_labels = pd.read_csv("train_output.csv", low_memory=False)


# 将 DOB 转换为 datetime 类型
df_beneficiary['DOB'] = pd.to_datetime(df_beneficiary['DOB'], errors='coerce')

# 以 2019 年为基准日期计算年龄
reference_date = pd.to_datetime('2019-01-01')
df_beneficiary['Age'] = (reference_date - df_beneficiary['DOB']).dt.days // 365

# 定义年龄分组
bins = [0, 40, 60, 80, 120]
labels = ['0-40', '41-60', '61-80','81+']
df_beneficiary['AgeGroup'] = pd.cut(df_beneficiary['Age'], bins=bins, labels=labels, right=True)

df_beneficiary['age_group_string'] = df_beneficiary['AgeGroup'].astype(str)


# 定义诊断码和手术码的列
physician_columns = ['AttendingPhysician', 'OperatingPhysician','OtherPhysician', ]
diagnosis_code_columns = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 
                  'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 
                  'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 
                  'ClmDiagnosisCode_10',]

procedure_code_columns = ['ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3', 
                  'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6']

chronic_cond_columns =['ChronicCond_Heartfailure','ChronicCond_Diabetes','ChronicCond_IschemicHeart',"ChronicCond_Alzheimer","ChronicCond_KidneyDisease", "ChronicCond_Cancer","ChronicCond_ObstrPulmonary",
        "ChronicCond_Depression","ChronicCond_Osteoporasis","ChronicCond_rheumatoidarthritis","ChronicCond_stroke"]        

df_inpatient['Physician_group_String'] = df_inpatient[physician_columns].astype(str).apply(lambda row: '-'.join(row), axis=1)
df_outpatient['Physician_group_String'] = df_outpatient[physician_columns].astype(str).apply(lambda row: '-'.join(row), axis=1)
df_inpatient['DiagnosisCode_group_String'] = df_inpatient[diagnosis_code_columns].astype(str).apply(lambda row: '-'.join(row), axis=1)
df_outpatient['DiagnosisCode_group_String'] = df_outpatient[diagnosis_code_columns].astype(str).apply(lambda row: '-'.join(row), axis=1)
df_inpatient['procedureCode_group_String'] = df_inpatient[procedure_code_columns].astype(str).apply(lambda row: '-'.join(row), axis=1)
df_outpatient['procedureCode_group_String'] = df_outpatient[procedure_code_columns].astype(str).apply(lambda row: '-'.join(row), axis=1)
df_beneficiary['condition_group_String'] = df_beneficiary[chronic_cond_columns].astype(str).apply(lambda row: '-'.join(row), axis=1)

category_columns = ['AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician','ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
       'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
       'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
       'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10','ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6','Physician_group_String','DiagnosisCode_group_String', 'procedureCode_group_String', 'BeneID']
df_inpatient[category_columns] = df_inpatient[category_columns].fillna("Unknown")
df_outpatient[category_columns] = df_outpatient[category_columns].fillna("Unknown")


# 合并 beneficiary 信息
df_inpatient = df_inpatient.merge(df_beneficiary, on='BeneID', how='left')
df_outpatient = df_outpatient.merge(df_beneficiary, on='BeneID', how='left')

# 日期处理
date_cols_inpatient = ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt', 'DOB', 'DOD']
date_cols_outpatient = ['ClaimStartDt', 'ClaimEndDt', 'DOB', 'DOD']

for col in date_cols_inpatient:
    if col in df_inpatient.columns:
        df_inpatient[col] = pd.to_datetime(df_inpatient[col], errors='coerce')

for col in date_cols_outpatient:
    if col in df_outpatient.columns:
        df_outpatient[col] = pd.to_datetime(df_outpatient[col], errors='coerce')


# 计算住院/门诊持续时间
df_inpatient['Inpatient_ClaimDuration'] = (df_inpatient['ClaimEndDt'] - df_inpatient['ClaimStartDt']).dt.days
df_outpatient['Outpatient_ClaimDuration'] = (df_outpatient['ClaimEndDt'] - df_outpatient['ClaimStartDt']).dt.days


# 编码类别变量
label_cols = ['Gender', 'Race', 'RenalDiseaseIndicator', 'State', 'County']
for col in label_cols:
    if col in df_inpatient.columns:
        df_inpatient[col] = LabelEncoder().fit_transform(df_inpatient[col].astype(str))
    if col in df_outpatient.columns:
        df_outpatient[col] = LabelEncoder().fit_transform(df_outpatient[col].astype(str))

for col in category_columns:
    if col in df_inpatient.columns:
        df_inpatient[col] = LabelEncoder().fit_transform(df_inpatient[col].astype(str))
    if col in df_outpatient.columns:
        df_outpatient[col] = LabelEncoder().fit_transform(df_outpatient[col].astype(str))
    # if col in df_labels.columns:
    #     df_labels[col] = LabelEncoder().fit_transform(df_labels[col].astype(str))


# 聚合函数定义
def provider_aggregation(df, prefix):
    agg_dict = {
        'InscClaimAmtReimbursed': ['sum', 'mean'],
        'DeductibleAmtPaid': ['sum', 'mean'],
        f'{prefix}_ClaimDuration': ['mean', 'max'],
        # 'ClaimID':'count',
        'RenalDiseaseIndicator':'nunique',
        # 'BeneID':'count',
        # 'age_group_string': 'nunique',
        'Gender': 'nunique',
        'Race': 'nunique',
        'State': 'nunique',
        'County': 'nunique',
        'ChronicCond_Heartfailure': 'nunique',
        'ChronicCond_Diabetes': 'nunique',
        'ChronicCond_IschemicHeart': 'nunique',
        "ChronicCond_Alzheimer": 'nunique',
        "ChronicCond_KidneyDisease": 'nunique',
        "ChronicCond_Cancer": 'nunique',
        "ChronicCond_ObstrPulmonary": 'nunique',
        "ChronicCond_Depression": 'nunique',
        "ChronicCond_Osteoporasis": 'nunique',
        "ChronicCond_rheumatoidarthritis": 'nunique',
        "ChronicCond_stroke": 'nunique',
        # 'AttendingPhysician': 'nunique',
        # 'OperatingPhysician': 'nunique',
        # 'OtherPhysician': 'nunique',
        'Physician_group_String': 'nunique',
        'DiagnosisCode_group_String': 'nunique',
        'procedureCode_group_String': 'nunique',
        # 'condition_group_String': 'nunique',
    }
    # diag_cols = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
    # proc_cols = [f'ClmProcedureCode_{i}' for i in range(1, 7)]
    # for col in diag_cols + proc_cols:
    #     agg_dict[col] = 'nunique'
    grouped = df.groupby('Provider').agg(agg_dict)
    grouped.columns = [f"{prefix}_{col[0]}_{col[1]}" for col in grouped.columns]
    grouped.reset_index(inplace=True)
    return grouped


# 分别聚合 inpatient / outpatient
agg_inpatient = provider_aggregation(df_inpatient, 'Inpatient')
# print(agg_inpatient.info())
agg_outpatient = provider_aggregation(df_outpatient, 'Outpatient')
# print(agg_outpatient.head())


# 合并特征
df_provider = pd.merge(agg_inpatient, agg_outpatient, on='Provider', how='outer')
df_provider = df_provider.fillna(0)


# 合并标签
df_model = df_provider.merge(df_labels, on='Provider', how='left')
df_model['PotentialFraud'] = LabelEncoder().fit_transform(df_model['PotentialFraud'])


# 模型训练
X = df_model.drop(columns=['Provider', 'PotentialFraud'])
y = df_model['PotentialFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=3927 / 401)
model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]  # 获取正类概率
threshold = 0.42  # 默认是 0.5，可以尝试更低的值
y_pred = (y_pred_prob > threshold).astype(int)
print(classification_report(y_test, y_pred))

roc_score = roc_auc_score(y_test, y_pred_prob)
print("AUC Score:", roc_score)

print(y_train.value_counts())

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=3927 / 401,
    early_stopping_rounds=20,
    n_estimators=600,  # 多放一点，early stopping 会帮你停
    learning_rate=0.14,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.758483).astype(int)

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))


y_probs = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# 找出 f1-score 最大对应的阈值
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

print("Best threshold for max F1:", best_threshold)

y_pred_custom = (y_probs > best_threshold).astype(int)
print(classification_report(y_test, y_pred_custom))
