import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


DATA_PATH = "读取保存的文件"
RANDOM_STATE = 42
TYPE_LIST = [
    "交通设施",
    "体育休闲",
    "公共设施",
    "公司企业",
    "医疗保健",
    "餐饮服务",
    "购物服务",
    "商务住宅"
]


TYPE_MAPPING = {
    name: i
    for i, name in enumerate(TYPE_LIST)
}
for name, number in TYPE_MAPPING.items():
    print(f"{name} -> {number}")


data = pd.read_csv(DATA_PATH)

print("\n原始数据量：", len(data))


# 普通数值特征
NUMERIC_FEATURES = [
    "od_count",             # 起点栅格总OD数量
    "origin_center_lat",    # 起点栅格中心纬度
    "origin_center_lon",    # 起点栅格中心经度
    "distance_km",          # 起终点栅格距离
    "dest_center_lat",      # 到达栅格中心纬度
    "dest_center_lon"       # 到达栅格中心经度
]


ORIGIN_TYPE_COL = "origin_type"
VECTOR_COL = "origin_poi_vector"
START_TIME_COL = "start_time"

# 模型预测目标
TARGET_COL = "dest_type"

required_cols = (
    NUMERIC_FEATURES
    + [
        ORIGIN_TYPE_COL,
        VECTOR_COL,
        TARGET_COL
    ]
)

data = data.dropna(
    subset=required_cols
).copy()


for col in NUMERIC_FEATURES:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )


data = data.dropna(
    subset=NUMERIC_FEATURES
).copy()


data[ORIGIN_TYPE_COL] = (
    data[ORIGIN_TYPE_COL]
    .astype(str)
    .str.strip()
)

data[TARGET_COL] = (
    data[TARGET_COL]
    .astype(str)
    .str.strip()
)

data = data[
    data[ORIGIN_TYPE_COL].isin(TYPE_LIST)
    &
    data[TARGET_COL].isin(TYPE_LIST)
].copy()

if START_TIME_COL not in data.columns:
    raise ValueError(f"缺少 {START_TIME_COL} 列，无法区分工作日和非工作日")

data[START_TIME_COL] = pd.to_datetime(data[START_TIME_COL], errors="coerce")
invalid_start_times = data[START_TIME_COL].isna().sum()
if invalid_start_times:
    print(f"排除 {invalid_start_times} 条出发时间缺失或无效的记录")
    data = data.dropna(subset=[START_TIME_COL]).copy()

def parse_vector(value):

    try:

        # 已经是list
        if isinstance(value, list):
            vector = np.array(
                value,
                dtype=np.float32
            )

        elif isinstance(value, str):

            vector = np.array(
                json.loads(value),
                dtype=np.float32
            )

        else:
            return None

        return vector

    except Exception:
        return None

data[VECTOR_COL] = (
    data[VECTOR_COL]
    .apply(parse_vector)
)

data["target"] = (
    data[TARGET_COL]
    .map(TYPE_MAPPING)
)


def split_rows(rows, holdout_count):
    if holdout_count == 0:
        return rows, rows.iloc[:0]

    class_counts = rows["target"].value_counts()
    can_stratify = (
        class_counts.min() >= 2
        and len(class_counts) <= min(holdout_count, len(rows) - holdout_count)
    )
    return train_test_split(
        rows,
        test_size=holdout_count,
        random_state=RANDOM_STATE,
        stratify=rows["target"] if can_stratify else None
    )


def split_day_group(rows):
    train_count = round(len(rows) * 0.70)
    test_count = round(len(rows) * 0.10)
    group_train, remaining = split_rows(rows, len(rows) - train_count)
    group_val, group_test = split_rows(remaining, test_count)
    return group_train, group_val, group_test


is_workday = data[START_TIME_COL].dt.dayofweek < 5
workday_splits = split_day_group(data.loc[is_workday])
non_workday_splits = split_day_group(data.loc[~is_workday])

train_data, val_data, test_data = [
    pd.concat([workday_split, non_workday_split], ignore_index=True)
    for workday_split, non_workday_split in zip(workday_splits, non_workday_splits)
]

if any(split.empty for split in (train_data, val_data, test_data)):
    raise ValueError("有效行程数据不足，无法划分出非空的训练集、验证集和测试集")


print("\n==============================")

print(
    "总数据量：",
    len(data)
)

print(
    "训练集：",
    len(train_data),
    f"({len(train_data) / len(data):.2%})"
)

print(
    "验证集：",
    len(val_data),
    f"({len(val_data) / len(data):.2%})"
)

print(
    "测试集：",
    len(test_data),
    f"({len(test_data) / len(data):.2%})"
)

print("==============================")

X_train_numeric = (
    train_data[
        NUMERIC_FEATURES
    ]
    .astype(float)
    .reset_index(drop=True)
)


X_val_numeric = (
    val_data[
        NUMERIC_FEATURES
    ]
    .astype(float)
    .reset_index(drop=True)
)


X_test_numeric = (
    test_data[
        NUMERIC_FEATURES
    ]
    .astype(float)
    .reset_index(drop=True)
)

try:

    origin_encoder = OneHotEncoder(
        categories=[TYPE_LIST],
        handle_unknown="ignore",
        sparse_output=False
    )

except TypeError:

    # 兼容旧版sklearn
    origin_encoder = OneHotEncoder(
        categories=[TYPE_LIST],
        handle_unknown="ignore",
        sparse=False
    )


# 只能在训练集fit
train_origin_type = (
    origin_encoder.fit_transform(
        train_data[
            [ORIGIN_TYPE_COL]
        ]
    )
)


# 验证集、测试集只transform
val_origin_type = (
    origin_encoder.transform(
        val_data[
            [ORIGIN_TYPE_COL]
        ]
    )
)


test_origin_type = (
    origin_encoder.transform(
        test_data[
            [ORIGIN_TYPE_COL]
        ]
    )
)


origin_type_columns = (
    origin_encoder
    .get_feature_names_out(
        [ORIGIN_TYPE_COL]
    )
)


X_train_type = pd.DataFrame(
    train_origin_type,
    columns=origin_type_columns
)


X_val_type = pd.DataFrame(
    val_origin_type,
    columns=origin_type_columns
)


X_test_type = pd.DataFrame(
    test_origin_type,
    columns=origin_type_columns
)


print(
    "\n起点类型One-Hot字段："
)

print(
    origin_type_columns.tolist()
)

X_train_vector = np.vstack(
    train_data[VECTOR_COL].values
)

X_val_vector = np.vstack(
    val_data[VECTOR_COL].values
)

X_test_vector = np.vstack(
    test_data[VECTOR_COL].values
)


vector_scaler = StandardScaler()


X_train_vector_scaled = (
    vector_scaler.fit_transform(
        X_train_vector
    )
)


X_val_vector_scaled = (
    vector_scaler.transform(
        X_val_vector
    )
)


X_test_vector_scaled = (
    vector_scaler.transform(
        X_test_vector
    )
)
pca = PCA(
    n_components=0.95
)
X_train_pca_array = (
    pca.fit_transform(
        X_train_vector_scaled
    )
)


X_val_pca_array = (
    pca.transform(
        X_val_vector_scaled
    )
)


X_test_pca_array = (
    pca.transform(
        X_test_vector_scaled
    )
)




# PCA结果转换为DataFrame
PCA_COLUMNS = [
    f"PC{i + 1}"
    for i in range(
        X_train_pca_array.shape[1]
    )
]

X_train_pca = pd.DataFrame(
    X_train_pca_array,
    columns=PCA_COLUMNS
)

X_val_pca = pd.DataFrame(
    X_val_pca_array,
    columns=PCA_COLUMNS
)

X_test_pca = pd.DataFrame(
    X_test_pca_array,
    columns=PCA_COLUMNS
)

X_train = pd.concat(
    [
        X_train_numeric,
        X_train_type,
        X_train_pca
    ],
    axis=1
)


X_val = pd.concat(
    [
        X_val_numeric,
        X_val_type,
        X_val_pca
    ],
    axis=1
)


X_test = pd.concat(
    [
        X_test_numeric,
        X_test_type,
        X_test_pca
    ],
    axis=1
)


# 标签
y_train = (
    train_data[
        "target"
    ].values
)

y_val = (
    val_data[
        "target"
    ].values
)

y_test = (
    test_data[
        "target"
    ].values
)



model = XGBClassifier(

    # 8分类
    objective="multi:softprob",

    num_class=len(
        TYPE_LIST
    ),
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",

    random_state=RANDOM_STATE,

    n_jobs=-1
)

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_val, y_val)
    ],

    verbose=False
)


y_pred = model.predict(X_test)

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(TYPE_LIST))),
    target_names=TYPE_LIST,
    output_dict=True,
    zero_division=0
)

metrics_df = pd.DataFrame([
    {
        "类别": class_name,
        "精确率": report[class_name]["precision"],
        "召回率": report[class_name]["recall"],
        "F1值": report[class_name]["f1-score"]
    }
    for class_name in TYPE_LIST
])

print(metrics_df.to_string(index=False, float_format="{:.4f}".format))
print(f"测试集整体准确率：{accuracy_score(y_test, y_pred):.4f}")
