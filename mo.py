import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

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
ID_TO_TYPE = {
    i: name
    for name, i in TYPE_MAPPING.items()
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


train_data, temp_data = train_test_split(
    data,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=data["target"]
)


val_data, test_data = train_test_split(
    temp_data,
    test_size=1/3,
    random_state=RANDOM_STATE,
    stratify=temp_data["target"]
)


train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)


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


print(
    "\nXGBoost训练完成"
)


train_pred = model.predict(
    X_train
)

train_accuracy = accuracy_score(
    y_train,
    train_pred
)

val_pred = model.predict(
    X_val
)

val_accuracy = accuracy_score(
    y_val,
    val_pred
)

y_pred = model.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\n=================================="
)

print(
    f"训练集准确率：{train_accuracy:.4f}"
)

print(
    f"验证集准确率：{val_accuracy:.4f}"
)

print(
    f"最终测试集准确率：{test_accuracy:.4f}"
)

print(
    f"最终测试集准确率：{test_accuracy * 100:.2f}%"
)

print(
    "=================================="
)

report = classification_report(
    y_test,
    y_pred,

    labels=list(
        range(len(TYPE_LIST))
    ),

    target_names=TYPE_LIST,

    output_dict=True,

    zero_division=0
)


metrics_rows = []


for class_name in TYPE_LIST:

    metrics_rows.append({

        "类别":
            class_name,

        "精确率":
            report[
                class_name
            ]["precision"],

        "召回率":
            report[
                class_name
            ]["recall"],

        "F1值":
            report[
                class_name
            ]["f1-score"],

        "样本数":
            int(
                report[
                    class_name
                ]["support"]
            )
    })


metrics_df = pd.DataFrame(
    metrics_rows
)


# 保留两位小数
metrics_df[
    [
        "精确率",
        "召回率",
        "F1值"
    ]
] = (
    metrics_df[
        [
            "精确率",
            "召回率",
            "F1值"
        ]
    ]
    .round(2)
)


print(
    "\n各类别评价指标：\n"
)

print(
    metrics_df.to_string(
        index=False
    )
)



print(
    "\n宏平均："
)

print(
    f"Precision = "
    f"{report['macro avg']['precision']:.4f}"
)

print(
    f"Recall    = "
    f"{report['macro avg']['recall']:.4f}"
)

print(
    f"F1        = "
    f"{report['macro avg']['f1-score']:.4f}"
)


print(
    "\n加权平均："
)

print(
    f"Precision = "
    f"{report['weighted avg']['precision']:.4f}"
)

print(
    f"Recall    = "
    f"{report['weighted avg']['recall']:.4f}"
)

print(
    f"F1        = "
    f"{report['weighted avg']['f1-score']:.4f}"
)




cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(
        range(len(TYPE_LIST))
    )
)


cm_df = pd.DataFrame(
    cm,
    index=TYPE_LIST,
    columns=TYPE_LIST
)


print(
    "\n混淆矩阵：\n"
)

print(
    cm_df
)




y_probability = (
    model.predict_proba(
        X_test
    )
)


# 每条样本最大预测概率
max_probability = (
    y_probability.max(
        axis=1
    )
)


actual_type = [
    ID_TO_TYPE[
        int(x)
    ]
    for x in y_test
]


predicted_type = [
    ID_TO_TYPE[
        int(x)
    ]
    for x in y_pred
]




prediction_result = pd.DataFrame({

    "origin_grid_id":
        test_data[
            "origin_grid_id"
        ].values,

    "origin_type":
        test_data[
            "origin_type"
        ].values,

    "dest_grid_id":
        test_data[
            "dest_grid_id"
        ].values,

    "distance_km":
        test_data[
            "distance_km"
        ].values,

    "实际终点类型":
        actual_type,

    "预测终点类型":
        predicted_type,

    "预测概率":
        max_probability
})


prediction_result[
    "预测正确"
] = (
    prediction_result[
        "实际终点类型"
    ]
    ==
    prediction_result[
        "预测终点类型"
    ]
)


print(
    "\n预测结果示例：\n"
)

print(
    prediction_result.head(
        20
    ).to_string(
        index=False
    )
)


print(
    f"最终测试集准确率："
    f"{test_accuracy * 100:.2f}%"
)
