import geopandas as gpd
import pandas as pd
import numpy as np
import torch
import json

from shapely.geometry import box
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity




# 北京五环 SHP
shp_path = "/Users/jzq/Postgrad/work/bjshp/五环.shp"

# POI 数据
poi_path = "/Users/jzq/Postgrad/work/poi.csv"

# POI字段名
LON_COL = "longitude"      # 经度
LAT_COL = "latitude"       # 纬度
NAME_COL = "poi_name"      # POI名称
TYPE_COL = "poi_type"      # POI类型

# 栅格大小
GRID_SIZE = 500

# 输出文件
output_csv = (
    "/Users/jzq/Postgrad/work/bjshp/"
    "北京五环500m栅格_POI类型.csv"
)


ring = gpd.read_file(shp_path)

print("五环原始坐标系：", ring.crs)

if ring.crs is None:
    raise ValueError("五环 SHP 没有坐标系信息")


ring_utm = ring.to_crs(epsg=32650)

# 合并五环区域
ring_geom = ring_utm.geometry.union_all()


minx, miny, maxx, maxy = ring_geom.bounds

# 对齐到500m
minx = np.floor(minx / GRID_SIZE) * GRID_SIZE
miny = np.floor(miny / GRID_SIZE) * GRID_SIZE
maxx = np.ceil(maxx / GRID_SIZE) * GRID_SIZE
maxy = np.ceil(maxy / GRID_SIZE) * GRID_SIZE



# 创建500m × 500m栅格


grids = []
grid_x_list = []
grid_y_list = []

# grid_x：从西向东
# grid_y：从南向北
for grid_x, x in enumerate(
    np.arange(minx, maxx, GRID_SIZE)
):

    for grid_y, y in enumerate(
        np.arange(miny, maxy, GRID_SIZE)
    ):

        grids.append(
            box(
                x,
                y,
                x + GRID_SIZE,
                y + GRID_SIZE
            )
        )

        grid_x_list.append(grid_x)
        grid_y_list.append(grid_y)


grid = gpd.GeoDataFrame(
    {
        "grid_x": grid_x_list,
        "grid_y": grid_y_list,
        "geometry": grids
    },
    crs="EPSG:32650"
)


grid = grid[
    grid.geometry.centroid.within(
        ring_geom
    )
].copy()

grid.reset_index(
    drop=True,
    inplace=True
)


grid["grid_id"] = (
    "("
    + grid["grid_x"].astype(str)
    + ","
    + grid["grid_y"].astype(str)
    + ")"
)

print("\n栅格数量：", len(grid))

print(
    grid[
        [
            "grid_x",
            "grid_y",
            "grid_id"
        ]
    ].head()
)



poi_df = pd.read_csv(
    poi_path
)

print(
    "\n原始POI数量：",
    len(poi_df)
)

print(
    "POI字段：",
    poi_df.columns.tolist()
)



poi_df = poi_df.dropna(
    subset=[
        LON_COL,
        LAT_COL,
        TYPE_COL
    ]
).copy()



poi_df[LON_COL] = pd.to_numeric(
    poi_df[LON_COL],
    errors="coerce"
)

poi_df[LAT_COL] = pd.to_numeric(
    poi_df[LAT_COL],
    errors="coerce"
)

poi_df = poi_df.dropna(
    subset=[
        LON_COL,
        LAT_COL
    ]
).copy()




poi = gpd.GeoDataFrame(
    poi_df,
    geometry=gpd.points_from_xy(
        poi_df[LON_COL],
        poi_df[LAT_COL]
    ),
    crs="EPSG:4326"
)



# 转为与栅格相同坐标系
poi = poi.to_crs(
    epsg=32650
)


# POI匹配到500m栅格


poi_grid = gpd.sjoin(
    poi,
    grid[
        [
            "grid_x",
            "grid_y",
            "grid_id",
            "geometry"
        ]
    ],
    how="inner",
    predicate="within"
)

print(
    "\n成功匹配POI数量：",
    len(poi_grid)
)


# 清理POI类型
poi_grid = poi_grid.dropna(
    subset=[
        "grid_id",
        TYPE_COL
    ]
).copy()

poi_grid[TYPE_COL] = (
    poi_grid[TYPE_COL]
    .astype(str)
    .str.strip()
)

poi_grid = poi_grid[
    poi_grid[TYPE_COL] != ""
].copy()



poi_type_count = (
    poi_grid
    .groupby(
        [
            "grid_id",
            TYPE_COL
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)





model_name = (
    "模型保存地址"
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        model_name
    )
)

model = (
    AutoModel
    .from_pretrained(
        model_name
    )
)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = model.to(device)
model.eval()




# RoBERTa向量化 + Mean Pooling
def get_roberta_vector(text):

    # -------------------------
    # Tokenizer
    # -------------------------

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }
    with torch.no_grad():

        outputs = model(
            **inputs
        )
    token_embeddings = (
        outputs.last_hidden_state
    )

    attention_mask = (
        inputs["attention_mask"]
    )

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(
            token_embeddings.size()
        )
        .float()
    )
    sum_embeddings = torch.sum(
        token_embeddings * mask,
        dim=1
    )
    sum_mask = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    # 平均池化
    pooled_vector = (
        sum_embeddings
        / sum_mask
    )


    return (
        pooled_vector
        .squeeze(0)
        .cpu()
        .numpy()
    )
poi_types = (
    poi_type_count[
        TYPE_COL
    ]
    .unique()
)

print(
    "\nPOI类型数量：",
    len(poi_types)
)


type_vectors = {}

for poi_type in poi_types:

    vector = get_roberta_vector(
        poi_type
    )

    type_vectors[
        poi_type
    ] = vector

    print(
        "完成：",
        poi_type,
        "维度：",
        vector.shape
    )

grid_vectors = []
for grid_id, group in (
    poi_type_count
    .groupby("grid_id")
):

    weighted_vector = np.zeros(
        768,
        dtype=np.float32
    )


    # 当前栅格POI总数
    total_count = (
        group["count"].sum()
    )


    for _, row in group.iterrows():

        poi_type = row[
            TYPE_COL
        ]

        count = row[
            "count"
        ]

        type_vector = (
            type_vectors[
                poi_type
            ]
        )


        # 数量加权
        weighted_vector += (
            count
            * type_vector
        )


    # 加权平均
    weighted_vector = (
        weighted_vector
        / total_count
    )


    grid_vectors.append(
        {
            "grid_id":
                grid_id,

            "poi_count":
                total_count,

            "poi_vector":
                weighted_vector
        }
    )



grid_vector_df = pd.DataFrame(
    grid_vectors
)

type_names = list(
    type_vectors.keys()
)

type_matrix = np.array(
    [
        type_vectors[
            poi_type
        ]
        for poi_type
        in type_names
    ]
)


grid_matrix = np.array(
    grid_vector_df[
        "poi_vector"
    ].tolist()
)


# 计算余弦相似度
similarity_matrix = cosine_similarity(
    grid_matrix,
    type_matrix
)
best_index = np.argmax(
    similarity_matrix,
    axis=1
)


grid_type_df = pd.DataFrame(
    {
        "grid_id":
            grid_vector_df[
                "grid_id"
            ].values,

        "poi_count":
            grid_vector_df[
                "poi_count"
            ].values,

        "poi_vector":
            grid_vector_df[
                "poi_vector"
            ].values,

        "final_poi_type":
            [
                type_names[i]
                for i
                in best_index
            ]
    }
)



grid_result = grid.merge(
    grid_type_df,
    on="grid_id",
    how="left"
)
grid_result[
    "poi_count"
] = (
    grid_result[
        "poi_count"
    ]
    .fillna(0)
    .astype(int)
)


grid_center = grid_result[
    [
        "grid_id",
        "geometry"
    ]
].copy()


grid_center[
    "geometry"
] = (
    grid_center
    .geometry
    .centroid
)


grid_center_wgs84 = (
    grid_center
    .to_crs(
        epsg=4326
    )
)

grid_result[
    "center_lon"
] = (
    grid_center_wgs84
    .geometry
    .x
    .values
)

grid_result[
    "center_lat"
] = (
    grid_center_wgs84
    .geometry
    .y
    .values
)


grid_result_wgs84 = (
    grid_result
    .to_crs(
        epsg=4326
    )
)


# 保存
grid_save = (
    grid_result_wgs84
    .copy()
)


def vector_to_json(vector):

    if isinstance(
        vector,
        np.ndarray
    ):

        return json.dumps(
            vector
            .astype(float)
            .tolist(),
            ensure_ascii=False
        )

    return None


grid_save[
    "poi_vector"
] = (
    grid_save[
        "poi_vector"
    ]
    .apply(
        vector_to_json
    )
)


grid_save = grid_save[
    [
        "grid_x",
        "grid_y",
        "grid_id",

        "center_lon",
        "center_lat",

        "poi_count",

        "poi_vector",

        "final_poi_type",

        "geometry"
    ]
]




# 保存CSV
grid_save.drop(
    columns="geometry"
).to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nCSV保存成功："
)

print(
    output_csv
)


