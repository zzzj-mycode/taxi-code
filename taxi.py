import pandas as pd
import geopandas as gpd
import numpy as np
import json
from shapely.geometry import box


# 前面保存的POI栅格类型CSV
poitype_path = (
    "保存的栅格信息csv文件"
)
od_path = (
    "od数据路径"
)
output_path = (
    "结果路径"
)


O_LON = "o_lon"
O_LAT = "o_lat"
D_LON = "d_lon"
D_LAT = "d_lat"
START_TIME = "start_time"
END_TIME = "end_time"


# 栅格大小
GRID_SIZE = 500


# 读取POI栅格类型数据
poitype = pd.read_csv(
    poitype_path
)



# 将保存的POI向量字符串恢复成数组
if "poi_vector" in poitype.columns:

    poitype["poi_vector"] = (
        poitype["poi_vector"]
        .apply(
            lambda x: np.array(
                json.loads(x),
                dtype=np.float32
            )
            if isinstance(x, str)
            and x.strip()
            and x.strip().lower() != "nan"
            else None
        )
    )
grid_center = gpd.GeoDataFrame(
    poitype.copy(),

    geometry=gpd.points_from_xy(
        poitype["center_lon"],
        poitype["center_lat"]
    ),

    crs="EPSG:4326"
)

# 单位为米
grid_center = grid_center.to_crs(
    epsg=32650
)


# 中心点生成500m × 500m栅格
half_size = GRID_SIZE / 2
grid_center["geometry"] = (
    grid_center.geometry.apply(
        lambda p: box(
            p.x - half_size,
            p.y - half_size,
            p.x + half_size,
            p.y + half_size
        )
    )
)


grid = gpd.GeoDataFrame(
    grid_center,
    geometry="geometry",
    crs="EPSG:32650"
)


# 读取出租车OD数据

od = pd.read_csv(
    od_path
)

coordinate_cols = [
    O_LON,
    O_LAT,
    D_LON,
    D_LAT
]

for col in coordinate_cols:

    od[col] = pd.to_numeric(
        od[col],
        errors="coerce"
    )


# 删除经纬度为空的数据
od = od.dropna(
    subset=coordinate_cols
).copy()


od[START_TIME] = pd.to_datetime(
    od[START_TIME],
    errors="coerce"
)

od[END_TIME] = pd.to_datetime(
    od[END_TIME],
    errors="coerce"
)


od = od.reset_index(
    drop=True
)

od["od_id"] = np.arange(
    len(od)
)


origin = gpd.GeoDataFrame(
    od[
        [
            "od_id",
            O_LON,
            O_LAT
        ]
    ].copy(),

    geometry=gpd.points_from_xy(
        od[O_LON],
        od[O_LAT]
    ),

    crs="EPSG:4326"
)


origin = origin.to_crs(
    epsg=32650
)



origin_grid = gpd.sjoin(
    origin,

    grid[
        [
            "grid_id",
            "grid_x",
            "grid_y",
            "center_lon",
            "center_lat",
            "poi_count",
            "poi_vector",
            "final_poi_type",
            "geometry"
        ]
    ],

    how="inner",

    predicate="within"
)


origin_grid = origin_grid.rename(
    columns={
        "grid_id":
            "origin_grid_id",

        "grid_x":
            "origin_grid_x",

        "grid_y":
            "origin_grid_y",

        "center_lon":
            "origin_center_lon",

        "center_lat":
            "origin_center_lat",

        "poi_count":
            "origin_poi_count",

        "poi_vector":
            "origin_poi_vector",

        "final_poi_type":
            "origin_type"
    }
)


origin_grid = origin_grid[
    [
        "od_id",

        "origin_grid_id",
        "origin_grid_x",
        "origin_grid_y",

        "origin_center_lon",
        "origin_center_lat",

        "origin_poi_count",
        "origin_poi_vector",

        "origin_type"
    ]
]


print(
    "\n成功匹配起点数量：",
    len(origin_grid)
)

destination = gpd.GeoDataFrame(
    od[
        [
            "od_id",
            D_LON,
            D_LAT
        ]
    ].copy(),

    geometry=gpd.points_from_xy(
        od[D_LON],
        od[D_LAT]
    ),

    crs="EPSG:4326"
)


destination = destination.to_crs(
    epsg=32650
)


destination_grid = gpd.sjoin(
    destination,

    grid[
        [
            "grid_id",
            "grid_x",
            "grid_y",
            "center_lon",
            "center_lat",
            "poi_count",
            "final_poi_type",
            "geometry"
        ]
    ],

    how="inner",

    predicate="within"
)


destination_grid = destination_grid.rename(
    columns={
        "grid_id":
            "dest_grid_id",

        "grid_x":
            "dest_grid_x",

        "grid_y":
            "dest_grid_y",

        "center_lon":
            "dest_center_lon",

        "center_lat":
            "dest_center_lat",

        "final_poi_type":
            "dest_type"
    }
)


destination_grid = destination_grid[
    [
        "od_id",

        "dest_grid_id",
        "dest_grid_x",
        "dest_grid_y",

        "dest_center_lon",
        "dest_center_lat",

        "dest_type"
    ]
]

out_count = (
    origin_grid
    .groupby(
        "origin_grid_id"
    )
    .size()
    .reset_index(
        name="out_count"
    )
)

out_count = out_count.rename(
    columns={
        "origin_grid_id":
            "grid_id"
    }
)


in_count = (
    destination_grid
    .groupby(
        "dest_grid_id"
    )
    .size()
    .reset_index(
        name="in_count"
    )
)

in_count = in_count.rename(
    columns={
        "dest_grid_id":
            "grid_id"
    }
)

grid_od_count = (
    poitype[
        ["grid_id"]
    ]
    .copy()
)


grid_od_count = grid_od_count.merge(
    out_count,
    on="grid_id",
    how="left"
)


grid_od_count = grid_od_count.merge(
    in_count,
    on="grid_id",
    how="left"
)


grid_od_count[
    "out_count"
] = (
    grid_od_count[
        "out_count"
    ]
    .fillna(0)
    .astype(int)
)


grid_od_count[
    "in_count"
] = (
    grid_od_count[
        "in_count"
    ]
    .fillna(0)
    .astype(int)
)



grid_od_count[
    "od_count"
] = (
    grid_od_count[
        "out_count"
    ]
    +
    grid_od_count[
        "in_count"
    ]
)



od_result = od.merge(
    origin_grid,
    on="od_id",
    how="inner"
)


od_result = od_result.merge(
    destination_grid,
    on="od_id",
    how="inner"
)

grid_od_count_origin = (
    grid_od_count.rename(
        columns={
            "grid_id":
                "origin_grid_id"
        }
    )
)


od_result = od_result.merge(
    grid_od_count_origin,
    on="origin_grid_id",
    how="left"
)


# 中心经纬度转为GeoDataFrame
origin_centers = gpd.GeoDataFrame(
    od_result[
        [
            "od_id"
        ]
    ].copy(),

    geometry=gpd.points_from_xy(
        od_result[
            "origin_center_lon"
        ],
        od_result[
            "origin_center_lat"
        ]
    ),

    crs="EPSG:4326"
)


dest_centers = gpd.GeoDataFrame(
    od_result[
        [
            "od_id"
        ]
    ].copy(),

    geometry=gpd.points_from_xy(
        od_result[
            "dest_center_lon"
        ],
        od_result[
            "dest_center_lat"
        ]
    ),

    crs="EPSG:4326"
)


# 转换成米制坐标系
origin_centers = origin_centers.to_crs(
    epsg=32650
)

dest_centers = dest_centers.to_crs(
    epsg=32650
)


distance_m = np.sqrt(

    (
        origin_centers.geometry.x.values
        -
        dest_centers.geometry.x.values
    ) ** 2

    +

    (
        origin_centers.geometry.y.values
        -
        dest_centers.geometry.y.values
    ) ** 2
)


od_result[
    "distance_km"
] = (
    distance_m / 1000
)


# 保留3位小数
od_result[
    "distance_km"
] = (
    od_result[
        "distance_km"
    ]
    .round(3)
)

result = od_result[
    [
        "od_id",

        "origin_grid_id",

        "origin_grid_x",
        "origin_grid_y",

        "origin_center_lon",
        "origin_center_lat",

        "origin_poi_count",

        "origin_poi_vector",

        "origin_type",

        "out_count",
        "in_count",
        "od_count",

        "dest_grid_id",

        "dest_grid_x",
        "dest_grid_y",

        "dest_center_lon",
        "dest_center_lat",

        "dest_type",

        START_TIME,

        END_TIME,

        "distance_km"
    ]
].copy()


result[
    "origin_poi_vector"
] = (
    result[
        "origin_poi_vector"
    ]
    .apply(
        lambda x:
        json.dumps(
            x.astype(float).tolist(),
            ensure_ascii=False
        )
        if isinstance(
            x,
            np.ndarray
        )
        else None
    )
)

result.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print(
    "\n最终数据保存成功："
)
