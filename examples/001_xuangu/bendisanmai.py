# -*- coding: utf-8 -*-

import czsc
from pathlib import Path

# 定义本地数据路径
data_path = Path(r"D:\001_DATA\Test\stk_data\d_tushare")

# 读取本地数据
import os
symbols = [x.split('.')[0] for x in os.listdir(data_path) if x.endswith('.csv')]

def read_bars(symbol):
    df = pd.read_csv(data_path / f"{symbol}.csv", index_col=0)
    return df

# 定义选股事件
events = [
    {
        "operate": "开多",
        "factors": [{"name": "三买形态", "signals_all": ["日线_D1#SMA#34_BS3辅助V230319_三买_均线底分_任意_0"]}]
    }
]

# 执行回测
ems = czsc.EventMatchSensor(
    events=events,
    symbols=symbols,
    read_bars=read_bars,
    start_dt="20150101",
    end_dt="20220901",
    results_path=r"D:\results",
)

# 处理结果
df = ems.data
df = df[df["三买形态"]==1][["symbol", "dt"]]

print(df)