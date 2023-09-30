# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2023/6/24 18:49
describe: Tushare数据源
"""
import os
from czsc import data

dc = data.TsDataCache(data_path=os.environ.get('ts_data_path', r'D:\ts_data'))

# 从data模块中导入TsDataCache类
# 创建TsDataCache类的一个实例dc,并传入参数data_path
# data_path参数使用os.environ.get()方法获取环境变量'ts_data_path'的值
# 如果环境变量不存在,则使用默认值r'D:\ts_data'
# 这样可以通过设置环境变量来配置cache的保存路径,如果未设置则使用默认路径










def get_symbols(step):
    # 定义函数get_symbols,参数为step

    if step.upper() == 'ALL':
        # 如果step转换为大写后等于'ALL'

        return data.get_symbols(dc, 'index') + data.get_symbols(dc, 'stock') + data.get_symbols(dc, 'etfs')
        # 则返回获取指数、股票、ETF的所有代码列表的拼接
        # 通过data模块的get_symbols方法,传入dc和类型获取不同类型的代码列表

    else:
        return data.get_symbols(dc, step)
        # 否则就直接通过data模块的get_symbols方法,传入dc和step获取指定类型的代码列表

    # 这样可以通过传入'ALL'获取所有代码,或者传入类型如'index'获取指定类型代码











def get_raw_bars(symbol, freq, sdt, edt, fq='后复权', raw_bar=True):
    """读取本地数据"""

    ts_code, asset = symbol.split("#")
    # 将symbol字符串以#拆分为ts_code和asset两部分

    freq = str(freq)
    adj = "qfq" if fq == "前复权" else "hfq"
    # 根据复权方式fq设置adj参数为qfq或hfq

    if "分钟" in freq:
        freq = freq.replace("分钟", "min")
        bars = dc.pro_bar_minutes(ts_code, sdt=sdt, edt=edt, freq=freq, asset=asset, adj=adj, raw_bar=raw_bar)
        # 如果频率包含“分钟”,则将“分钟”替换为“min”,调用pro_bar_minutes方法

    else:
        _map = {"日线": "D", "周线": "W", "月线": "M"}
        freq = _map[freq]
        bars = dc.pro_bar(ts_code, start_date=sdt, end_date=edt, freq=freq, asset=asset, adj=adj, raw_bar=raw_bar)
        # 否则使用频率映射表转换为D/W/M
        # 调用pro_bar方法

    return bars
    # 返回读取的数据