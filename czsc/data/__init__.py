# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2022/2/24 16:17
describe: 数据工具
"""

from .ts_cache import TsDataCache
from . import ts
from .base import *


def get_symbols(dc: TsDataCache, step):
    """获取择时策略投研不同阶段对应的标的列表

    :param dc: 数据缓存
    :param step: 投研阶段
    :return:
    这段代码主要是定义了一个函数`get_symbols`，用于获取不同投研阶段对应的标的列表。函数接受两个参数：`dc`表示数据缓存对象，`step`表示投研阶段。

    在函数内部，首先通过`dc.stock_basic()`获取股票基本信息，并将其赋值给变量`stocks`。然后，根据`list_date`在`stocks`数据中选择出所有在"2010-01-01"之前上市的股票，并将其ts_code转成列表形式，赋值给变量`stocks_`。

    接下来，定义了一个字典`stocks_map`，其中存储了不同阶段对应的标的列表。例如，`"index"`对应的是一些指数的标的代码列表，`"stock"`对应的是所有股票的标的代码列表，以此类推。

    然后，定义了另一个字典`asset_map`，存储了不同阶段对应的资产类别。例如，`"index"`对应的是指数类资产，`"stock"`对应的是股票类资产，`"etfs"`对应的是ETF类资产，以此类推。

    接着，根据给定的投研阶段`step`，从`stocks_map`中取出对应的标的列表，并与对应的资产类别进行拼接成形如"标的代码#资产类别"的字符串，将这些字符串存入`symbols`列表中。

    最后，将`symbols`列表作为函数的返回值。

    总结起来，这段代码的作用是根据给定的投研阶段，获取该阶段对应的标的列表，并返回这些标的的代码和资产类别的组合字符串列表。
    """
    stocks = dc.stock_basic()
    stocks_ = stocks[stocks['list_date'] < '2010-01-01'].ts_code.to_list()
    stocks_map = {
        "index": ['000905.SH', '000016.SH', '000300.SH', '000001.SH', '000852.SH',
                  '399001.SZ', '399006.SZ', '399376.SZ', '399377.SZ', '399317.SZ', '399303.SZ'],
        "stock": stocks.ts_code.to_list(),
        "check": ['000001.SZ'],
        "train": stocks_[:200],
        "valid": stocks_[200:600],
        "etfs": ['512880.SH', '518880.SH', '515880.SH', '513050.SH', '512690.SH',
                 '512660.SH', '512400.SH', '512010.SH', '512000.SH', '510900.SH',
                 '510300.SH', '510500.SH', '510050.SH', '159992.SZ', '159985.SZ',
                 '159981.SZ', '159949.SZ', '159915.SZ'],
    }

    asset_map = {
        "index": "I",
        "stock": "E",
        "check": "E",
        "train": "E",
        "valid": "E",
        "etfs": "FD"
    }
    asset = asset_map[step]
    symbols = [f"{ts_code}#{asset}" for ts_code in stocks_map[step]]
    return symbols



