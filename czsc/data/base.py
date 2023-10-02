# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2022/2/24 16:17
describe: 基础工具
"""
#该代码实现的功能是将中文频率转换为英文频率。具体步骤如下：

#1. 创建一个名为`freq_cn2ts`的字典。
#2. 字典中的键是中文频率，值是对应的英文频率。
#3. 中文频率包括："1分钟"、"5分钟"、"15分钟"、"30分钟"、"60分钟"、"日线"、"周线"、"月线"。
#4. 对应的英文频率分别是："1min"、"5min"、"15min"、"30min"、"60min"、"D"、"W"、"M"。

freq_cn2ts = {"1分钟": "1min", "5分钟": "5min", "15分钟": "15min", "30分钟": "30min",
              '60分钟': "60min", "日线": "D", "周线": "W", "月线": "M"}
#该代码实现的功能是将一个字典中的键和值进行交换，并创建一个新的字典。
freq_ts2cn = {v: k for k, v in freq_cn2ts.items()}

freq_cn2jq = freq_cn2ts
freq_jq2cn = freq_ts2cn

freq_gm2cn = {"60s": "1分钟", "300s": "5分钟", "900s": "15分钟", "1800s": "30分钟",
              "3600s": "60分钟", "1d": "日线"}
freq_cn2gm = {v: k for k, v in freq_gm2cn.items()}

def jq_symbol_to_gm(symbol: str) -> str:
    """聚宽代码转掘金代码"""
    code, exchange = symbol.split(".")
    if exchange == 'XSHG':
        gm_symbol = "SHSE." + code
    elif exchange == 'XSHE':
        gm_symbol = "SZSE." + code
    else:
        raise ValueError
    return gm_symbol


def jq_symbol_to_ts(symbol):
    """将聚宽代码转成 Tushare 代码"""
    symbol = jq_symbol_to_gm(symbol)
    symbol = gm_symbol_to_ts(symbol)
    return symbol


def jq_symbol_to_tdx(symbol):
    """将聚宽的代码转成通达信代码"""
    code, exchange = symbol.split(".")
    if exchange == 'XSHE':
        tdx_symbol = "0" + code
    elif exchange == 'XSHG':
        tdx_symbol = "1" + code
    else:
        print(f"{exchange} - {code} is not supported.")
        return None
    return tdx_symbol


def gm_symbol_to_jq(symbol: str) -> str:
    """掘金代码转聚宽代码"""
    exchange, code = symbol.split(".")
    if exchange == 'SHSE':
        jq_symbol = code + ".XSHG"
    elif exchange == 'SZSE':
        jq_symbol = code + ".XSHE"
    else:
        raise ValueError
    return jq_symbol


def gm_symbol_to_ts(symbol: str) -> str:
    """掘金代码转Tushare代码"""
    exchange, code = symbol.split(".")
    if exchange == 'SHSE':
        ts_symbol = code + ".SH"
    elif exchange == 'SZSE':
        ts_symbol = code + ".SZ"
    else:
        raise ValueError
    return ts_symbol


def gm_symbol_to_tdx(symbol: str) -> str:
    """掘金代码转通达信代码"""
    symbol = gm_symbol_to_jq(symbol)
    symbol = jq_symbol_to_tdx(symbol)
    return symbol


def tdx_symbol_to_jq(symbol):
    """将通达信的代码转成聚宽代码"""
    exchange, code = symbol[0], symbol[1:]
    if exchange == '0':
        jq_symbol = code + ".XSHE"
    elif exchange == '1':
        jq_symbol = code + ".XSHG"
    else:
        print(f"{exchange} - {code} is not supported.")
        return None
    return jq_symbol


def tdx_symbol_to_gm(symbol):
    """将通达信的代码转成掘金代码"""
    symbol = tdx_symbol_to_jq(symbol)
    symbol = jq_symbol_to_gm(symbol)
    return symbol


def tdx_symbol_to_ts(symbol):
    """将通达信的代码转成 Tushare 代码"""
    symbol = tdx_symbol_to_jq(symbol)
    symbol = jq_symbol_to_ts(symbol)
    return symbol


def ts_symbol_to_gm(symbol):
    """将 Tushare 代码转成掘金代码"""
    code, ex = symbol.split(".")
    if ex == 'SH':
        gm_symbol = "SHSE." + code
    elif ex == 'SZ':
        gm_symbol = "SZSE." + code
    else:
        raise ValueError
    return gm_symbol


def ts_symbol_to_jq(symbol):
    """将 Tushare 代码转成聚宽代码"""
    symbol = ts_symbol_to_gm(symbol)
    symbol = gm_symbol_to_jq(symbol)
    return symbol


def ts_symbol_to_tdx(symbol):
    """将 Tushare 代码转成通达信代码"""
    symbol = ts_symbol_to_jq(symbol)
    symbol = jq_symbol_to_tdx(symbol)
    return symbol


def save_symbols_to_ebk(symbols, file_ebk, source='ts'):
    """
    将股票代码列表保存到 EBK 文件，用来导入标的到同花顺、通达信软件中

    :param symbols: 股票代码列表
    :param file_ebk: EBK结果文件
    :param source: 代码格式，可选值为 'ts'、'jq' 或 'gm'
    :return: 无返回值
    """
    # 将source参数转换为小写
    source = source.lower()

    # 根据source参数选择符号映射函数
    if source == 'ts':
        symbol_to_tdx = ts_symbol_to_tdx
    elif source == 'jq':
        symbol_to_tdx = jq_symbol_to_tdx
    elif source == 'gm':
        symbol_to_tdx = gm_symbol_to_tdx
    else:
        # 如果source参数不在可选值中，引发ValueError异常
        raise ValueError

    # 使用符号映射函数将股票代码列表转换为通达信格式的代码列表
    tdx_symbols = [symbol_to_tdx(ts_code) for ts_code in symbols]

    # 打开EBK结果文件，并将转换后的代码列表写入文件中
    with open(file_ebk, encoding='utf-8', mode='w') as f:
        f.write("\n".join(tdx_symbols))

