# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2021/6/25 18:52
"""
import time # 导入time模块,用于时间相关操作
import json # 导入json模块,用于JSON数据序列化和反序列化
import requests # 导入requests模块,用于发送网络请求获取数据
import pandas as pd # 导入pandas模块,用于数据分析,pd为别名
import tushare as ts # 导入tushare模块,用于获取股票数据,ts为别名
from deprecated import deprecated # 从deprecated模块中导入deprecated装饰器,用于标记不推荐使用的函数
from datetime import datetime # 从datetime模块中导入datetime类,用于日期时间处理
from typing import List # 从typing模块中导入List,用于标注类型注解
from functools import partial # 从functools模块中导入partial,用于偏函数
from loguru import logger # 从loguru模块中导入logger,用于日志记录
from tenacity import retry, stop_after_attempt, wait_random # 从tenacity模块中导入重试机制相关功能
from czsc.objects import RawBar, Freq # 从czsc模块中导入RawBar和Freq类



'''
这是一个使用Python进行股票数据获取和处理的代码示例。
导入了许多数据分析和网络请求相关的模块,如pandas、requests等。
定义了一个TushareProApi类,封装了调用Tushare Pro接口获取股票K线数据的功能。
定义了频率与字符串之间的映射关系,以及一个标准的K线结构RawBar。
具体的get_kline函数可以获取指定股票在指定时间范围内的K线数据。
获取到原始数据后会进行转换,比如日线的数据里成交量和成交额需要调整单位。
然后将原始数据装换成RawBar对象存入列表。
为很多函数、类添加了详细的注释,解释了功能、参数、返回值等,利于理解。
使用了一些Python的高级特性,如装饰器、偏函数等。
代码结构清晰,有明确的函数分工,体现了一定的代码设计能力。
'''
# 数据频度 ：支持分钟(min)/日(D)/周(W)/月(M)K线，其中1min表示1分钟（类推1/5/15/30/60分钟）。
# 对于分钟数据有600积分用户可以试用（请求2次），正式权限请在QQ群私信群主或积分管理员。
freq_map = {Freq.F1: "1min", Freq.F5: '5min', Freq.F15: "15min", Freq.F30: '30min',
            Freq.F60: "60min", Freq.D: 'D', Freq.W: "W", Freq.M: "M"} # 定义freq与字符串映射关系
freq_cn_map = {"1分钟": Freq.F1, "5分钟": Freq.F5, "15分钟": Freq.F15, "30分钟": Freq.F30,
               "60分钟": Freq.F60, "日线": Freq.D} # 定义freq的中文与对象映射关系
dt_fmt = "%Y-%m-%d %H:%M:%S" # 日期时间格式字符串
date_fmt = "%Y%m%d" # 日期格式字符串


class TushareProApi: # 定义TushareProApi类
    __token = '' # 私有类属性,token
    __http_url = 'http://api.waditu.com' # 私有类属性,请求URL

    def __init__(self, token, timeout=30): # 初始化方法
        """
        Parameters
        ----------
        token: str，   API接口TOKEN,用于用户认证
          这段代码是一个类的初始化方法（`__init__()`），在创建类的实例时被调用。该方法有两个参数：`token`和`timeout`（默认值为30）。
        `token`参数是一个字符串类型的参数，用于API接口的认证。在初始化方法中，`token`被赋值给了类的私有属性`__token`。
        `timeout`参数也是一个整数类型的参数，用于设置超时时间。在初始化方法中，`timeout`被赋值给了类的私有属性`__timeout`。
        私有属性通常用双下划线开头（例如`__token`和`__timeout`），表示它们是类的内部使用的属性，不应该被类外部访问。
        """
        self.__token = token # 设置token属性
        self.__timeout = timeout # 设置timeout属性

    @retry(stop=stop_after_attempt(10), wait=wait_random(1, 5)) # 设置重试装饰器
    def query(self, api_name, fields='', **kwargs):
        if api_name in ['__getstate__', '__setstate__']: # 排除特殊方法
            return pd.DataFrame()

        req_params = { # 构造请求参数
            'api_name': api_name,
            'token': self.__token,
            'params': kwargs,
            'fields': fields
        }

        res = requests.post(self.__http_url, json=req_params, timeout=self.__timeout) # 发送POST请求
        if res:
            result = json.loads(res.text) # 把返回结果加载为json对象
            if result['code'] != 0:
                logger.warning(f"{req_params}: {result}") # 日志记录错误
                raise Exception(result['msg']) # 抛出异常

            data = result['data']
            columns = data['fields']
            items = data['items']
            return pd.DataFrame(items, columns=columns) # 返回DataFrame
        else:
            return pd.DataFrame()
    """
    这部分代码是一个类的方法，用于查询数据。
    首先，代码检查提供的`api_name`是否是特殊方法`__getstate__`或`__setstate__`，如果是，则返回一个空的`pd.DataFrame()`对象。
    接下来，代码构造请求参数`req_params`：
    - `api_name`是要调用的API名称
    - `token`是访问API所需的令牌
    - `params`是其他可选的请求参数
    - `fields`是要返回的字段
    然后，代码使用`requests.post()`方法发送POST请求到指定的url`self.__http_url`，并将请求参数以JSON格式传递。
    如果请求成功（`res`有值），代码将返回结果解析为JSON对象，并检查`result['code']`是否为0。如果不是0，则使用日志记录错误，并抛出异常，异常消息为`result['msg']`。
    如果请求成功且`result['code']`为0，代码将提取返回结果中的数据，并将其转换为`pd.DataFrame`对象。返回的DataFrame将由`data['items']`中的行和`data['fields']`中的列组成。
    如果请求失败（`res`为空），代码将返回一个空的`pd.DataFrame()`对象。
    """


















    def __getattr__(self, name):
        return partial(self.query, name) # 通过偏函数调用query方法



try:
    from tushare.util import upass
    pro = TushareProApi(upass.get_token(), timeout=60) # 创建pro对象
except:
    print("Tushare Pro 初始化失败")


def format_kline(kline: pd.DataFrame, freq: Freq) -> List[RawBar]:
    """Tushare K线数据转换

    :param kline: Tushare 数据接口返回的K线数据
    :param freq: K线周期
    :return: 转换好的K线数据
    这段代码是用来将Tushare返回的K线数据转换为自定义的K线数据格式。该函数的参数包括一个Tushare返回的K线数据（kline）和K线周期（freq），返回值是转换后的K线数据列表（bars）。
    首先，函数定义了一个空的列表bars，用来存储转换后的K线数据。
    然后，根据K线周期的不同，确定使用哪个字段作为时间键。如果K线周期中包含"分钟"，则使用"trade_time"作为时间键；否则，使用"trade_date"作为时间键。
    接下来，对传入的kline数据按照时间键进行降序排序，并将排序后的数据转换为字典格式。
    然后，使用for循环遍历排序后的数据，并逐条进行转换。根据不同的K线周期，需要将成交量和成交额做一些转换。如果K线周期是日线（Freq.D），需要将成交量乘以100，将成交额乘以1000；否则，不进行转换。
    在每一次循环中，将每条K线数据转换成自定义的RawBar对象，并将其添加到bars列表中。
     最后，返回转换后的K线数据列表bars。
    """
    bars = []
    dt_key = 'trade_time' if '分钟' in freq.value else 'trade_date'  # 不同周期取不同的时间键
    kline = kline.sort_values(dt_key, ascending=True, ignore_index=True)  # 按时间排序
    records = kline.to_dict('records')

    for i, record in enumerate(records):
        if freq == Freq.D:
            vol = int(record['vol'] * 100)  # 日线成交量转换成股
            amount = int(record.get('amount', 0) * 1000)  # 日线成交额转换成元
        else:
            vol = int(record['vol'])
            amount = int(record.get('amount', 0))

        # 将每一根K线转换成 RawBar 对象
        bar = RawBar(symbol=record['ts_code'], dt=pd.to_datetime(record[dt_key]),
                     id=i, freq=freq, open=record['open'], close=record['close'],
                     high=record['high'], low=record['low'],
                     vol=vol,  # 成交量,单位:股
                     amount=amount,  # 成交额,单位:元
                     )
        bars.append(bar)
    return bars


@deprecated(reason="统一到 TsDataCache 对象中", version='0.9.0')
def get_kline(ts_code: str,
              start_date: [datetime, str],
              end_date: [datetime, str],
              asset: str = 'E',
              freq: Freq = Freq.F1,
              fq: str = "qfq") -> List[RawBar]:
    """
    通用行情接口: https://tushare.pro/document/2?doc_id=109

    :param ts_code:
    :param asset:
    :param freq:
    :param start_date:
    :param end_date:
    :param fq:
    :return:
    这段代码是一个获取K线数据的函数。

    首先，函数的参数说明如下：
    - ts_code: 股票代码
    - start_date: K线数据的起始日期，可以是datetime对象或者字符串形式的日期
    - end_date: K线数据的结束日期，同样可以是datetime对象或字符串形式的日期
    - asset: 数据源，默认为'E'，表示股票
    - freq: 数据类型，默认为Freq.F1，表示日线数据
    - fq: 复权类型，默认为"qfq"，表示前复权

    接下来，代码将start_date和end_date转换为datetime对象。
    如果freq中包含"分钟"，则将start_date和end_date转换为字符串形式的时间，否则转换为字符串形式的日期。
    然后，代码调用ts.pro_bar函数获取K线数据，传入的参数包括ts_code、fq、asset、freq等。
    接着，代码调用format_kline函数对获取到的K线数据进行格式化。
    最后，代码进行一些判断和打印输出。如果获取到的K线数据非空且最后一根K线的日期小于end_date，并且获取到的K线数量等于8000，打印一条消息。
    最后返回获取到的K线数据。
    """
    start_date = pd.to_datetime(start_date)  # 转换为datetime
    end_date = pd.to_datetime(end_date)

    if "分钟" in freq.value:
        start_date = start_date.strftime(dt_fmt)  # 分钟线传入时间格式
        end_date = end_date.strftime(dt_fmt)
    else:
        start_date = start_date.strftime(date_fmt)  # 日线传入日期格式
        end_date = end_date.strftime(date_fmt)

    df = ts.pro_bar(ts_code=ts_code, adj=fq, asset=asset, freq=freq_map[freq],  # 调用接口
                    start_date=start_date, end_date=end_date)
    bars = format_kline(df, freq)  # 格式化
    if bars and bars[-1].dt < pd.to_datetime(end_date) and len(bars) == 8000:
        print(f"获取K线数量达到8000根,数据获取到 {bars[-1].dt},目标 end_date 为 {end_date}")
    return bars