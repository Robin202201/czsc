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
        token: str
            API接口TOKEN,用于用户认证
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