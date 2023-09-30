# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2021/10/24 16:12
describe:
Tushare 数据缓存，这是用pickle缓存数据，是临时性的缓存。
单次缓存，多次使用，但是不做增量更新，适用于研究场景。
数据缓存是一种用空间换时间的方法，需要有较大磁盘空间，跑全市场至少需要50GB以上。

根据代码和注释的内容,这个Python代码实现了以下主要功能:

数据缓存
使用pickle序列化缓存Tushare获取的数据,可以重复使用,避免重复请求数据
定义TsDataCache类对缓存进行封装,包含缓存的基本配置和管理
提供清空缓存的方法clear()
Tushare接口调用
封装了Tushare Pro的多个行情数据获取接口,如日线、分钟线等
使用了类似装饰器的机制,通过TsDataCache的接口调用底层的Tushare接口
支持从缓存读取数据或者强制刷新缓存
数据处理
格式化处理原始K线数据,转换成标准格式
计算K线的各类指标,如前N日涨跌幅等
为不同的接口和频率统一字段名称,方便使用
组合接口
提供了几个组合接口,可以批量获取某段时间的行情数据
如获取全市场股票的日线信息等
融合利用
与czsc等模块配合,将缓存的数据进行技术分析等运算
提供了基础的金融数据支撑
其他
支持日志记录,控制缓存刷新等配置
使用了一些Python的高级功能如装饰器、类型注解等
代码结构清晰,易于维护和二次开发
综上,这个Python代码通过数据缓存和封装,实现了股票数据的高效获取和处理,为量化分析奠定了基础。
"""
import time # 导入time模块,用于时间相关操作
import os.path # 导入os.path模块,用于路径操作
import shutil # 导入shutil模块,用于目录和文件操作
import pandas as pd # 导入pandas模块,用于数据分析
from tqdm import tqdm # 导入tqdm模块,用于显示进度条
from typing import List # 从typing模块导入List类型,用于类型注解
from deprecated import deprecated # 从deprecated模块导入deprecated装饰器,用于标记过时函数
from datetime import timedelta, datetime # 从datetime模块中导入时间处理类
from czsc import envs # 从czsc模块中导入envs配置
from czsc.enum import Freq # 从czsc模块中导入K线频率枚举
from czsc.utils import io # 从czsc模块中导入io工具模块
from czsc.data.ts import ts, pro, format_kline, dt_fmt, date_fmt # 导入tushare相关接口


def update_bars_return(kline: pd.DataFrame, bar_numbers=None):
    """
    给K线加上未来收益和过去收益的计算

    :param kline: K线数据,pd.DataFrame类型
    :param bar_numbers: 需要计算的向前和向后的Bar数列表,默认为[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    :return: 返回增加了未来收益和过去收益列的K线数据
    """

    # 统一处理数据精度,将open/close/high/low列的数据类型转换为4位小数的float
    for col in ['open', 'close', 'high', 'low']:
        kline[col] = kline[col].round(4)

    # 断言kline的时间索引是升序的,否则会报错
    assert kline['dt'][0] < kline['dt'][1], "kline 必须是时间升序"

    # 如果bar_numbers参数为空,则默认设置为[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    if not bar_numbers:
        bar_numbers = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377)

    # 遍历bar_numbers中的每个bar数
    for bar_number in bar_numbers:
        # 向后看bar_number个bar,今天的收益等于bar_number+1天后的开盘价相对于bar_number天后的开盘价的变化
        # 收益以基点(BP)表示
        n_col = 'n' + str(bar_number) + 'b'
        kline[n_col] = (kline['open'].shift(-bar_number - 1) / kline['open'].shift(-1) - 1) * 10000
        kline[n_col] = kline[n_col].round(4)

        # 向前看bar_number个bar,今天的收益等于今天收盘价相对于bar_number天前的收盘价的变化
        # 收益以基点(BP)表示
        b_col = 'b' + str(bar_number) + 'b'
        kline[b_col] = (kline['close'] / kline['close'].shift(bar_number) - 1) * 10000
        kline[b_col] = kline[b_col].round(4)

    return kline

class TsDataCache:
    """Tushare 数据缓存"""

    def __init__(self, data_path, refresh=False, sdt="20120101", edt=datetime.now()):
        """
        初始化函数

        :param data_path: 保存数据的根目录
        :param refresh: 是否刷新缓存
        :param sdt: 缓存开始时间
        :param edt: 缓存结束时间
        """

        # 定义日期格式
        self.date_fmt = "%Y%m%d"

        # 获取日志级别
        self.verbose = envs.get_verbose()

        # 刷新标志
        self.refresh = refresh

        # 缓存开始时间
        self.sdt = pd.to_datetime(sdt).strftime(self.date_fmt)

        # 缓存结束时间
        self.edt = pd.to_datetime(edt).strftime(self.date_fmt)

        # 数据根目录
        self.data_path = data_path

        # 缓存子目录名称
        self.prefix = "TS_CACHE"

        # 缓存完整路径
        self.cache_path = os.path.join(self.data_path, self.prefix)

        # 创建缓存目录
        os.makedirs(self.cache_path, exist_ok=True)

        # 初始化tushare接口
        self.pro = pro

        # 准备缓存路径
        self.__prepare_api_path()

        # 初始化频率映射字典
        self.freq_map = {
            "1min": Freq.F1,
            "5min": Freq.F5,
            "15min": Freq.F15,
            "30min": Freq.F30,
            "60min": Freq.F60,
            "D": Freq.D,
            "W": Freq.W,
            "M": Freq.M,
        }

    def __prepare_api_path(self):
        """
        为每个tushare接口生成缓存路径
        """

        # 获取缓存根目录
        cache_path = self.cache_path

        # 定义接口名称列表
        self.api_names = [
            'ths_daily', 'ths_index', 'ths_member', 'pro_bar',
            'hk_hold', 'cctv_news', 'daily_basic', 'index_weight',
            'pro_bar_minutes', 'limit_list', 'bak_basic',

            # CZSC加工缓存
            "stocks_daily_bars", "stocks_daily_basic", "stocks_daily_bak",
            "daily_basic_new", "stocks_daily_basic_new"
        ]

        # 生成接口名称与缓存路径的字典映射
        self.api_path_map = {k: os.path.join(cache_path, k) for k in self.api_names}

        # 遍历接口名称与路径字典,创建路径
        for k, path in self.api_path_map.items():
            os.makedirs(path, exist_ok=True)

    def clear(self):
        """
        清空缓存
        """
        # 遍历数据根目录
        for path in os.listdir(self.data_path):

            # 保留文件名以缓存前缀开头的目录
            if path.startswith(self.prefix):

                # 拼接完整目录路径
                path = os.path.join(self.data_path, path)

                # 删除目录
                shutil.rmtree(path)

                # 打印日志
                if self.verbose:
                    print(f"clear: remove {path}")

                # 检查是否删除成功
                if os.path.exists(path):
                    print(f"Tushare 数据缓存清理失败,请手动删除缓存文件夹:{self.cache_path}")

    # ------------------------------------Tushare 原生接口----------------------------------------------
    def ths_daily(self, ts_code, start_date=None, end_date=None, raw_bar=True):
        """
        获取同花顺概念板块的日线行情

        :param ts_code: 证券代码
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param raw_bar: 是否返回原始K线结构
        :return: 行情数据
        """
        # 获取缓存路径
        cache_path = self.api_path_map['ths_daily']

        # 生成缓存文件路径
        file_cache = os.path.join(cache_path, f"ths_daily_{ts_code}_sdt{self.sdt}.feather")

        # 优先读取缓存
        if not self.refresh and os.path.exists(file_cache):
            kline = pd.read_feather(file_cache)
            if self.verbose:
                print(f"ths_daily: read cache {file_cache}")

        else:
            # 调用接口拉取数据
            if self.verbose:
                print(f"ths_daily: refresh {file_cache}")
            kline = pro.ths_daily(ts_code=ts_code, start_date=self.sdt, end_date=self.edt,
                                  fields='ts_code,trade_date,open,close,high,low,vol')

            # 排序处理
            kline = kline.sort_values('trade_date', ignore_index=True)
            kline['trade_date'] = pd.to_datetime(kline['trade_date'], format=self.date_fmt)
            kline['dt'] = kline['trade_date']

            # 计算收益率
            update_bars_return(kline)

            # 缓存到本地
            kline.to_feather(file_cache)

        # 转换为日期类型
        kline['trade_date'] = pd.to_datetime(kline['trade_date'], format=self.date_fmt)

        # 过滤时间范围
        if start_date:
            kline = kline[kline['trade_date'] >= pd.to_datetime(start_date)]
        if end_date:
            kline = kline[kline['trade_date'] <= pd.to_datetime(end_date)]

        # 重置索引
        kline.reset_index(drop=True, inplace=True)

        # 格式化
        if raw_bar:
            kline = format_kline(kline, freq=Freq.D)

        return kline

    def ths_index(self, exchange="A", type_="N"):
        """
        获取同花顺概念板块

        https://tushare.pro/document/2?doc_id=259

        :param exchange: 板块所在交易所,A表示主板,B表示中小板,C表示创业板
        :param type_: 板块类型,N表示行业板块,C表示概念板块
        :return: 板块数据
        """

        # 获取缓存路径
        cache_path = self.api_path_map['ths_index']

        # 生成缓存文件路径
        file_cache = os.path.join(cache_path, f"ths_index_{exchange}_{type_}.feather")

        # 检查缓存
        if not self.refresh and os.path.exists(file_cache):

            # 读取缓存
            df = pd.read_feather(file_cache)

            if self.verbose:
                print(f"ths_index: read cache {file_cache}")

        else:

            # 调用接口拉取数据
            df = pro.ths_index(exchange=exchange, type=type_)

            # 缓存到本地
            df.to_feather(file_cache)

        return df

    def ths_member(self, ts_code):
        """
        获取同花顺概念板块成分股

        :param ts_code: 板块代码
        :return: 成分股信息
        """

        # 获取缓存路径
        cache_path = self.api_path_map['ths_member']

        # 生成缓存文件路径
        file_cache = os.path.join(cache_path, f"ths_member_{ts_code}.feather")

        # 检查缓存
        if not self.refresh and os.path.exists(file_cache):

            # 读取缓存
            df = pd.read_feather(file_cache)

            if self.verbose:
                print(f"ths_index: read cache {file_cache}")

        else:

            # 调用接口拉取数据
            df = pro.ths_member(ts_code=ts_code,
                                fields="ts_code,code,name,weight,in_date,out_date,is_new")

            # 重置索引
            df = df.reset_index(drop=True, inplace=False)

            # 缓存到本地
            df.to_feather(file_cache)

        return df

    def pro_bar(self, ts_code, start_date=None, end_date=None, freq='D', asset="E", adj='qfq', raw_bar=True):
        """
        获取日线以上K线数据

        :param ts_code: 证券代码
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param freq: 数据频率
        :param asset: 资产类别
        :param adj: 复权类型
        :param raw_bar: 是否格式化为K线结构
        :return: K线数据
        """

        # 获取缓存路径
        cache_path = self.api_path_map['pro_bar']

        # 生成缓存文件路径
        file_cache = os.path.join(cache_path, f"pro_bar_{ts_code}#{asset}_{self.sdt}_{freq}_{adj}.feather")

        # 优先读取缓存
        if not self.refresh and os.path.exists(file_cache):
            kline = pd.read_feather(file_cache)
            if self.verbose:
                print(f"pro_bar: read cache {file_cache}")

        else:
            # 调用接口拉取数据
            start_date_ = (pd.to_datetime(self.sdt) - timedelta(days=1000)).strftime('%Y%m%d')
            kline = ts.pro_bar(ts_code=ts_code, asset=asset, adj=adj, freq=freq,
                               start_date=start_date_, end_date=self.edt)

            # 排序处理
            kline = kline.sort_values('trade_date', ignore_index=True)
            kline['trade_date'] = pd.to_datetime(kline['trade_date'], format=self.date_fmt)
            kline['dt'] = kline['trade_date']

            # 增加均价
            kline['avg_price'] = kline['amount'] / kline['vol']

            # 计算收益率
            update_bars_return(kline)

            # 缓存
            kline.to_feather(file_cache)

        # 过滤时间范围
        if start_date:
            kline = kline[kline['trade_date'] >= pd.to_datetime(start_date)]
        if end_date:
            kline = kline[kline['trade_date'] <= pd.to_datetime(end_date)]

        # 重置索引
        kline.reset_index(drop=True, inplace=True)

        # 格式化
        if raw_bar:
            kline = format_kline(kline, freq=self.freq_map[freq])

        return kline

    def pro_bar_minutes(self, ts_code, sdt=None, edt=None, freq='60min', asset="E", adj=None, raw_bar=True):
        """
        获取分钟线数据

        :param ts_code: 证券代码
        :param sdt: 开始时间
        :param edt: 结束时间
        :param freq: 分钟频率
        :param asset: 资产类别
        :param adj: 复权类型
        :param raw_bar: 是否格式化为K线
        :return: K线数据
        """

        # 获取缓存路径
        cache_path = self.api_path_map['pro_bar_minutes']

        # 生成缓存文件路径
        file_cache = os.path.join(cache_path, f"pro_bar_minutes_{ts_code}#{asset}_{self.sdt}_{freq}_{adj}.feather")

        # 读取缓存
        if not self.refresh and os.path.exists(file_cache):
            kline = pd.read_feather(file_cache)
            if self.verbose:
                print(f"pro_bar_minutes: read cache {file_cache}")

        else:
            # 分批拉取数据
            klines = []
            end_dt = pd.to_datetime(self.edt)
            dt1 = pd.to_datetime(self.sdt)
            delta = timedelta(days=20 * int(freq.replace("min", "")))
            dt2 = dt1 + delta
            while dt1 < end_dt:
                df = ts.pro_bar(ts_code=ts_code, asset=asset, freq=freq,
                                start_date=dt1.strftime(dt_fmt), end_date=dt2.strftime(dt_fmt))
                klines.append(df)
                dt1 = dt2
                dt2 = dt1 + delta
                if self.verbose:
                    print(f"pro_bar_minutes: {ts_code} - {asset} - {freq} - {dt1} - {dt2} - {len(df)}")

            # 合并分批数据
            df_klines = pd.concat(klines, ignore_index=True)

            # 去重排序
            kline = df_klines.drop_duplicates('trade_time') \
                .sort_values('trade_time', ascending=True, ignore_index=True)

            # 处理时间
            kline['trade_time'] = pd.to_datetime(kline['trade_time'], format=dt_fmt)
            kline['dt'] = kline['trade_time']

            # 处理数值列
            float_cols = ['open', 'close', 'high', 'low', 'vol', 'amount']
            kline[float_cols] = kline[float_cols].astype('float32')
            kline['avg_price'] = kline['amount'] / kline['vol']

            # 删除无效K线
            kline['keep'] = kline['trade_time'].apply(lambda x: 0 if x.hour == 9 and x.minute == 30 else 1)
            kline = kline[kline['keep'] == 1]
            kline = kline[kline['vol'] > 0]
            kline.drop(['keep'], axis=1, inplace=True)

            # 过滤时间范围
            start_date = pd.to_datetime(self.sdt)
            end_date = pd.to_datetime(self.edt)
            kline = kline[(kline['trade_time'] >= start_date) &
                          (kline['trade_time'] <= end_date)]

            # 处理日期
            kline = kline.reset_index(drop=True)
            kline['trade_date'] = kline.trade_time.apply(lambda x: x.strftime(date_fmt))

            # 处理复权
            if asset == 'E':
                factor = pro.adj_factor(ts_code=ts_code, start_date=self.sdt, end_date=self.edt)
            elif asset == 'FD':
                factor = pro.fund_adj(ts_code=ts_code, start_date=self.sdt, end_date=self.edt)
            else:
                factor = pd.DataFrame()

            # 复权处理
            if len(factor) > 0 and adj and adj == 'qfq':
                latest_factor = factor.iloc[-1]['adj_factor']
                adj_map = {row['trade_date']: row['adj_factor'] for _, row in factor.iterrows()}
                for col in ['open', 'close', 'high', 'low']:
                    kline[col] = kline.apply(lambda x: x[col] * adj_map[x['trade_date']] / latest_factor, axis=1)

            if len(factor) > 0 and adj and adj == 'hfq':
                adj_map = {row['trade_date']: row['adj_factor'] for _, row in factor.iterrows()}
                for col in ['open', 'close', 'high', 'low']:
                    kline[col] = kline.apply(lambda x: x[col] * adj_map[x['trade_date']], axis=1)

            # 计算收益率
            update_bars_return(kline)

            # 缓存
            kline.to_feather(file_cache)

        # 过滤时间范围
        if sdt:
            kline = kline[kline['trade_time'] >= pd.to_datetime(sdt)]
        if edt:
            kline = kline[kline['trade_time'] <= pd.to_datetime(edt)]

        # 格式化
        bars = kline.reset_index(drop=True)
        if raw_bar:
            bars = format_kline(bars, freq=self.freq_map[freq])

        return bars













    def stock_basic(self):
        """
        获取股票基础信息数据

        返回数据包含股票代码、名称、上市日期、退市日期等基本信息

        :return: 股票基础信息表
        """

        # 缓存文件路径
        file_cache = os.path.join(self.cache_path, f"stock_basic.feather")

        # 优先读取缓存
        if not self.refresh and os.path.exists(file_cache):
            df = pd.read_feather(file_cache)

        else:
            # 调用接口获取数据
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

            # 更新缓存
            df.to_feather(file_cache)

        return df

    def trade_cal(self):
        """
        获取各大交易所交易日历数据,默认提取的是上交所

        接口:trade_cal
        可通过数据工具调试和查看数据

        描述:获取各大交易所交易日历数据,默认提取的是上交所
        积分:需2000积分

        输入参数:

        名称          类型   必选  描述
        exchange      str    N     交易所 SSE上交所,SZSE深交所,CFFEX 中金所,SHFE 上期所,CZCE 郑商所,DCE 大商所,INE 上能源
        start_date    str    N     开始日期 (格式:YYYYMMDD 下同)
        end_date      str    N     结束日期
        is_open       str    N     是否交易 '0'休市 '1'交易

        输出参数:

        名称          类型    默认显示  描述
        exchange      str     Y        交易所 SSE上交所 SZSE深交所
        cal_date      str     Y        日历日期
        is_open       str     Y        是否交易 0休市 1交易
        pretrade_date str     Y        上一个交易日

        """

        # 从缓存中加载数据
        file_cache = os.path.join(self.cache_path, "trade_cal.feather")
        if os.path.exists(file_cache):
            df = pd.read_feather(file_cache)

        else:
            # 调用pro接口获取数据
            df = pro.trade_cal(exchange='', start_date='19900101', end_date='20300101')

            # 按日期排序
            df = df.sort_values('cal_date', ascending=True).reset_index(drop=True)

            # 缓存到本地
            df.to_feather(file_cache)

        return df

    def hk_hold(self, trade_date='20190625'):
        """
        沪深港股通持股明细

        https://tushare.pro/document/2?doc_id=188

        输入参数:
        trade_date: 交易日期,默认20190625

        输出参数:
        主要字段:
        trade_date     交易日期
        share_hk       沪股通持股数量(股)
        market_value_hk    沪股通持股市值(元)
        share_cn       深股通持股数量(股)
        market_value_cn   深股通持股市值(元)
        ......

        """

        # 定义缓存路径
        cache_path = self.api_path_map['hk_hold']

        # 将交易日期转换为字符串格式
        trade_date = pd.to_datetime(trade_date).strftime("%Y%m%d")

        # 拼接缓存文件名
        file_cache = os.path.join(cache_path, f"hk_hold_{trade_date}.feather")

        # 优先从缓存读数据
        if os.path.exists(file_cache):
            df = pd.read_feather(file_cache)

        else:
            # 调用接口获取数据
            df = pro.hk_hold(trade_date=trade_date)

            # 数据为空则直接返回
            if df.empty:
                return df

            # 缓存数据
            df.to_feather(file_cache)

        return df

    def cctv_news(self, date='20190625'):
        """
        新闻联播

        https://tushare.pro/document/2?doc_id=154

        输入参数:
        date: 日期,默认是20190625

        输出参数:
        标题
        来源
        时间
        内容

        主要步骤:
        1. 定义缓存路径
        2. 处理日期格式
        3. 拼接缓存文件名
        4. 优先从缓存读取
        5. 如果缓存不存在,调用接口获取数据
        6. 数据为空直接返回
        7. 否则缓存数据
        8. 返回结果

        """

        # 1. 定义缓存路径
        cache_path = self.api_path_map['cctv_news']

        # 2. 处理日期格式
        date = pd.to_datetime(date).strftime("%Y%m%d")

        # 3. 拼接缓存文件名
        file_cache = os.path.join(cache_path, f"cctv_news_{date}.feather")

        # 4. 优先从缓存读取
        if os.path.exists(file_cache):
            df = pd.read_feather(file_cache)

        else:
            # 5. 调用接口获取数据
            df = pro.cctv_news(date=date)

            # 6. 数据为空直接返回
            if df.empty:
                return df

            # 7. 否则缓存数据
            df.to_feather(file_cache)

        # 8. 返回结果
        return df

    def index_weight(self, index_code: str, trade_date: str):
        """
        指数成分和权重

        https://tushare.pro/document/2?doc_id=96

        输入参数:
        index_code:指数代码,如000300.SH
        trade_date:交易日期,格式YYYYMMDD

        输出参数:
        con_code 成分代码
        con_name 成分名称
        weight   权重

        主要步骤:
        1. 转换交易日期格式
        2. 拼接缓存文件路径及文件名
        3. 优先从缓存文件读取数据
        4. 若缓存不存在,调用接口获取数据
            - 指定开始、结束日期
            - 去重keeping='first'
        5. 数据为空,直接返回
        6. 否则,保存到缓存
        7. 返回结果

        """

        # 1. 转换交易日期格式
        trade_date = pd.to_datetime(trade_date)

        # 2. 拼接缓存文件路径及文件名
        cache_path = self.api_path_map['index_weight']
        file_cache = os.path.join(cache_path, f"index_weight_{index_code}_{trade_date.strftime('%Y%m')}.feather")

        # 3. 优先从缓存文件读取数据
        if os.path.exists(file_cache):
            df = pd.read_feather(file_cache)

        else:
            # 4. 调用接口获取数据
            start_date = (trade_date.replace(day=1) - timedelta(days=31)).strftime('%Y%m%d')
            end_date = (trade_date.replace(day=1) + timedelta(days=31)).strftime('%Y%m%d')
            df = pro.index_weight(index_code=index_code, start_date=start_date, end_date=end_date)

            # 去重
            df = df.drop_duplicates('con_code', ignore_index=True)

            # 5. 数据为空,直接返回
            if df.empty:
                return df

            # 6. 否则,保存到缓存
            df.to_feather(file_cache)

        # 7. 返回结果
        return df






    def limit_list(self, trade_date: str):
        """https://tushare.pro/document/2?doc_id=198

        :param trade_date: 交易日期
        :return: 每日涨跌停统计
        """
        trade_date = pd.to_datetime(trade_date).strftime("%Y%m%d")
        cache_path = self.api_path_map['limit_list']
        file_cache = os.path.join(cache_path, f"limit_list_{trade_date}.pkl")

        if os.path.exists(file_cache):
            df = pd.read_pickle(file_cache)
            if not df.empty:
                return df

        df = pro.limit_list(trade_date=trade_date)
        df.to_pickle(file_cache)
        return df

    @deprecated(reason='推荐使用 daily_basic_new 替代', version='0.9.0')
    def daily_basic(self, ts_code: str, start_date: str, end_date: str):
        """每日指标

        https://tushare.pro/document/2?doc_id=32
        """
        cache_path = self.api_path_map['daily_basic']
        file_cache = os.path.join(cache_path, f"daily_basic_{ts_code}.pkl")

        if os.path.exists(file_cache):
            df = io.read_pkl(file_cache)
        else:
            start_date_ = (pd.to_datetime(self.sdt) - timedelta(days=1000)).strftime('%Y%m%d')
            df = pro.daily_basic(ts_code=ts_code, start_date=start_date_, end_date="20300101")
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            io.save_pkl(df, file_cache)

        df = df[(df.trade_date >= pd.to_datetime(start_date)) & (df.trade_date <= pd.to_datetime(end_date))]
        return df

    # ------------------------------------以下是 CZSC 加工接口----------------------------------------------

    def daily_basic_new(self, trade_date: str):
        """股票每日指标接口整合

        每日指标：https://tushare.pro/document/2?doc_id=32
        备用列表：https://tushare.pro/document/2?doc_id=262

        :param trade_date: 交易日期
        :return:
        """
        trade_date = pd.to_datetime(trade_date).strftime("%Y%m%d")
        cache_path = self.api_path_map['daily_basic_new']
        file_cache = os.path.join(cache_path, f"bak_basic_new_{trade_date}.pkl")
        if os.path.exists(file_cache):
            df = pd.read_pickle(file_cache)
            if not df.empty:
                return df

        df1 = pro.bak_basic(trade_date=trade_date)
        df2 = pro.daily_basic(trade_date=trade_date)
        df1 = df1[['trade_date', 'ts_code', 'name', 'industry', 'area',
                   'total_assets', 'liquid_assets',
                   'fixed_assets', 'reserved', 'reserved_pershare', 'eps', 'bvps',
                   'list_date', 'undp', 'per_undp', 'rev_yoy', 'profit_yoy', 'gpr', 'npr',
                   'holder_num']]
        df = df2.merge(df1, on=['ts_code', 'trade_date'], how='left')
        df['is_st'] = df['name'].str.contains('ST')
        df.to_pickle(file_cache)
        return df

    def get_all_ths_members(self, exchange="A", type_="N"):
        """获取同花顺A股全部概念列表"""
        file_cache = os.path.join(self.cache_path, f"{exchange}_{type_}_ths_members.feather")
        if not self.refresh and os.path.exists(file_cache):
            df = pd.read_feather(file_cache)
        else:
            concepts = self.ths_index(exchange, type_)
            concepts = concepts.to_dict('records')

            res = []
            for concept in tqdm(concepts, desc='get_all_ths_members'):
                _df = self.ths_member(ts_code=concept['ts_code'])
                _df['概念名称'] = concept['name']
                _df['概念代码'] = concept['ts_code']
                _df['概念类别'] = concept['type']
                res.append(_df)
                time.sleep(0.3)

            df = pd.concat(res, ignore_index=True)
            df.to_feather(file_cache)
        return df

    def get_next_trade_dates(self, date, n: int = 1, m: int = None):
        """获取将来的交易日期

        如果 m = None，返回基准日期后第 n 个交易日；否则返回基准日期后第 n ~ m 个交易日

        :param date: 基准日期
        :param n:
        :param m:
        :return:
        """
        date = pd.to_datetime(date).strftime("%Y%m%d")
        trade_cal = self.trade_cal()
        trade_cal = trade_cal[trade_cal.is_open == 1]
        trade_dates = trade_cal.cal_date.to_list()
        assert date in trade_dates, "基准日期 date 必须是开市交易日期"

        i = trade_dates.index(date)
        if not m:
            ntd = trade_dates[i + n]
            return ntd
        else:
            assert m > n > 0 or m < n < 0, "abs(m) 必须大于 abs(n)"
            if m > n > 0:
                ntd_list = trade_dates[i+n: i+m]
            else:
                ntd_list = trade_dates[i+m: i+n]
            return ntd_list

    def get_dates_span(self, sdt: str, edt: str, is_open: bool = True) -> List[str]:
        """获取日期区间列表

        :param sdt: 开始日期
        :param edt: 结束日期
        :param is_open: 是否是交易日
        :return: 日期区间列表
        """
        sdt = pd.to_datetime(sdt).strftime("%Y%m%d")
        edt = pd.to_datetime(edt).strftime("%Y%m%d")

        trade_cal = self.trade_cal()
        if is_open:
            trade_cal = trade_cal[trade_cal['is_open'] == 1]

        trade_dates = [x for x in trade_cal['cal_date'] if edt >= x >= sdt]
        return trade_dates

    def stocks_daily_bars(self, sdt="20190101", edt="20220218", adj='hfq'):
        """读取A股全部历史日线

        :param sdt: 开始日期
        :param edt: 结束日期
        :param adj: 复权类型
        :return:
        """
        cache_path = self.api_path_map['stocks_daily_bars']
        file_cache = os.path.join(cache_path, f"stocks_daily_bars_{sdt}_{edt}_{adj}.feather")
        if os.path.exists(file_cache):
            print(f"stocks_daily_bars :: read cache from {file_cache}")
            df = pd.read_feather(file_cache)
            return df

        stocks = self.stock_basic()

        def __is_one_line(row_):
            """判断 row_ 是否是一字板"""
            if row_['open'] == row_['close'] == \
                    row_['high'] == row_['low'] and row_['b1b']:
                if row_['b1b'] > 500:
                    return 1
                if row_['b1b'] < -500:
                    return -1
            return 0

        results = []
        for row in tqdm(stocks.to_dict('records'), desc="stocks_daily_bars"):
            ts_code = row['ts_code']
            try:
                n_bars: pd.DataFrame = self.pro_bar(ts_code, sdt, edt, freq='D', asset='E', adj=adj, raw_bar=False)
                # 计算下期一字板
                n_bars['当期一字板'] = n_bars.apply(__is_one_line, axis=1)
                n_bars['下期一字板'] = n_bars['当期一字板'].shift(-1)
                results.append(n_bars)
            except:
                continue
        df = pd.concat(results, ignore_index=True)

        # 涨跌停判断
        def __is_zdt(row_):
            # 涨停描述：收盘价等于最高价，且当日收益 b1b 大于700BP
            if row_['close'] == row_['high'] and row_['b1b'] > 700:
                return 1
            elif row_['close'] == row_['low']:
                return -1
            else:
                return 0

        df['zdt'] = df.apply(__is_zdt, axis=1)
        float_cols = [k for k, v in df.dtypes.to_dict().items() if v.name.startswith('float')]
        df[float_cols] = df[float_cols].astype('float32')

        df.to_feather(file_cache)
        return df

    def stocks_daily_basic_new(self, sdt: str, edt: str):
        """读取A股 sdt ~ edt 时间区间的全部历史 daily_basic_new

        :param sdt: 开始日期
        :param edt: 结束日期
        :return:
        """
        cache_path = self.api_path_map['stocks_daily_basic_new']
        file_cache = os.path.join(cache_path, f"stocks_daily_basic_new_{sdt}_{edt}.feather")
        if os.path.exists(file_cache):
            print(f"stocks_daily_basic_new :: read cache from {file_cache}")
            df = pd.read_feather(file_cache)
            return df

        dates = self.get_dates_span(sdt, edt, is_open=True)
        results = [self.daily_basic_new(d) for d in tqdm(dates, desc='stocks_daily_basic_new')]
        dfb = pd.concat(results, ignore_index=True)
        dfb['trade_date'] = pd.to_datetime(dfb['trade_date'])
        dfb['上市天数'] = (dfb['trade_date'] - pd.to_datetime(dfb['list_date'], errors='coerce')).apply(lambda x: x.days)
        dfb.to_feather(file_cache)
        return dfb



