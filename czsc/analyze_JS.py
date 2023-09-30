# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2021/3/10 11:21
describe: 缠论分型、笔的识别
"""
# 这是文件的元信息，包括编码格式、作者、邮箱、创建日期和描述

import os  # 用于操作系统相关的操作，如文件路径处理
import webbrowser  # 用于打开web浏览器
import numpy as np  # 科学计算库
from loguru import logger  # 日志库
from typing import List, Callable  # python 类型注释库
from collections import OrderedDict  # 有序字典
from czsc.enum import Mark, Direction  # 导入czsc库中的枚举类
from czsc.objects import BI, FX, RawBar, NewBar  # 导入czsc库中的对象
from czsc.utils.echarts_plot import kline_pro  # 导入czsc库中的画图函数
from czsc import envs  # 导入czsc库中的环境变量

logger.disable('czsc.analyze')
#
"""禁用名为'czsc.analyze'的logger,logger是一个日志模块,用于输出日志信息,
# disable方法可以禁用指定名字的logger,这样该logger就不会输出日志了
"""




# 定义函数，去除K线的包含关系
def remove_include(k1: NewBar, k2: NewBar, k3: RawBar):
    """
    去除包含关系:
    输入三根k线,其中k1和k2为没有包含关系的K线,k3为原始K线
    """

    # 比较k1和k2的高低点,判断方向
    if k1.high < k2.high:
        direction = Direction.Up  # 如果k1高点低于k2,则方向为上涨
    elif k1.high > k2.high:
        direction = Direction.Down  # 如果k1高点高于k2,则方向为下跌
    else:
        # 如果k1和k2高低点相同,返回k3不做处理
        k4 = NewBar(symbol=k3.symbol, id=k3.id, freq=k3.freq, dt=k3.dt, open=k3.open,
                    close=k3.close, high=k3.high, low=k3.low, vol=k3.vol, amount=k3.amount, elements=[k3])
        return False, k4

    # 判断 k2 和 k3 之间是否存在包含关系,有则处理
    if (k2.high <= k3.high and k2.low >= k3.low) or (k2.high >= k3.high and k2.low <= k3.low):
        # 如果k2包含k3,合并处理
        if direction == Direction.Up:
            # 方向上涨取高低点最大值
            high = max(k2.high, k3.high)
            low = max(k2.low, k3.low)
            # 取dt最大值
            dt = k2.dt if k2.high > k3.high else k3.dt
        elif direction == Direction.Down:
            # 方向下跌取高低点最小值
            high = min(k2.high, k3.high)
            low = min(k2.low, k3.low)
            # 取dt最小值
            dt = k2.dt if k2.low < k3.low else k3.dt
        else:
            raise ValueError

        # 开收口按涨跌方向取高低点
        open_, close = (high, low) if k3.open > k3.close else (low, high)
        vol = k2.vol + k3.vol  # 成交量合并
        amount = k2.amount + k3.amount  # 成交额合并
        # 由于k2.elements在某些极端情况下长度会异常,这里限定长度为100
        elements = [x for x in k2.elements[:100] if x.dt != k3.dt] + [k3]
        k4 = NewBar(symbol=k3.symbol, id=k2.id, freq=k2.freq, dt=dt, open=open_,
                    close=close, high=high, low=low, vol=vol, amount=amount, elements=elements)
        return True, k4
    else:
        # k2和k3没有包含关系,直接返回k3
        k4 = NewBar(symbol=k3.symbol, id=k3.id, freq=k3.freq, dt=k3.dt, open=k3.open,
                    close=k3.close, high=k3.high, low=k3.low, vol=k3.vol, amount=k3.amount, elements=[k3])
        return False, k4







# 定义函数，查找分型
def check_fx(k1: NewBar, k2: NewBar, k3: NewBar):
    """
    查找分型
    输入3根K线,判断k2是否存在分型
    """
    fx = None
    # 如果k1低点低于k2,k2高点高于k3,则k2为顶分型
    if k1.high < k2.high > k3.high and k1.low < k2.low > k3.low:
        fx = FX(symbol=k1.symbol, dt=k2.dt, mark=Mark.G, high=k2.high,
                low=k2.low, fx=k2.high, elements=[k1, k2, k3])

    # 如果k1高点高于k2,k2低点低于k3,则k2为底分型
    if k1.low > k2.low < k3.low and k1.high > k2.high < k3.high:
        fx = FX(symbol=k1.symbol, dt=k2.dt, mark=Mark.D, high=k2.high,
                low=k2.low, fx=k2.low, elements=[k1, k2, k3])

    return fx
"""

注释关键:
1. check_fx函数用来查找分型,输入3根K线,判断中间的K线k2是否存在分型
2. 通过判断k1和k3的高低点关系来确定k2的分型类型
- 如果k1低点低于k2,k2高点高于k3,则k2为顶分型
- 如果k1高点高于k2,k2低点低于k3,则k2为底分型
3. 如果判断成立,则返回包含分型信息的FX对象,否则返回None
4. FX对象包含了分型的所有信息,如符号、时间、高低点、分型价位等
以上注释详细解释了函数的逻辑和语法,方便初学者理解。

"""







# 定义函数，查找在无包含关系K线中的所有分型
def check_fxs(bars: List[NewBar]) -> List[FX]:
    """
    输入一串无包含关系K线,查找其中所有分型

    bars: List[NewBar] - 输入的K线Bar对象列表
    -> 表示函数的返回值类型,在Python 3.5及以上版本中可以用来注释函数的返回值类型。
    那么在这个函数中:
    def check_fxs(bars: List[NewBar]) -> List[FX]:
    -> List[FX] 表示该函数返回一个FX对象的列表。
     也就是说,check_fxs函数接收一个NewBar对象的列表作为参数,返回一个FX对象的列表。
    """

    fxs = []
    for i in range(1, len(bars) - 1):

        # 调用check_fx()函数判断bars[i]是否为分型
        # 实参bars[i-1]、bars[i]、bars[i+1]分别为前一Bar、当前Bar、后一Bar
        fx = check_fx(bars[i - 1], bars[i], bars[i + 1])

        if isinstance(fx, FX):

            # 默认情况下,fxs本身是顶底交替的
            # 但是对于一些特殊情况下不是这样

            # 如果fxs列表中已经有2个或更多分型
            # 并且当前分型fx的类型与fxs最后一个分型的类型相同
            # 则记录错误日志
            if len(fxs) >= 2 and fx.mark == fxs[-1].mark:
                logger.error(f"check_fxs错误: {bars[i].dt},{fx.mark},{fxs[-1].mark}")

            else:
                # 将当前分型fx添加到fxs中
                fxs.append(fx)

    return fxs


'''

上面代码逐句为输入的K线Bar列表bars找出所有分型, 并返回分型FX对象列表。

关键点:
1.使用for循环遍历bars中的每个Bar
2.调用check_fx()
判断当前Bar是否为分型
3.将分型添加到fxs列表中, 要求保持顶底交替
4.如果出现异常的不交替情况, 记录错误日志
5.最终返回分型列表fxs
'''











# 定义函数，查找在无包含关系K线中的一笔
def check_bi(bars: List[NewBar], benchmark=None):
    """
    输入一串无包含关系K线,查找其中的一笔

    :param bars: 无包含关系K线列表
    :param benchmark: 当下笔能量的比较基准
    :return:
    """

    # min_bi_len: 最小笔长度
    min_bi_len = envs.get_min_bi_len()

    # 使用check_fxs函数识别顶底分型
    fxs = check_fxs(bars)

    # 如果顶底分型数量少于2个,无法组成笔,返回None
    if len(fxs) < 2:
        return None, bars

    # 取第一个分型作为笔开始的分型
    fx_a = fxs[0]

    try:
        # 如果第一个分型是底分型
        if fxs[0].mark == Mark.D:
            # 笔的方向定为上涨
            direction = Direction.Up
            # 在后面的分型中找到高点分型
            fxs_b = [x for x in fxs if x.mark == Mark.G and x.dt > fx_a.dt and x.fx > fx_a.fx]
            # 如果找不到高点分型,无法组成笔,返回None
            if not fxs_b:
                return None, bars

            # 取第一个高点分型
            fx_b = fxs_b[0]
            # 在候选高点分型中选择最高价格的那个作为结束分型
            for fx in fxs_b[1:]:
                if fx.high >= fx_b.high:
                    fx_b = fx

        # 如果第一个分型是高分型
        elif fxs[0].mark == Mark.G:
            # 笔的方向定为下跌
            direction = Direction.Down
            # 在后面的分型中找到低点分型
            fxs_b = [x for x in fxs if x.mark == Mark.D and x.dt > fx_a.dt and x.fx < fx_a.fx]
            # 如果找不到低点分型,无法组成笔,返回None
            if not fxs_b:
                return None, bars

            # 取第一个低点分型
            fx_b = fxs_b[0]
            # 在候选低点分型中选择最低价格的那个作为结束分型
            for fx in fxs_b[1:]:
                if fx.low <= fx_b.low:
                    fx_b = fx
        else:
            raise ValueError
    except Exception as e:
        logger.exception(f"笔识别错误: {e}")
        return None, bars

    # 根据起始分型和结束分型区间框选中间的K线作为笔的价格区间
    bars_a = [x for x in bars if fx_a.elements[0].dt <= x.dt <= fx_b.elements[2].dt]
    # 剩下的K线作为后续分析
    bars_b = [x for x in bars if x.dt >= fx_b.elements[0].dt]

    # 判断fx_a和fx_b价格区间是否存在包含关系
    ab_include = (fx_a.high > fx_b.high and fx_a.low < fx_b.low) or (fx_a.high < fx_b.high and fx_a.low > fx_b.low)

    # 判断当前笔的涨跌幅是否超过benchmark的一定比例
    if benchmark and abs(fx_a.fx - fx_b.fx) > benchmark * envs.get_bi_change_th():
        power_enough = True
    else:
        power_enough = False

    # 成笔的条件:
    # 1)顶底分型之间没有包含关系;
    # 2)笔长度大于等于min_bi_len 或 当前笔的涨跌幅已经够大
    if (not ab_include) and (len(bars_a) >= min_bi_len or power_enough):
        # 取笔区间内的所有分型
        fxs_ = [x for x in fxs if fx_a.elements[0].dt <= x.dt <= fx_b.elements[2].dt]
        # 构建BI对象
        bi = BI(symbol=fx_a.symbol, fx_a=fx_a, fx_b=fx_b, fxs=fxs_, direction=direction, bars=bars_a)
        return bi, bars_b
    else:
        return None, bars





# 定义类CZSC，处理K线数据
'''
以下代码CZSC类的说明，
这是一个名为"CZSC"的Python类，它用于进行股票市场技术分析。CZSC代表"缠中说禅"，是一种基于缠论的技术分析方法。以下是此类的主要功能：

初始化（__init__方法）：在类初始化过程中，将原始的K线数据、最大允许保留的笔数量和自定义的信号计算函数作为参数输入。此外，类还会初始化多个列表来储存原始K线数据、未完成笔的无包含K线序列、笔列表等信息。

更新分析结果（update方法）：此方法用于更新分析结果。它首先更新K线序列，然后更新笔，然后根据最大笔数量限制完成笔列表和原始K线序列的数量控制。最后，如果有信号计算函数，则进行信号计算。

绘制K线分析图（to_echarts和to_plotly方法）：这两个方法用于绘制K线分析图。to_echarts方法使用Echarts库进行绘制，而to_plotly方法使用Plotly库进行绘制。

打开浏览器查看分析结果（open_in_browser方法）：此方法将分析结果绘制成图表，并在浏览器中打开以查看。

获取属性值（last_bi_extend、finished_bis、ubi_fxs、ubi和fx_list方法）：这些方法用于获取类的一些属性值，例如最后一笔是否在延伸中、已完成的笔、未完成笔的分型、未完成的笔以及分型列表等。

注意：此类中的一些名称（如"笔"、"分型"、"K线"等）来自缠论，这是一种技术分析方法，旨在通过分析价格走势图表来预测股票和其他金融市场的未来走势。
'''
class CZSC:
    def __init__(self,
                 bars: List[RawBar],
                 get_signals=None,
                 max_bi_num=envs.get_max_bi_num(),
                 ):
        """
        初始化对象

        :param bars: K线数据,传入的原始K线序列
        :param max_bi_num: 最大允许保留的笔数量
        :param get_signals: 自定义的信号计算函数
        """

        # envs.get_verbose() 获取全局变量中的verbose的值,用于控制日志输出
        self.verbose = envs.get_verbose()

        # 将传入的最大笔数量保存在对象属性中
        self.max_bi_num = max_bi_num

        # 原始K线序列,空列表,用于保存传入的原始K线数据
        self.bars_raw: List[RawBar] = []

        # 未完成笔的无包含K线序列,空列表,用于保存识别过程中的未完成笔数据
        self.bars_ubi: List[NewBar] = []

        # 完成笔序列,空列表,用于保存识别出的完成笔
        self.bi_list: List[BI] = []

        # symbol和freq保存K线数据的品种和周期信息
        self.symbol = bars[0].symbol
        self.freq = bars[0].freq

        # get_signals保存传入的自定义信号计算函数
        self.get_signals = get_signals

        # signals为空,用于保存信号计算结果
        self.signals = None

        # cache 是信号计算过程的缓存容器,字典结构,需要信号计算函数自行维护
        self.cache = OrderedDict()

        # 循环遍历传入的K线数据,调用update方法逐个传入更新
        for bar in bars:
            self.update(bar)




    def __repr__(self):
        """
        __repr__魔术方法被调用时,返回对象的字符串表示
        这里返回类名以及symbol和freq信息组成的字符串
        这个方法实现了
        repr(__repr__)
        魔术方法, 当打印对象时, 会调用该方法, 返回一个包含类名、symbol和freq信息的字符串, 用于显示对象的简要信息。
        """
        return "<CZSC~{}~{}>".format(self.symbol, self.freq.value)





    # 更新笔的方法
    def __update_bi(self):
        bars_ubi = self.bars_ubi
        # bars_ubi表示未完成笔的K线Bar对象序列

        if len(bars_ubi) < 3:
            return

        # 查找笔
        if not self.bi_list:
            # self.bi_list为空表示没有查找到任何笔

            # 第一笔的查找
            fxs = check_fxs(bars_ubi)
            # 调用check_fxs函数在bars_ubi中查找第一个分型

            if not fxs:
                return

            fx_a = fxs[0]
            # 取第一个分型fx_a

            fxs_a = [x for x in fxs if x.mark == fx_a.mark]
            # 取所有与fx_a标记方向相同的分型

            for fx in fxs_a:
                if (fx_a.mark == Mark.D and fx.low <= fx_a.low) \
                        or (fx_a.mark == Mark.G and fx.high >= fx_a.high):
                    fx_a = fx
            # 找出StreamHandler: 该方向分型中力度最大的分型作为fx_a

            bars_ubi = [x for x in bars_ubi if x.dt >= fx_a.elements[0].dt]
            # 取bars_ubi中dt >= fx_a首个元素的K线

            bi, bars_ubi_ = check_bi(bars_ubi)
            # 使用bars_ubi调用check_bi寻找第一笔,返回笔bi和更新后的bars_ubi_

            if isinstance(bi, BI):
                self.bi_list.append(bi)
                # 如果返回的bi是一个笔对象,则添加到self.bi_list中

            self.bars_ubi = bars_ubi_
            # 更新self.bars_ubi

            return

        # 如果bars_ubi长度超过100根,打印日志信息
        if self.verbose and len(bars_ubi) > 100:
            logger.info(f"{self.symbol} - {self.freq} - {bars_ubi[-1].dt} 未完成笔延伸数量: {len(bars_ubi)}")

        # 根据环境参数判断是否使用benchmark
        if envs.get_bi_change_th() > 0.5 and len(self.bi_list) >= 5:
            price_seq = [x.power_price for x in self.bi_list[-5:]]
            benchmark = min(self.bi_list[-1].power_price, sum(price_seq) / len(price_seq))
        else:
            benchmark = None

        # 使用bars_ubi和benchmark调用check_bi查找新的笔
        bi, bars_ubi_ = check_bi(bars_ubi, benchmark)
        self.bars_ubi = bars_ubi_
        if isinstance(bi, BI):
            self.bi_list.append(bi)

        # 后处理:检查最后一个笔是否被新的K线破坏,如果被破坏,合并最后一个笔的bars和bars_ubi
        last_bi = self.bi_list[-1]
        bars_ubi = self.bars_ubi
        if (last_bi.direction == Direction.Up and bars_ubi[-1].high > last_bi.high) \
                or (last_bi.direction == Direction.Down and bars_ubi[-1].low < last_bi.low):
            # 当前笔被破坏,合并bars
            self.bars_ubi = last_bi.bars[:-2] + [x for x in bars_ubi if x.dt >= last_bi.bars[-2].dt]
            self.bi_list.pop(-1)



    # 更新分析结果的方法
    def update(self, bar: RawBar):
        """更新分析结果
        代码主要完成K线序列的更新、笔的识别、结果序列的限制长度等功能。
        1.    更新K线序列bars_raw, 处理时间延伸关系。
        2.    通过去除包含关系生成bars_ubi序列。
        3.    调用__update_bi()    更新笔识别。
        4.    限制保存的结果序列数量。
        5.    如果指定了信号计算函数get_signals, 则进行交易信号计算。
        :param bar: 单根K线对象
        """
        # 更新K线序列
        if not self.bars_raw or bar.dt != self.bars_raw[-1].dt:
            # 如果bars_raw为空,或者bar的时间与最后一根K线不同
            # 表示是新K线,直接追加
            self.bars_raw.append(bar)
            last_bars = [bar]
        else:
            # 当前bar的时间与最后一根K线相同,进行时间延伸
            self.bars_raw[-1] = bar
            # 从未完成笔序列bars_ubi中取出最后一根K线
            last_bars = self.bars_ubi.pop(-1).raw_bars
            assert bar.dt == last_bars[-1].dt
            # 用当前bar更新最后一根K线
            last_bars[-1] = bar

        # 去除包含关系
        bars_ubi = self.bars_ubi
        for bar in last_bars:
            if len(bars_ubi) < 2:
                # 少于2根无包含关系
                # 使用bar新建一个NewBar对象并添加到bars_ubi
                bars_ubi.append(NewBar(symbol=bar.symbol, id=bar.id, freq=bar.freq, dt=bar.dt,
                                       open=bar.open, close=bar.close, amount=bar.amount,
                                       high=bar.high, low=bar.low, vol=bar.vol, elements=[bar]))
            else:
                # 如果已经有2根以上K线,则检测包含关系
                k1, k2 = bars_ubi[-2:]
                # 调用remove_include移除包含关系
                has_include, k3 = remove_include(k1, k2, bar)
                if has_include:
                    bars_ubi[-1] = k3
                else:
                    bars_ubi.append(k3)
        self.bars_ubi = bars_ubi

        # 更新笔
        self.__update_bi()

        # 限制保存的结果序列数量
        self.bi_list = self.bi_list[-self.max_bi_num:]
        # 保留bi_list中最后max_bi_num个笔对象

        if self.bi_list:
            # 如果bi_list不为空

            sdt = self.bi_list[0].fx_a.elements[0].dt
            # 获取第一笔的起始分型时间

            s_index = 0
            for i, bar in enumerate(self.bars_raw):
                if bar.dt >= sdt:
                    s_index = i
                    break
            # 找到bars_raw中第一个时间大于等于sdt的K线索引

            self.bars_raw = self.bars_raw[s_index:]
            # 只保留从s_index开始后的K线序列

        # 计算交易信号
        if self.get_signals:
            self.signals = self.get_signals(c=self)
        # 如果指定了get_signals函数,则调用它计算交易信号

        else:
            self.signals = OrderedDict()
        # 否则使用空的有序字典

    def to_echarts(self, width: str = "1400px", height: str = '580px', bs=[]):
        """
        将分析结果绘制成K线图

        width: 图表宽度
        height: 图表高度
        bs: 交易标记,默认为空列表

        return: echarts K线图对象
        """

        # 提取K线数据
        kline = [x.__dict__ for x in self.bars_raw]

        # 提取笔数据
        if len(self.bi_list) > 0:
            bi = [{'dt': x.fx_a.dt, "bi": x.fx_a.fx} for x in self.bi_list]
            bi += [{'dt': self.bi_list[-1].fx_b.dt, "bi": self.bi_list[-1].fx_b.fx}]

        else:
            bi = []

        # 提取分型数据
        fx = [{'dt': x.dt, "fx": x.fx} for x in self.fx_list]

        # 调用kline_pro()生成echarts K线图
        chart = kline_pro(kline, bi=bi, fx=fx, width=width, height=height, bs=bs,
                          title="{}-{}".format(self.symbol, self.freq.value))

        return chart











    def to_plotly(self):
        """使用 plotly 绘制K线分析图"""

        import pandas as pd
        from czsc.utils.plotly_plot import KlineChart

        # 将self.bars_raw转换为DataFrame
        df = pd.DataFrame(self.bars_raw)

        # 创建KlineChart对象
        kline = KlineChart(n_rows=3, title="{}-{}".format(self.symbol, self.freq.value))

        # 添加K线图
        kline.add_kline(df, name="")

        # 添加均线
        kline.add_sma(df, ma_seq=(5, 10, 21), row=1, visible=True, line_width=1.2)
        kline.add_sma(df, ma_seq=(34, 55, 89, 144), row=1, visible=False, line_width=1.2)

        # 添加成交量
        kline.add_vol(df, row=2)

        # 添加MACD
        kline.add_macd(df, row=3)

        # 如果存在笔,提取笔和分型的数据
        if len(self.bi_list) > 0:
            bi1 = [{'dt': x.fx_a.dt, "bi": x.fx_a.fx, "text": x.fx_a.mark.value} for x in self.bi_list]
            bi2 = [{'dt': self.bi_list[-1].fx_b.dt, "bi": self.bi_list[-1].fx_b.fx,
                    "text": self.bi_list[-1].fx_b.mark.value[0]}]
            bi = pd.DataFrame(bi1 + bi2)
            fx = pd.DataFrame([{'dt': x.dt, "fx": x.fx} for x in self.fx_list])

            # 添加分型和笔标记
            kline.add_scatter_indicator(fx['dt'], fx['fx'], name="分型", row=1, line_width=2)
            kline.add_scatter_indicator(bi['dt'], bi['bi'], name="笔", text=bi['text'], row=1, line_width=2)

        # 返回plotly图表对象
        return kline.fig

    def open_in_browser(self, width: str = "1400px", height: str = '580px'):
        """
        直接在浏览器中打开K线分析结果图表

        width: 图表宽度
        height: 图表高度

        """

        home_path = os.path.expanduser("~")
        # 获取用户主文件夹路径

        file_html = os.path.join(home_path, "temp_czsc.html")
        # 拼接临时html文件路径

        chart = self.to_echarts(width, height)
        # 生成echarts图表对象

        chart.render(file_html)
        # 输出渲染后的html文件

        webbrowser.open(file_html)
        # 使用webbrowser模块打开html文件










    # 定义一些属性方法，获取分析结果在Python中.##
    # ## property装饰器用于将一个方法转换为一个属性。
    # @property的作用:将一个方法转换为一个属性,
    # 使得像访问属性一样调用方法。
    # 设置访问限制,对属性值进行验证

    @property
    def last_bi_extend(self):
        """
        判断最后一笔是否在延伸中

        return: True 表示最后一笔正在延伸,False 表示最后一笔已完成
        """

        # 获取最后一个笔对象
        last_bi = self.bi_list[-1]

        # 判断方向
        if last_bi.direction == Direction.Up:

            # 如果方向向上,判断未完成K线的最高价是否超过笔的高点
            if max([x.high for x in self.bars_ubi]) > last_bi.high:
                return True

        elif last_bi.direction == Direction.Down:

            # 如果方向向下,判断未完成K线的最低价是否低于笔的低点
            if min([x.low for x in self.bars_ubi]) < last_bi.low:
                return True

        # 默认返回False
        return False




    @property
    def finished_bis(self) -> List[BI]:
        """
        返回已完成的笔对象列表

        :return: 完成笔对象列表
        """

        if not self.bi_list:
            # 如果self.bi_list为空,表示还没有识别到任何笔
            # 返回空列表
            return []

        if len(self.bars_ubi) < 5:
            # bars_ubi为未完成K线序列
            # 如果未完成K线数量少于5根
            # 则认为除了最后一笔之外的笔应该都是完成的
            # 取bi_list中的除最后一笔之外的所有笔
            return self.bi_list[:-1]

        # 默认情况
        # 返回bi_list中所有识别到的笔
        # 这表示暂时认为所有笔都已完成
        return self.bi_list






    @property
    def ubi_fxs(self) -> List[FX]:
        """
        返回未完成笔K线(bars_ubi)中的分型列表

        :return: bars_ubi中的分型对象列表
        """

        if not self.bars_ubi:
            # 如果bars_ubi为空,直接返回空列表
            return []

        else:
            # 否则调用check_fxs函数识别bars_ubi中的分型
            # check_fxs将返回包含分型的列表
            return check_fxs(self.bars_ubi)








    @property
    def ubi(self):
        """
        生成当前未完成笔(ubi)的详情

        return: 未完成笔详情字典对象
        """

        ubi_fxs = self.ubi_fxs()
        # 调用self.ubi_fxs()获取未完成笔中的分型列表

        if not self.bars_ubi or not self.bi_list or not ubi_fxs:
            return None
            # 如果以下信息任意一个为空,返回None
            # 1. bars_ubi 未完成笔的K线
            # 2. bi_list 已完成的笔列表
            # 3. ubi_fxs 未完成笔中的分型

        bars_raw = [y for x in self.bars_ubi for y in x.raw_bars]
        # 将bars_ubi中的每一个NewBar对象中的raw_bars取出并拼接
        # 得到未完成笔内所有原始K线对象

        high_bar = max(bars_raw, key=lambda x: x.high)
        # 在bars_raw中找出高点价格对应的K线对象

        low_bar = min(bars_raw, key=lambda x: x.low)
        # 在bars_raw中找出低点价格对应的K线对象

        direction = Direction.Up if self.bi_list[-1].direction == Direction.Down else Direction.Down
        # 判断未完成笔的方向:如果上一笔向下,当前笔向上

        bi = {
            "symbol": self.symbol,
            "direction": direction,
            "high": high_bar.high,
            "low": low_bar.low,
            "high_bar": high_bar,
            "low_bar": low_bar,
            "bars": self.bars_ubi,
            "raw_bars": bars_raw,
            "fxs": ubi_fxs,
            "fx_a": ubi_fxs[0],
        }
        # 构建字典对象保存未完成笔相关数据

        return bi
        # 返回未完成笔详情字典







    @property
    def fx_list(self) -> List[FX]:
        """
        生成包含所有笔的分型列表

        return: 分型对象列表
        """

        fxs = []
        # 结果分型列表

        for bi_ in self.bi_list:
            fxs.extend(bi_.fxs[1:])
            # 从每一个笔bi中提取除首分型外的其他分型,添加到fxs

        ubi = self.ubi_fxs()
        # 获取未完成笔的分型列表

        for x in ubi:
            if not fxs or x.dt > fxs[-1].dt:
                fxs.append(x)
                # 添加新分型:如果fxs为空或新分型时间大于最后一个分型

        return fxs
        # 返回结果分型列表