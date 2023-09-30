# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2022/10/14 22:54
describe: 命令行工具集

https://click.palletsprojects.com/en/8.0.x/quickstart/
"""
import click
from loguru import logger
import  random


'''
在Python的click模块中,@click.group()是一个装饰器,用于将一个函数变为一个命令行接口组。
@click.group()的主要作用有:
将命令行工具打包为组,可以组织多个命令。
自动处理命令解析和调用。
统一的help信息。
'''
@click.group()
def czsc():
    """CZSC命令行工具"""

    # 1. 添加命令行参数
    # 例如添加--symbol参数指定股票代码

    # 2. 解析命令行参数
    # 使用argparse等模块解析参数

    # 3. 根据命令参数进行初始化
    # 根据传入的symbol创建Czsc对象进行初始化分析

    # 4. 调用Czsc对象方法进行计算
    # 例如调用add_bar进行更新
    # 调用to_echarts生成K线图

    # 5. 将结果输出到命令行或文件
    # 打印分析结果,绘制K线图等
    # 可调用click模块进行格式化输出

    # 6. 提供交互式操作
    # 如添加输入循环,提供交互式操作接口

    # 7. 处理异常,打印日志信息
    # 使用logging模块记录日志
    # 用try...except块处理异常

    # 8. 添加文档注释
    # 对模块、类、方法等添加文档注释
    # 用于生成命令行帮助信息

    pass






'''
@czsc.command()的作用是:
将一个函数注册为czsc的一个子命令
该命令会自动添加到czsc的命令行接口中
click会自动解析这个命令的参数和选项
调用命令时会执行被装饰的函数
'''


@czsc.command()
def aphorism():
    """随机输出一条缠中说禅良言警句"""

    # 从APHORISMS中随机选择一条
    aphorism = random.choice(APHORISMS)

    # 打印选中的警句
    print(aphorism)