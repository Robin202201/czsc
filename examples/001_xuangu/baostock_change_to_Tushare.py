import os
import pandas as pd

# 源文件夹路径
源文件夹 = r'D:\001_DATA\Test\stk_data\d'

# 目标文件夹路径
目标文件夹 = r'D:\001_DATA\Test\stk_data\d_tushare'

# 确保目标文件夹存在
os.makedirs(目标文件夹, exist_ok=True)

# 对源文件夹中的每一个文件进行操作
for 文件名 in os.listdir(源文件夹):
    # 构造完整的文件路径
    源文件路径 = os.path.join(源文件夹, 文件名)

    # 将文件加载成pandas DataFrame
    df = pd.read_csv(源文件路径)

    # 重命名列
    df.columns = ['trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']

    # 从文件名中提取股票代码，将其转换为ts_code格式
    ts_code = 文件名.replace('sh.', '').replace('sz.', '').replace('.csv', '') + '.SZ'

    # 在DataFrame中添加一个新的ts_code列
    df['ts_code'] = ts_code

    # 添加新的列 'freq' 并将其值设置为 '日线'
    df['freqs'] = '日线'

    # 重新排列列
    df = df[['ts_code', 'trade_date', 'close', 'open', 'high', 'low', 'freqs']]

    # 构造目标文件路径，确保只有一个.csv在文件名中
    目标文件路径 = os.path.join(目标文件夹, f'{ts_code}.csv')

    # 将DataFrame保存到CSV文件
    df.to_csv(目标文件路径, index=False, encoding='utf-8-sig')

print('转换完成')