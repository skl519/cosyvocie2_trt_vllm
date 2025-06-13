import json
import sys
import akshare as ak
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import os  

# 设置中文字体
font_path = 'C:/Windows/Fonts/simsun.ttc'  # 这里指定一个支持中文的字体路径
my_font = fm.FontProperties(fname=font_path)
# 设置全局字体
plt.rcParams['font.family'] = my_font.get_name()  # 设置全局字体为指定的中文字体

# 处理数值，将"亿"和%转换为实际数值
def convert_to_number(x):
    if isinstance(x, str):
        x = x.replace(',', '')  # 移除千位分隔符
        if '%' in x:  # 处理百分比
            return float(x.replace('%', '')) / 100
        elif '万亿' in x:
            return float(x.replace('万亿', '')) * 1000000000000  # 处理万亿
        elif '亿' in x:
            return float(x.replace('亿', '')) * 100000000
        elif '万' in x:
            return float(x.replace('万', '')) * 10000
        else:
            return float(x)
    elif x is False or x is None:
        return np.nan
    return x

# 格式化数值为带单位的字符串
def format_with_unit(value):
    if value is None or np.isnan(value):
        return "N/A"
    elif value >= 1e12:
        return f"{value / 1e12:.2f}万亿"
    elif value >= 1e8:
        return f"{value / 1e8:.2f}亿"
    elif value >= 1e4:
        return f"{value / 1e4:.2f}万"
    else:
        return str(value)

def sanitize_filename(filename):
    """清理文件名，去掉不允许的字符"""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '_')).rstrip()

# 估值面的可视化
def polt_data(symbol, save_path):
    stock_a_indicator_lg_df = ak.stock_a_indicator_lg(symbol=symbol)
    pe_data = stock_a_indicator_lg_df[['trade_date', 'pe', 'pb', 'dv_ratio', 'total_mv']].copy()
    pe_data['trade_date'] = pd.to_datetime(pe_data['trade_date'])  # 转换为日期格式

    # 计算每个指标的平均值
    averages = {
        'pe': pe_data['pe'].mean(),
        'pb': pe_data['pb'].mean(),
        'dv_ratio': pe_data['dv_ratio'].mean(),
        'total_mv': pe_data['total_mv'].mean()
    }

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    for cur_index in ['pe', 'pb', 'dv_ratio', 'total_mv']:
        plt.figure(figsize=(12, 6))
        plt.plot(pe_data['trade_date'], pe_data[cur_index], label=cur_index, color='blue')
        
        # 绘制平均线
        plt.axhline(y=averages[cur_index], color='red', linestyle='--', label='平均值')

        # 添加分布线
        plt.axhline(y=1.4*averages[cur_index], color='blue', linestyle=':', label=f'{70}% 分线')
        plt.axhline(y=0.6*averages[cur_index], color='green', linestyle=':', label=f'{30}% 分线')


        plt.title(cur_index, fontproperties=my_font)
        plt.xlabel('日期', fontproperties=my_font)
        plt.ylabel(cur_index, fontproperties=my_font)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # 保存图表到指定路径
        plt.savefig(f'{save_path}/{symbol}_{cur_index}.png', dpi=300, bbox_inches='tight')
        plt.close()
#polt_data('601398', 'indicator')  # 替换为您希望保存图表的路径

# 对基本面单个股票可视化，或者多个股票进行对比
def plot_multiple_indicators_trend(json_file, save_path, indicators,start_year=None, end_year=None):
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 存储所有股票的指标数据
    all_indicators1 = {
        "短期借款": {},
        "长期借款": {},
        "*经营活动产生的现金流量净额": {},
        '资产负债率':{},
        '销售毛利率': {},
        '销售净利率': {},
        '净资产收益率': {},
        '扣非净利润同比增长率': {},
        "货币资金/负债合计": {},
        "经营净额/归母净利润": {},
        "应收/营收":{},
        "费用/毛总营收": {},
    }
    all_indicators = {cur:{} for cur in indicators}
    years = None  # 初始化年份

    # 遍历每个银行的数据
    for share_name, share_data in data.items():
        base_data = share_data['base_data']
        if years is None:
            years = sorted(base_data.keys())  # 获取年份

                # 过滤年份
        if start_year is not None:
            years = [year for year in years if year >= start_year]
        if end_year is not None:
            years = [year for year in years if year <= end_year]

        # 提取指标数据
        for year in years:
            if year in base_data:  # 检查年份是否存在
                for indicator in all_indicators.keys():
                    value = base_data[year].get(indicator)
                    value = convert_to_number(value)  # 转换为数值
                    all_indicators[indicator].setdefault(share_name, []).append(value)
            else:
                for indicator in all_indicators.keys():
                    all_indicators[indicator].setdefault(share_name, []).append(np.nan)  # 如果年份不存在，填充为NaN

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 绘制每个指标的变化曲线
    for indicator, indicator_data in all_indicators.items():
        plt.figure(figsize=(10, 6))
        for share_name, values in indicator_data.items():
            plt.plot(years, values, marker='o', linestyle='-', linewidth=2, label=share_name)

        # 计算平均值并绘制平均线
        #average_values = np.nanmean(np.array(list(indicator_data.values())), axis=0)  # 计算平均值
        #plt.plot(years, average_values, marker='o', linestyle='-', color='yellow', linewidth=2, label='平均值')

        # 设置 y 轴格式化
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}' if x < 0 else f'{x:.2f}'))


        if indicator == '货币资金/负债合计' or indicator == '经营净额/归母净利润':
            plt.axhline(y=1, color='red', linestyle='--', linewidth=2, label='标准线')
        elif indicator == '应收/营收' or indicator == '费用/毛总营收':
            plt.axhline(y=0.3, color='blue', linestyle='--', linewidth=2, label='标准线')
            if indicator == '应收/营收':
                plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='红线')
            else:
                plt.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='红线')

                
        elif indicator == '销售毛利率':
            plt.axhline(y=0.4, color='red', linestyle='--', linewidth=2, label='标准线')
        elif indicator == '销售净利率':
            plt.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='标准线')
        elif indicator == '净资产收益率':
            plt.axhline(y=0.15, color='red', linestyle='--', linewidth=2, label='标准线')
        elif indicator == '资产负债率':
            plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='标准线')


        # 设置图表标题和标签
        plt.title(f'{indicator} 变化曲线', fontproperties=my_font)
        plt.xlabel('年份', fontproperties=my_font)
        plt.ylabel(indicator, fontproperties=my_font)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()

        # 保存图表到指定路径
        sanitized_indicator = sanitize_filename(indicator)  # 清理文件名
        plt.tight_layout()
        plt.savefig(f'{save_path}/{sanitized_indicator}_trend.png', dpi=300)
        plt.close()
