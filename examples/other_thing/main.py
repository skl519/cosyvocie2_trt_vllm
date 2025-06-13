import json
import sys
import akshare as ak
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from util import *



def get_base_data(share_code):
    # 负债表
    stock_financial_debt_ths_df = ak.stock_financial_debt_ths(symbol=share_code, indicator="按年度")
    stock_financial_debt_ths_df.to_json(f'{share_code}_debt.json', orient='records', force_ascii=False, indent=4)
    sys.exit()

    # 利润表
    stock_financial_benefit_ths_df = ak.stock_financial_benefit_ths(symbol=share_code, indicator="按年度")
    # stock_financial_benefit_ths_df.to_json(f'{share_code}_benefit.json', orient='records', force_ascii=False, indent=4)

    # 现金流
    stock_financial_cash_ths_df = ak.stock_financial_cash_ths(symbol=share_code, indicator="按年度")
    # stock_financial_cash_ths_df.to_json(f'{share_code}_cash.json', orient='records', force_ascii=False, indent=4)

    # 扣非净利润同比增长率  净利润
    stock_financial_abstract_ths_df = ak.stock_financial_abstract_ths(symbol=share_code, indicator="按年度")
    # stock_financial_abstract_ths_df.to_json(f'{share_code}_abstract.json', orient='records', force_ascii=False, indent=4)
    # 提取需要的列
    debt_columns = ['报告期', '货币资金', '负债合计', '短期借款', '长期借款', '应收票据及应收账款', '其他应收款合计']
    benefit_columns = ['报告期', '*营业总收入', "*归属于母公司所有者的净利润", "销售费用", "管理费用", "研发费用","财务费用"]
    cash_columns = ['报告期', "*经营活动产生的现金流量净额", "*投资活动产生的现金流量净额","*筹资活动产生的现金流量净额"]
    abstract_columns = ['报告期', '资产负债率', '销售毛利率', '销售净利率', '净资产收益率', '扣非净利润同比增长率']
    # 创建一个新的数据框，只保留存在的列，并将不存在的列设置为 None
    debt_df = pd.DataFrame(columns=debt_columns)
    for col in debt_columns:
        if col in stock_financial_debt_ths_df.columns:
            debt_df[col] = stock_financial_debt_ths_df[col]
        else:
            debt_df[col] = np.nan  # 设置为 None
    benefit_df = pd.DataFrame(columns=benefit_columns)
    for col in benefit_columns:
        if col in stock_financial_benefit_ths_df.columns:
            benefit_df[col] = stock_financial_benefit_ths_df[col]
        else:
            benefit_df[col] = np.nan  # 设置为 None
    cash_df = pd.DataFrame(columns=cash_columns)
    for col in cash_columns:
        if col in stock_financial_cash_ths_df.columns:
            cash_df[col] = stock_financial_cash_ths_df[col]
        else:
            cash_df[col] = np.nan  # 设置为 None
    abstract_df = pd.DataFrame(columns=abstract_columns)
    for col in abstract_columns:
        if col in stock_financial_abstract_ths_df.columns:
            abstract_df[col] = stock_financial_abstract_ths_df[col]
        else:
            abstract_df[col] = np.nan  # 设置为 None

    # 转换数据
    debt_df['货币资金'] = debt_df['货币资金'].apply(convert_to_number)
    debt_df['负债合计'] = debt_df['负债合计'].apply(convert_to_number)
    debt_df['应收票据及应收账款'] = debt_df['应收票据及应收账款'].apply(convert_to_number)
    debt_df['其他应收款合计'] = debt_df['其他应收款合计'].apply(convert_to_number)
    debt_df['短期借款'] = debt_df['短期借款']
    debt_df['长期借款'] = debt_df['长期借款']

    benefit_df['营业总收入'] = benefit_df['*营业总收入'].apply(convert_to_number)
    benefit_df['归母净利润'] = benefit_df['*归属于母公司所有者的净利润'].apply(convert_to_number)
    benefit_df['销售费用'] = benefit_df["销售费用"].apply(convert_to_number)
    benefit_df['管理费用'] = benefit_df["管理费用"].apply(convert_to_number)
    benefit_df['研发费用'] = benefit_df["研发费用"].apply(convert_to_number)
    benefit_df['财务费用'] = benefit_df["财务费用"].apply(convert_to_number)

    cash_df['经营净额'] = cash_df["*经营活动产生的现金流量净额"].apply(convert_to_number)
    cash_df['投资净额'] = cash_df["*投资活动产生的现金流量净额"].apply(convert_to_number)
    cash_df['筹资净额'] = cash_df["*筹资活动产生的现金流量净额"].apply(convert_to_number)

    abstract_df['毛利率'] = abstract_df['销售毛利率'].apply(convert_to_number)

    # 合并数据
    df = pd.merge(debt_df, benefit_df, on='报告期', how='inner')
    df = pd.merge(df, cash_df, on='报告期', how='inner')
    df = pd.merge(df, abstract_df, on='报告期', how='inner')

    # 计算比值
    # 计算比值
    df['货币资金/负债合计'] = np.where(df['货币资金'].notna() & df['负债合计'].notna() & (df['负债合计'] != 0),
                                       df['货币资金'] / df['负债合计'], np.nan)
    df['应收/营收'] = np.where(
        df['营业总收入'].notna() & (df['营业总收入'] != 0) & df['应收票据及应收账款'].notna() & df[
            '其他应收款合计'].notna(), (df['应收票据及应收账款'] + df['其他应收款合计']) / df['营业总收入'], np.nan)
    df['经营净额/归母净利润'] = np.where(df['归母净利润'].notna() & (df['归母净利润'] != 0) & df['经营净额'].notna(),
                                         df['经营净额'] / df['归母净利润'], np.nan)
    df['费用/毛总营收'] = np.where(df['营业总收入'].notna() & (df['营业总收入'] != 0) & df['毛利率'].notna() & (
                df['销售费用'].notna() & df['管理费用'].notna() & df['研发费用'].notna() & df['财务费用'].notna()),
                                   (df['销售费用'] + df['管理费用'] + df['研发费用'] + df['财务费用']) / (
                                               df['营业总收入'] * df['毛利率']), np.nan)
    # 选择需要的列
    result_df = df[['报告期', '资产负债率', '短期借款', '长期借款', '货币资金/负债合计', '应收/营收',
                    '*经营活动产生的现金流量净额', '经营净额/归母净利润', '销售毛利率', '销售净利率', '净资产收益率',
                    '扣非净利润同比增长率', '费用/毛总营收', ]]
    result_df.set_index('报告期', inplace=True)
    result_df = result_df.to_dict(orient='index')
    return result_df


def get_estimate_data(share_code, pe=None, dv=None, z=None):
    # trade_date交易日  pe市盈率 pe_ttm市盈率TT pb市净 ps市销率  ps_ttm市销率TTM  dv_ratio股息率  dv_ttm股息率TTM  total_mv 总市值
    stock_a_indicator_lg_df = ak.stock_a_indicator_lg(symbol=share_code)
    print(stock_a_indicator_lg_df)
    # 总市值 PE(TTM)
    stock_value_em_df = ak.stock_value_em(symbol=share_code)
    # 扣非净利润同比增长率  净利润
    stock_financial_abstract_ths_df = ak.stock_financial_abstract_ths(symbol=share_code, indicator="按年度")

    abstract_df = stock_financial_abstract_ths_df[['报告期', '净利润', '扣非净利润同比增长率']].copy()
    abstract_df['扣非净利润同比增长率'] = abstract_df['扣非净利润同比增长率'].apply(
        lambda x: 0 if x is False or x is None else convert_to_number(x))
    latest_profit = abstract_df.iloc[-1]['净利润']  # 获取最后一行的净利润
    latest_profit_value = convert_to_number(latest_profit)  # 转换为数值

    latest_indicators = stock_a_indicator_lg_df.iloc[-1]  # 获取最后一行数据
    # latest_indicators = stock_value_em_df.iloc[-1]
    pe_ttm = stock_value_em_df.iloc[-1]['PE(TTM)'] if pe is None else pe
    dv_ratio = stock_a_indicator_lg_df.iloc[-1]['dv_ratio'] if pe is None else pe
    total_mv = stock_value_em_df.iloc[-1]['总市值'] if pe is None else pe

    # 获取最近5年的数据
    recent_5_years = abstract_df['扣非净利润同比增长率'].tail(5)

    # 定义平滑因子，ema加权，最近的数据更高的权重，而较早的数据权重逐渐减小
    n = len(recent_5_years)  # 期数
    alpha = 2 / (n + 1)
    ema = recent_5_years.iloc[0]  # 使用 iloc 访问第一个元素
    for growth_rate in recent_5_years.iloc[1:]:
        ema = alpha * growth_rate + (1 - alpha) * ema
    # 除以2 安全边际
    estimated_growth_rate = ema / 2

    deng = total_mv / (latest_profit_value * (1 + estimated_growth_rate) ** 5)
    zong = (100 * estimated_growth_rate + dv_ratio) / pe_ttm

    deng_need = (total_mv / (5 * latest_profit_value)) ** (1 / 5) - 1
    zong_need = (pe_ttm * 0.7 - dv_ratio) / 100

    print(f'预测增长率：{estimated_growth_rate:.2f}')

    print(f'邓的估值：{deng:.2f}')
    print(f'总的估值：{zong:.2f}')

    print(f'邓估值需要的增长率：{deng_need:.2%}')
    print(f'总估值需要的增长率:{zong_need:.2%}')
    print(latest_indicators)

    # 创建一个字典来保存需要的数据
    data_to_save = {
        '市值': format_with_unit(total_mv),
        'PE': pe_ttm,
        'DV': dv_ratio,
        '预测增长率': estimated_growth_rate,
        "deng": deng,
        "zong": zong,
        "deng_need": deng_need,
        "zong_need": zong_need
    }

    return data_to_save

def get_json_data(process_stock_codes, save_json_path='all_share_data.json'):
    all_estimates = {}
    for name, code in process_stock_codes.items():
        print(name)
        base_data = get_base_data(code)
        estimate_data = get_estimate_data(code)
        all_estimates[name] = {'base_data': base_data, 'estimate_data': estimate_data}
    with open(save_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_estimates, json_file, ensure_ascii=False, indent=4)

drink_stock_codes = {
    '贵州茅台': '600519',
    '山西汾酒': '600809',
    '泸州老窖': '000568',
    '五粮液': '000858',
}

banks_stock_codes = {
    '工商银行': '601398',
    '农业银行': '601288',
    '中国银行': '601988',
    '建设银行': '601939',
    '交通银行': '601328',
    '招商银行': '600036',
}

bank_other = {
    '民生银行': '600016',
    '平安银行': '000001',
    '兴业银行': '601166',
    '浦发银行': '600000',
    '中信银行': '601998',
    '光大银行': '601818',
}

milk_stock_codes = {
    '伊利股份': '600877',
    '光明乳业': '600597',
}

engery_stock_codes = {
    '长江电力': '600900',
    '中国核电': '601985',
    '华能国际': '600011'
}

bao_stock_code = {
    '中国人寿': '601628',
    '中国平安': '601318',
    '中国太保': '601601',
    '中国人保': '601319',
}


base_dir='白酒'
os.makedirs(base_dir, exist_ok=True) 
base_indicators= ['资产负债率','销售毛利率','销售净利率','净资产收益率','扣非净利润同比增长率',]


get_json_data(milk_stock_codes, save_json_path=f'{base_dir}/all_share_data.json')


plot_multiple_indicators_trend(json_file=f'{base_dir}/all_share_data.json', save_path=f'{base_dir}/base_data',indicators=base_indicators, start_year='2010')



