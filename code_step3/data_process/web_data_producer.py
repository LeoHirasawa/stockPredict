# 读取记载着超额收益相关数据的文件，读取训练集和验证集，改造成回归版本的训练集和验证集

import datetime
import numpy as np
import pandas as pd
import tushare as ts
import time

COMMON_ROOT_PATH = "../../data/Common/"
QUOTATION_ROOT_PATH = "../../data/Quotation_side/"
FINANCIAL_REPORT_ROOT_PATH = "../../data/Financial_side/"
TRAIN_TEST_ROOT_PATH = "../../data/Train&Test/"
EX_RETURN_PATH = "New_Label2/"
DATE_LIST = ["20130331", "20130630", "20130930", "20131231",
             "20140331", "20140630", "20140930", "20141231",
             "20150331", "20150630", "20150930", "20151231",
             "20160331", "20160630", "20160930", "20161231",
             "20170331", "20170630", "20170930", "20171231",
             "20180331", "20180630", "20180930", "20181231"]

# 每次生成数据之前，修改这几个参数
# 要读取的原数据
DATA_PATH = "C1S4_newlabel_hybrid_timestep15_20170930/"
# temp数据和最终输出数据的目标目录
TARGET_PATH = "C1S4_newlabel_hybrid_timestep15_20170930_regression/"
# 为了查找对应日期数据所要检查的训练集/验证集中的列名
CHECK_DATE_NAME = "end_date_1"


def data_producer():
    # 读取训练集，测试集和超额收益数据
    train_set = pd.read_csv(TRAIN_TEST_ROOT_PATH + DATA_PATH + "hybrid_train_set.csv")
    validate_set = pd.read_csv(TRAIN_TEST_ROOT_PATH + DATA_PATH + "hybrid_validate_set.csv")
    selected_stock_list_1 = train_set.loc[:, "ts_code"].drop_duplicates(keep='first').tolist()
    selected_stock_list_2 = validate_set.loc[:, "ts_code"].drop_duplicates(keep='first').tolist()
    selected_stock_list = list(set(selected_stock_list_1).union(set(selected_stock_list_2)))

    ex_return_csv_dict = {}
    for stock in selected_stock_list:
        stock_name = str(stock).replace(".", "_")
        ex_return_csv_dict[str(stock)] = pd.read_csv(COMMON_ROOT_PATH + EX_RETURN_PATH + stock_name + ".csv")

    # 按位置取数据并计算回归label
    train_set_row_num = train_set.shape[0]
    validate_set_row_num = validate_set.shape[0]

    for index in range(train_set_row_num):
        # 找出样本的股票代码和日期
        check_stock = train_set.loc[index, "ts_code"]
        check_date = train_set.loc[index, CHECK_DATE_NAME]
        end_date_index = DATE_LIST.index(str(check_date))
        check_end_date_index = end_date_index + 1
        check_end_date = DATE_LIST[check_end_date_index]

        # 读取所需的数据
        ex_return_df = ex_return_csv_dict[check_stock]
        check_df = ex_return_df.loc[ex_return_df["jidu_date"] == int(check_end_date)].reset_index(drop=True)
        pct_change = check_df.loc[0, "s_pct_change"] - check_df.loc[0, "b_pct_change"]

        # 替换label
        train_set.loc[index, "label"] = pct_change
        print(str(check_stock) + " " + str(check_date) + " ok!")

    for index in range(validate_set_row_num):
        # 找出样本的股票代码和日期
        check_stock = validate_set.loc[index, "ts_code"]
        check_date = validate_set.loc[index, CHECK_DATE_NAME]
        end_date_index = DATE_LIST.index(str(check_date))
        check_end_date_index = end_date_index + 1
        check_end_date = DATE_LIST[check_end_date_index]

        # 读取所需的数据
        ex_return_df = ex_return_csv_dict[check_stock]
        check_df = ex_return_df.loc[ex_return_df["jidu_date"] == int(check_end_date)].reset_index(drop=True)
        pct_change = check_df.loc[0, "s_pct_change"] - check_df.loc[0, "b_pct_change"]

        # 替换label
        validate_set.loc[index, "label"] = pct_change
        print("validate_set " + str(check_stock) + " " + str(check_date) + " ok!")

    # 保存新数据
    train_set.to_csv(TRAIN_TEST_ROOT_PATH + TARGET_PATH + "hybrid_train_set" + ".csv", index=False, index_label=False)
    validate_set.to_csv(TRAIN_TEST_ROOT_PATH + TARGET_PATH + "hybrid_validate_set" + ".csv", index=False,
                        index_label=False)
    return None


data_producer()
print("data all ok!")
