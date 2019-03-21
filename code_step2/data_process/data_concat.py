# 用来将时序特征拼接到原来的数据集上，形成完整数据集

import datetime
import numpy as np
import pandas as pd
import tushare as ts
import time

COMMON_ROOT_PATH = "../../data/Common/"
QUOTATION_ROOT_PATH = "../../data/Quotation_side/"
FINANCIAL_REPORT_ROOT_PATH = "../../data/Financial_side/"
TRAIN_TEST_ROOT_PATH = "../../data/Train&Test/"
DATE_LIST = ["20130331", "20130630", "20130930", "20131231",
             "20140331", "20140630", "20140930", "20141231",
             "20150331", "20150630", "20150930", "20151231",
             "20160331", "20160630", "20160930", "20161231",
             "20170331", "20170630", "20170930", "20171231",
             "20180331", "20180630", "20180930", "20181231"]

# 每次生成数据之前，修改这几个参数
# 要读取的原数据
DATA_PATH = "C1S4_newlabel_20171231/"
# temp数据和最终输出数据的目标目录
TARGET_PATH = "C1S4_newlabel_hybrid_timestep15_20171231/"
# 需要加入的时序特征列表
FEATURE_TO_CONCAT = ["trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]

# 以下三项根据原始数据的CnSm的n和m来确定
# 例如想要验证的日期为20180930，季度窗口为2，则训练集起始日期为20171231，验证集起始日期为20180331
# 训练集起始日期，从[XXXX0331, XXXX0630, XXXX0930, XXXX1231]中选择一个
VALIDATE_Y_DATE = "20171231"
# 数据集的季度数量
TRAIN_USE_SEASON_NUM = 1
# 滑动窗口大小，0代表不滑动
SLIDE_WINDOW_SIZE = 4

# 以下为时间相关特征的三个参数
# 从最终日期向回推多少天
DAYS = 90
# 取数据的频率(需要保证能整除DAYS)，例如90 / 6 = 15，15就是timestep
STEP = 6
# 容错倍率
FAULT_TOLERANT_RATE = 1.7


def data_reader():
    # 读取原数据,并获取股票和时间列表
    train_set = pd.read_csv(TRAIN_TEST_ROOT_PATH + DATA_PATH + "full_train_set.csv")
    validate_set = pd.read_csv(TRAIN_TEST_ROOT_PATH + DATA_PATH + "full_validate_set.csv")
    selected_stock_list = train_set.loc[:, "ts_code"].drop_duplicates(keep='first').tolist()

    # 确定需要取数据的DATE，考虑滑动窗口，所以train_set的DATE是一个列表，列表长度和滑动窗口次数相同
    end_date_index = DATE_LIST.index(VALIDATE_Y_DATE)
    end_date_index_validate = end_date_index - 1
    end_date_validate = DATE_LIST[end_date_index_validate]
    end_date_train_list = []
    end_date_full_list = []
    for slide_time in range(SLIDE_WINDOW_SIZE):
        end_date_index_train = end_date_index - 2 - slide_time
        end_date_train = DATE_LIST[end_date_index_train]
        end_date_train_list.append(end_date_train)
    end_date_full_list.append(end_date_validate)
    for end_date in end_date_train_list:
        end_date_full_list.append(end_date)

    # 根据股票列表读取时间相关数据
    quotation_data_dict = {}
    for ts_code in selected_stock_list:
        quotation_file_name = ts_code[:6] + "_" + ts_code[7:9] + "_quotation.csv"
        quotation_csv = pd.read_csv(QUOTATION_ROOT_PATH + quotation_file_name)
        quotation_data_dict[ts_code] = quotation_csv
        print(ts_code + " loaded!")

    # 取出对应位置的数据
    # 先定位到指定日期，然后根据days向前取n天数据，再按step按频率从中取数据。
    # 要注意验证时间段是否一致
    # 如果指定日期取不到，则向前找直到找到，但是不能太长
    processed_quotation_data_dict = {}
    error_code_list = []
    error_date_list = []
    for ts_code in selected_stock_list:
        csv_now = quotation_data_dict[ts_code]

        each_date_processed_quotation_data_dict = {}
        # 遍历日期列表
        for df_date in end_date_full_list:
            flag = 0
            back_days = 0
            # 字符串类型日期转datetime类型
            date_origin = datetime.datetime.strptime(str(df_date), "%Y%m%d")
            now_date = df_date
            while flag == 0:
                if back_days > 30:
                    # 如果向前回溯一个月都找不到行情，则认为这段时间股票处于停盘状态，删除各表中对应日期的条目，并继续。
                    # temp_df_list = []
                    # for df in df_list:
                    #     df = df.loc[~(df["end_date"] == df_date)]
                    #     df = df.reset_index(drop=True)
                    #     temp_df_list.append(df)
                    # df_list = temp_df_list
                    print(ts_code + "---" + df_date + " oh shit!!!!!!!!!!!!!!")
                    break
                else:
                    one_day_daily_df = csv_now.loc[csv_now["trade_date"] == int(now_date)]
                    if one_day_daily_df.empty:
                        # 如果财报发布当日找不到行情数据，则向前不断查找，直到找到
                        date_origin = date_origin - datetime.timedelta(days=1)
                        back_days = back_days + 1
                        now_date = date_origin.strftime("%Y%m%d %H:%M:%S")[:8]
                    else:
                        # 如果找到了对应日期的数据，则从该index向前查找DAYS个条目，取出来。
                        flag = 1
                        start_index = csv_now[csv_now["trade_date"] == int(now_date)].index.tolist()[0]
                        csv_temp = csv_now.ix[start_index:start_index + DAYS - 1, :]
                        # 反转（reverse）数据
                        csv_temp = csv_temp.iloc[::-1]
                        csv_temp = csv_temp.reset_index(drop=True)

                        # 按照指定的step取数据,并记录错误的数据，从数据集中放弃对应的样本。
                        csv_step = pd.DataFrame()
                        try:
                            jump_times = int(DAYS / STEP)
                            for i in range(jump_times):
                                now_index = (i+1) * STEP - 1
                                if csv_step.empty:
                                    csv_step = csv_temp.loc[now_index, :].to_frame().T
                                else:
                                    csv_step = csv_step.append(csv_temp.loc[now_index, :].to_frame().T)
                            csv_step.reset_index(drop=True)
                            # 将新找到的行情数据添加到dataframe里
                            each_date_processed_quotation_data_dict[df_date] = csv_step
                        except Exception as e:
                            print(str(ts_code) + " error!!! " + str(df_date) + " " + str(e))
                            error_code_list.append(ts_code)
                            error_date_list.append(df_date)

        # 各日期遍历完成后，将数据放入对应的dict中以待使用
        processed_quotation_data_dict[ts_code] = each_date_processed_quotation_data_dict
        print(ts_code + " quotation data pre-processed!")
    print("pre-processed!")

    # 数据错误检查

    # 返回读取的数据
    return train_set, validate_set, selected_stock_list, processed_quotation_data_dict


def data_concator(train_set, validate_set, processed_quotation_data_dict):
    # 逐条样本读取股票代码和日期，并将对应的时间相关数据拼接到对应位置
    train_set_code_list = train_set.loc[:, "ts_code"].tolist()
    train_set_date_list = train_set.loc[:, "end_date_" + str(TRAIN_USE_SEASON_NUM)].tolist()

    validate_set_code_list = validate_set.loc[:, "ts_code"].tolist()
    validate_set_date_list = validate_set.loc[:, "end_date_" + str(TRAIN_USE_SEASON_NUM)].tolist()

    # 遍历上述List,读取对应的时间相关特征数据，拼接在train_set上。
    train_set_sequential_data_df = pd.DataFrame()
    for index, code_now in enumerate(train_set_code_list):
        date_now = train_set_date_list[index]

        # 获取指定code和date的时间相关数据
        try:
            sequential_data_df_oneline = processed_quotation_data_dict[str(code_now)][str(date_now)]
        except Exception as e:
            # 数据不完整，由于在前面处理步骤中就被过滤了，所以查找不到
            print("train_set can't find the target data!" + str(code_now) + " " + str(date_now) + str(e))
            sequential_data_df_oneline = None

        # 根据dataframe构造一条样本所需要的特征
        train_sequential_data_to_concat = pd.DataFrame(columns=FEATURE_TO_CONCAT)
        check_flag = 0
        for fea_name in FEATURE_TO_CONCAT:
            if sequential_data_df_oneline is None:
                train_sequential_data_to_concat.loc[0, str(fea_name)] = np.nan
            else:
                one_feature = sequential_data_df_oneline.loc[:, str(fea_name)].tolist()
                # 对trade_date特征进行连续性检验
                if fea_name == "trade_date":
                    tolerant_days = DAYS * FAULT_TOLERANT_RATE
                    start_date = one_feature[0]
                    end_date = one_feature[-1]
                    start_date = datetime.datetime.strptime(str(start_date), "%Y%m%d")
                    end_date = datetime.datetime.strptime(str(end_date), "%Y%m%d")
                    data_days = (end_date - start_date).days
                    if tolerant_days < data_days:
                        # 数据不连续，直接放弃
                        check_flag = 1
                        print(str(code_now) + " train check error!!!")
                if check_flag == 0:
                    train_sequential_data_to_concat.loc[0, str(fea_name)] = one_feature
                else:
                    train_sequential_data_to_concat.loc[0, str(fea_name)] = np.nan
        # 将一条样本所需要的特征加入到大dataframe里
        if train_set_sequential_data_df.empty:
            train_set_sequential_data_df = train_sequential_data_to_concat
        else:
            train_set_sequential_data_df = train_set_sequential_data_df.append(train_sequential_data_to_concat)
        print(str(code_now) + " train_found!")

    # 遍历上述List,读取对应的时间相关特征数据，拼接在validate_set上。
    validate_set_sequential_data_df = pd.DataFrame()
    for index, code_now in enumerate(validate_set_code_list):
        date_now = validate_set_date_list[index]

        # 获取指定code和date的时间相关数据
        try:
            sequential_data_df_oneline = processed_quotation_data_dict[str(code_now)][str(date_now)]
        except Exception as e:
            # 数据不完整，由于在前面处理步骤中就被过滤了，所以查找不到
            print("validate_set can't find the target data!" + str(code_now) + " " + str(date_now) + str(e))
            sequential_data_df_oneline = None

        # 根据dataframe构造一条样本所需要的特征
        validate_sequential_data_to_concat = pd.DataFrame(columns=FEATURE_TO_CONCAT)
        check_flag = 0
        for fea_name in FEATURE_TO_CONCAT:
            if sequential_data_df_oneline is None:
                validate_sequential_data_to_concat.loc[0, str(fea_name)] = np.nan
            else:
                one_feature = sequential_data_df_oneline.loc[:, str(fea_name)].tolist()
                # 对trade_date特征进行连续性检验
                if fea_name == "trade_date":
                    tolerant_days = DAYS * FAULT_TOLERANT_RATE
                    start_date = one_feature[0]
                    end_date = one_feature[-1]
                    start_date = datetime.datetime.strptime(str(start_date), "%Y%m%d")
                    end_date = datetime.datetime.strptime(str(end_date), "%Y%m%d")
                    data_days = (end_date - start_date).days
                    if tolerant_days < data_days:
                        # 数据不连续，直接放弃
                        check_flag = 1
                        print(str(code_now) + " validate check error!!!")
                if check_flag == 0:
                    validate_sequential_data_to_concat.loc[0, str(fea_name)] = one_feature
                else:
                    validate_sequential_data_to_concat.loc[0, str(fea_name)] = np.nan
        # 将一条样本所需要的特征加入到大dataframe里
        if validate_set_sequential_data_df.empty:
            validate_set_sequential_data_df = validate_sequential_data_to_concat
        else:
            validate_set_sequential_data_df = validate_set_sequential_data_df.append(validate_sequential_data_to_concat)
        print(str(code_now) + " train_found!")

    # 进行拼接并写入新文件
    train_set_sequential_data_df = train_set_sequential_data_df.reset_index(drop=True)
    validate_set_sequential_data_df = validate_set_sequential_data_df.reset_index(drop=True)
    train_set_hybrid = pd.concat([train_set, train_set_sequential_data_df], axis=1)
    validate_set_hybrid = pd.concat([validate_set, validate_set_sequential_data_df], axis=1)

    # 删除有缺失值的样本
    for fea_name in FEATURE_TO_CONCAT:
        train_set_hybrid = train_set_hybrid.loc[~(train_set_hybrid[str(fea_name)].isna())]
        validate_set_hybrid = validate_set_hybrid.loc[~(validate_set_hybrid[str(fea_name)].isna())]

    # 写入文件
    train_set_hybrid.to_csv(TRAIN_TEST_ROOT_PATH + TARGET_PATH + "hybrid_train_set" + ".csv",
                            index=False, index_label=False)
    validate_set_hybrid.to_csv(TRAIN_TEST_ROOT_PATH + TARGET_PATH + "hybrid_validate_set" + ".csv",
                               index=False, index_label=False)


train_set, validate_set, selected_stock_list, processed_quotation_data_dict = data_reader()
print("data loaded!")
data_concator(train_set, validate_set, processed_quotation_data_dict)
print("data all ok!")
