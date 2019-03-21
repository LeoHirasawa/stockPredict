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
# temp数据和最终输出数据的目标目录
TARGET_PATH = "C1S4_newlabel_20171231/"
# 需要连接的表
TABLE_TO_CONCAT = "1,2,3,7"
# 数据集的季度数量
TRAIN_USE_SEASON_NUM = 1
# 例如想要验证的日期为20180930，季度窗口为2，则训练集起始日期为20171231，验证集起始日期为20180331
# 训练集起始日期，从[XXXX0331, XXXX0630, XXXX0930, XXXX1231]中选择一个
VALIDATE_Y_DATE = "20171231"
# label是读取还是计算，0表示读取
LABEL_CALCULATE = 0
LABEL_PATH = "New_Label2/"
# 滑动窗口大小，0代表不滑动
SLIDE_WINDOW_SIZE = 4


def training_data_creator(stock_code, table_path_list):
    # 读取需要拼接的表数据
    csv_path_list = []
    for path in table_path_list:
        csv_name = stock_code[:6] + "_" + stock_code[7:9] + "_" + path[:-1] + ".csv"
        csv_path_list.append(FINANCIAL_REPORT_ROOT_PATH + path + csv_name)

    # 读取所有要拼接的csv表,并检查
    df_list = []
    for csv_path in csv_path_list:
        df = pd.read_csv(csv_path)
        # 按日期去重
        df.drop_duplicates(subset=["end_date"], keep='first', inplace=True)
        df = df.reset_index(drop=True)
        df_list.append(df)
    if len(df_list) == 0:
        print(stock_code + " has zero data csv!!!")
        return None, None
    for df in df_list:
        if df.empty:
            print(stock_code + " data csv is empty!!!")
            return None, None

    # 读取股票行情csv
    quotation_csv_name = stock_code[:6] + "_" + stock_code[7:9] + "_" + "quotation.csv"
    daily_df = pd.read_csv(QUOTATION_ROOT_PATH + quotation_csv_name)
    daily_df.drop_duplicates(subset=["trade_date"], keep='first', inplace=True)
    daily_df = daily_df.reset_index(drop=True)

    # 获取财报日期列表
    # 检查是否多个表的条数不一致，若不一致，选择条目最少的交集，删除其他条目，保证多个表的行数相同
    dates_list = []
    for df in df_list:
        dates = df.loc[:, "end_date"].tolist()
        dates_list.append(dates)
    dates_intersection = dates_list[0]
    for dates in dates_list:
        dates_intersection = list(set(dates_intersection).intersection(set(dates)))
    dates_intersection = sorted(dates_intersection, reverse=True)

    # 每个表只选取日期在交集内的条目
    df_list_temp = []
    for df in df_list:
        df = df.loc[df["end_date"].isin(dates_intersection), :]
        df = df.reset_index(drop=True)
        df_list_temp.append(df)
    df_list = df_list_temp

    # 根据财报日期列表拿到对应日期的行情
    df_date_list = df_list[0].loc[:, "end_date"].tolist()

    selected_daily_df = pd.DataFrame()
    for df_date in df_date_list:
        flag = 0
        back_days = 0
        # 字符串类型日期转datetime类型
        date_origin = datetime.datetime.strptime(str(df_date), "%Y%m%d")
        now_date = df_date
        while flag == 0:
            if back_days > 30:
                # 如果向前回溯一个月都找不到行情，则认为这段时间股票处于停盘状态，删除各表中对应日期的条目，并继续。
                temp_df_list = []
                for df in df_list:
                    df = df.loc[~(df["end_date"] == df_date)]
                    df = df.reset_index(drop=True)
                    temp_df_list.append(df)
                df_list = temp_df_list
                break
            else:
                one_day_daily_df = daily_df.loc[daily_df["trade_date"] == int(now_date)]
                if one_day_daily_df.empty:
                    # 如果财报发布当日找不到行情数据，则向前不断查找，直到找到
                    date_origin = date_origin - datetime.timedelta(days=1)
                    back_days = back_days + 1
                    now_date = date_origin.strftime("%Y%m%d %H:%M:%S")[:8]
                else:
                    flag = 1
                    # 将新找到的行情数据添加到dataframe里
                    if selected_daily_df.empty:
                        selected_daily_df = one_day_daily_df
                    else:
                        selected_daily_df = selected_daily_df.append(one_day_daily_df)
    selected_daily_df = selected_daily_df.reset_index(drop=True)

    # 拼接df_list中的所有df
    # 然后，将selected_daily_df的收盘价列取出来转型成dataframe，然后重置index并不保留index，拼接到每行最后一列。
    if selected_daily_df.empty:
        print(stock_code + " has zero quotation data!!!")
        return None, None
    else:
        try:
            full_df = pd.DataFrame()
            for df in df_list:
                if full_df.empty:
                    full_df = df
                else:
                    # 删除掉一些可能重复的列
                    drop_col_name = ["ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type"]
                    for name in drop_col_name:
                        try:
                            df.drop([name], axis=1, inplace=True)
                        except Exception as e:
                            # print(e)
                            continue
                    full_df = pd.concat([full_df, df], axis=1)

            full_df = full_df.reset_index(drop=True)
            selected_daily_df = selected_daily_df["close"].to_frame().reset_index(drop=True)
            full_df = pd.concat([full_df, selected_daily_df], axis=1)
        except Exception as e:
            print(e)
            return None, None

    if SLIDE_WINDOW_SIZE == 0:
        # 首先根据所需的日期把对应的数据选取出来，如果取出过程中发现出现异常，则说明该数据在这段时间内是不可用的，放弃
        end_date_index = DATE_LIST.index(VALIDATE_Y_DATE)
        start_date_index = end_date_index - TRAIN_USE_SEASON_NUM - 1
        need_date_list = DATE_LIST[start_date_index:end_date_index + 1]
        try:
            full_df = full_df.loc[full_df["end_date"].isin(need_date_list), :]
            full_df = full_df.reset_index(drop=True)
            if full_df.shape[0] != len(need_date_list):
                print("some season data missing!!! abandon...")
                return None, None
        except Exception as e:
            print(e)
            return None, None

        # 窗口不滑动，取最近的数据构造样本集
        index_count = full_df.shape[0]
        # 比如训练和验证时使用两个季度的数据，则从index=2开始，向上取出数据
        start_index = TRAIN_USE_SEASON_NUM + 1
        # 创建训练集和验证集
        train_df = pd.DataFrame()
        validate_df = pd.DataFrame()

        # 验证数据条数是否够用来构造训练集和验证集
        if index_count < start_index + 1:
            return None, None
        else:
            # 每次循环得到一条样本，每次先判断N个季度是否连续，若不连续则continue
            end_date_list = []
            for season in range(TRAIN_USE_SEASON_NUM):
                end_date_list.append(full_df.ix[start_index - season, "end_date"])

            # 检查季度是否连续，continue_flag为0表示不连续，1为连续
            continue_flag = 1
            for index, end_date in enumerate(end_date_list):
                if index + 1 < TRAIN_USE_SEASON_NUM:
                    end_date_pre = datetime.datetime.strptime(str(end_date), "%Y%m%d")
                    end_date_next = datetime.datetime.strptime(str(end_date_list[index + 1]), "%Y%m%d")
                    days_delta = (end_date_next - end_date_pre).days
                    if days_delta >= 95:
                        continue_flag = 0
                        break

            # 当前窗口中N个季度若连续，则执行拼接，否则该数据不合法，只能舍弃
            if continue_flag == 1:
                # 先计算好训练集和验证集的两个label,第一个是训练集的label，第二个是验证集的
                label_list = []
                if LABEL_CALCULATE == 0:
                    # 直接读取label
                    csv_name = stock_code[:6] + "_" + stock_code[7:9] + ".csv"
                    label_csv = pd.read_csv(COMMON_ROOT_PATH + LABEL_PATH + csv_name)
                    for i in range(2):
                        try:
                            date_to_read = full_df.loc[start_index - TRAIN_USE_SEASON_NUM - i, "end_date"]
                            selected_label_df = label_csv.loc[(label_csv["jidu_date"] == date_to_read)]
                            label = selected_label_df.reset_index(drop=True).loc[0, "label"]
                            label_list.append(label)
                        except Exception as e:
                            label_list.append(404)
                else:
                    for i in range(2):
                        next_close = full_df.loc[start_index - TRAIN_USE_SEASON_NUM - i, "close"]
                        pre_close = full_df.loc[start_index - TRAIN_USE_SEASON_NUM + 1 - i, "close"]
                        if next_close >= pre_close:
                            label = 1
                        else:
                            label = 0
                        label_list.append(label)

                # 将指定的季度滑动拼接成两条样本，分别作为训练集样本和验证集样本
                one_line_df_list = []
                for i in range(2):
                    season = i
                    one_line_df = pd.DataFrame()
                    while season < TRAIN_USE_SEASON_NUM + i:
                        if one_line_df.empty:
                            one_line_df = full_df.ix[start_index - season, :].to_frame().T
                            one_line_df.rename(columns=lambda x: x + "_" + str(season + 1 - i), inplace=True)
                            # 将ts_code列名字改回来
                            one_line_df.rename(columns={'ts_code_1': 'ts_code'}, inplace=True)
                            one_line_df = one_line_df.reset_index(drop=True)
                        else:
                            temp_df = full_df.ix[start_index - season, :].to_frame().T
                            del temp_df['ts_code']
                            temp_df.rename(columns=lambda x: x + "_" + str(season + 1 - i), inplace=True)
                            temp_df = temp_df.reset_index(drop=True)
                            one_line_df = pd.concat([one_line_df, temp_df], axis=1)
                        season = season + 1
                    one_line_df_list.append(one_line_df)

                # 把对应的label拼上,并将生成好的数据集拼接到train_df和validate_df中
                for i in range(2):
                    one_line_df_temp = one_line_df_list[i]
                    one_line_df_temp["label"] = label_list[i]
                    if i == 0:
                        if train_df.empty:
                            train_df = one_line_df_temp
                        else:
                            train_df = train_df.append(one_line_df_temp)
                    else:
                        if validate_df.empty:
                            validate_df = one_line_df_temp
                        else:
                            validate_df = validate_df.append(one_line_df_temp)
            else:
                # 当前数据季度不连续，例如[20170331, 20170630, 20171231, 20180331]，缺少了一个季度，则不能作为合法数据使用
                return None, None

        # 重置index
        train_df.reset_index(drop=True)
        validate_df.reset_index(drop=True)
    else:
        # 按照SLIDE_WINDOW_SIZE滑动窗口，构建数据集

        # 创建完整数据集
        slide_train_df = pd.DataFrame()
        slide_validate_df = pd.DataFrame()

        # 循环进行日期窗口滑动，每次循环生成一个日期的训练集样本和验证集样本
        for i in range(SLIDE_WINDOW_SIZE):
            # range(N), if N=4, i will be 0,1,2,3.
            # 算出当前需要生成样本的日期
            end_date_index = DATE_LIST.index(VALIDATE_Y_DATE) - (SLIDE_WINDOW_SIZE - i - 1)
            # 根据所需的日期把对应的数据选取出来，如果取出过程中发现出现异常，则说明该数据在这段时间内是不可用的，放弃
            start_date_index = end_date_index - TRAIN_USE_SEASON_NUM - 1
            need_date_list = DATE_LIST[start_date_index:end_date_index + 1]
            try:
                # 窗口滑动需要多次使用原数据，所以不能覆盖原数据
                temp_full_df = full_df.loc[full_df["end_date"].isin(need_date_list), :]
                temp_full_df = temp_full_df.reset_index(drop=True)
                if temp_full_df.shape[0] != len(need_date_list):
                    # 对应日期的数据缺失，继续滑动
                    print("some season data missing!!! continue slide...")
                    continue
            except Exception as e:
                print(e)
                continue

            # 窗口不滑动，取最近的数据构造样本集
            index_count = temp_full_df.shape[0]
            # 比如训练和验证时使用两个季度的数据，则从index=2开始，向上取出数据
            start_index = TRAIN_USE_SEASON_NUM + 1
            # 创建训练集和验证集
            train_df = pd.DataFrame()
            validate_df = pd.DataFrame()

            # 验证数据条数是否够用来构造训练集和验证集
            if index_count < start_index + 1:
                continue
            else:
                # 每次循环得到一条样本，每次先判断N个季度是否连续，若不连续则continue
                end_date_list = []
                for season in range(TRAIN_USE_SEASON_NUM):
                    end_date_list.append(temp_full_df.ix[start_index - season, "end_date"])

                # 检查季度是否连续，continue_flag为0表示不连续，1为连续
                continue_flag = 1
                for index, end_date in enumerate(end_date_list):
                    if index + 1 < TRAIN_USE_SEASON_NUM:
                        end_date_pre = datetime.datetime.strptime(str(end_date), "%Y%m%d")
                        end_date_next = datetime.datetime.strptime(str(end_date_list[index + 1]), "%Y%m%d")
                        days_delta = (end_date_next - end_date_pre).days
                        if days_delta >= 95:
                            continue_flag = 0
                            break

                # 当前窗口中N个季度若连续，则执行拼接，否则该数据不合法，只能舍弃
                if continue_flag == 1:
                    # 先计算好训练集和验证集的两个label,第一个是训练集的label，第二个是验证集的
                    label_list = []
                    if LABEL_CALCULATE == 0:
                        # 直接读取label
                        csv_name = stock_code[:6] + "_" + stock_code[7:9] + ".csv"
                        label_csv = pd.read_csv(COMMON_ROOT_PATH + LABEL_PATH + csv_name)
                        for j in range(2):
                            try:
                                date_to_read = temp_full_df.loc[start_index - TRAIN_USE_SEASON_NUM - j, "end_date"]
                                selected_label_df = label_csv.loc[(label_csv["jidu_date"] == date_to_read)]
                                label = selected_label_df.reset_index(drop=True).loc[0, "label"]
                                label_list.append(label)
                            except Exception as e:
                                label_list.append(404)
                    else:
                        for j in range(2):
                            next_close = temp_full_df.loc[start_index - TRAIN_USE_SEASON_NUM - j, "close"]
                            pre_close = temp_full_df.loc[start_index - TRAIN_USE_SEASON_NUM + 1 - j, "close"]
                            if next_close >= pre_close:
                                label = 1
                            else:
                                label = 0
                            label_list.append(label)

                    # 将指定的季度滑动拼接成两条样本，分别作为训练集样本和验证集样本
                    one_line_df_list = []
                    for j in range(2):
                        season = j
                        one_line_df = pd.DataFrame()
                        while season < TRAIN_USE_SEASON_NUM + j:
                            if one_line_df.empty:
                                one_line_df = temp_full_df.ix[start_index - season, :].to_frame().T
                                one_line_df.rename(columns=lambda x: x + "_" + str(season + 1 - j), inplace=True)
                                # 将ts_code列名字改回来
                                one_line_df.rename(columns={'ts_code_1': 'ts_code'}, inplace=True)
                                one_line_df = one_line_df.reset_index(drop=True)
                            else:
                                temp_df = temp_full_df.ix[start_index - season, :].to_frame().T
                                del temp_df['ts_code']
                                temp_df.rename(columns=lambda x: x + "_" + str(season + 1 - j), inplace=True)
                                temp_df = temp_df.reset_index(drop=True)
                                one_line_df = pd.concat([one_line_df, temp_df], axis=1)
                            season = season + 1
                        one_line_df_list.append(one_line_df)

                    # 把对应的label拼上,并将生成好的数据集拼接到train_df和validate_df中
                    for j in range(2):
                        one_line_df_temp = one_line_df_list[j]
                        one_line_df_temp["label"] = label_list[j]
                        if j == 0:
                            if train_df.empty:
                                train_df = one_line_df_temp
                            else:
                                train_df = train_df.append(one_line_df_temp)
                        else:
                            if validate_df.empty:
                                validate_df = one_line_df_temp
                            else:
                                validate_df = validate_df.append(one_line_df_temp)
                else:
                    # 当前数据季度不连续，例如[20170331, 20170630, 20171231, 20180331]
                    # 缺少了一个季度，则不能作为合法数据使用,继续循环，滑动窗口
                    continue

            # 重置index
            train_df.reset_index(drop=True)
            validate_df.reset_index(drop=True)

            # 将新生成的样本拼接到完整数据集中
            if slide_train_df.empty:
                slide_train_df = train_df
            else:
                slide_train_df = pd.concat([slide_train_df, train_df], axis=0)
            if slide_validate_df.empty:
                slide_validate_df = validate_df
            else:
                slide_validate_df = pd.concat([slide_validate_df, validate_df], axis=0)

        # 重命名完整样本集
        if slide_train_df.empty:
            train_df = None
        else:
            train_df = slide_train_df.reset_index(drop=True).sort_index(ascending=False, axis=0)
        if slide_validate_df.empty:
            validate_df = None
        else:
            validate_df = slide_validate_df.iloc[-1, :].to_frame().T

    return train_df, validate_df


def selected_stock_traverse(table_to_concat="1,2,3"):
    # 解析参数，得到要拼接的表所在的路径列表
    table_index_list = table_to_concat.split(",")
    table_path_list = []
    for index in table_index_list:
        if index == "1":
            table_path_list.append("income/")
        elif index == "2":
            table_path_list.append("balancesheet/")
        elif index == "3":
            table_path_list.append("cashflow/")
        elif index == "4":
            table_path_list.append("forecast/")
        elif index == "5":
            table_path_list.append("express/")
        elif index == "6":
            table_path_list.append("dividend/")
        elif index == "7":
            table_path_list.append("fina_indicator/")
        elif index == "8":
            table_path_list.append("fina_audit/")
        else:
            table_path_list.append("fina_mainbz/")

    # 获取要作为样本的股票代码列表
    selected_stock_df = pd.read_csv(COMMON_ROOT_PATH + "selected_stock.csv")
    selected_stock_list = selected_stock_df.loc[:, "ts_code"].tolist()

    # 创建最终返回的完整数据集
    full_train_df = pd.DataFrame()
    full_validate_df = pd.DataFrame()

    # 读取断点续传记录文件（记录的是原始数据和label进行自由拼接的过程）
    train_test_record_file = open(TRAIN_TEST_ROOT_PATH + TARGET_PATH + "Data_record", "r")
    lines = train_test_record_file.readlines()
    finished_list = []
    for line in lines:
        finished_list.append(str(line)[:9])
    train_test_record_file.close()

    # 得到还没有处理的股票列表
    difference_list = list(set(selected_stock_list).difference(set(finished_list)))

    for stock_code in difference_list:
        print("now " + str(stock_code) + " ......")
        # 针对每支股票读取财报数据中指定的表，和对应的日期的行情数据，进行拼接后，根据滑动窗口（滑动窗口可以为0）得到一系列样本
        one_train_df, one_validate_df = training_data_creator(stock_code, table_path_list)
        if one_train_df is None or one_validate_df is None:
            print(stock_code + "error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue
        else:
            # 将Dataframe拼接到full_train_df和full_validate_df中
            # if full_train_df.empty:
            #     full_train_df = one_train_df
            # else:
            #     full_train_df = full_train_df.append(one_train_df)
            # if full_validate_df.empty:
            #     full_validate_df = one_validate_df
            # else:
            #     full_validate_df = full_validate_df.append(one_validate_df)

            # 为了保险起见，每个文件处理完以后，单独保存
            one_train_df.to_csv(TRAIN_TEST_ROOT_PATH + TARGET_PATH + str(stock_code) + "_train" + ".csv",
                                index=False, index_label=False)
            one_validate_df.to_csv(TRAIN_TEST_ROOT_PATH + TARGET_PATH + str(stock_code) + "_validate" + ".csv",
                                   index=False, index_label=False)

            # 完成的股票写入文件，以备断点续传
            print(str(stock_code) + " data process OK!")
            train_test_record_file = open(TRAIN_TEST_ROOT_PATH + TARGET_PATH + "Data_record", "a")
            train_test_record_file.write(str(stock_code) + " data process OK!" + "\n")
            train_test_record_file.close()


def train_data_concator():
    # 把处理好了的train和validate数据拼接到一整个数据集中以备训练和验证使用
    selected_stock_df = pd.read_csv(COMMON_ROOT_PATH + "selected_stock.csv")
    selected_stock_list = selected_stock_df.loc[:, "ts_code"].tolist()

    full_train_df = pd.DataFrame()
    full_validate_df = pd.DataFrame()

    # 列数检查
    check_stock_train_df = pd.read_csv(TRAIN_TEST_ROOT_PATH + TARGET_PATH + str("000001.SZ") + "_train" + ".csv")
    fea_num = check_stock_train_df.shape[1]

    for stock_code in selected_stock_list:
        # 拼接train_set
        try:
            one_stock_train_df = pd.read_csv(TRAIN_TEST_ROOT_PATH + TARGET_PATH + str(stock_code) + "_train" + ".csv")
            if one_stock_train_df.empty:
                print(stock_code + ".csv is empty!")
                continue
            elif one_stock_train_df.shape[1] != fea_num:
                print(stock_code + ".csv(train) feature num miss!!!")
                continue
            else:
                if full_train_df.empty:
                    full_train_df = one_stock_train_df
                else:
                    full_train_df = full_train_df.append(one_stock_train_df)
                print(stock_code + ".csv concat ok!")
        except Exception as e:
            print(stock_code + " error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(e)
            continue

        # 拼接validate_set
        try:
            one_stock_validate_df = pd.read_csv(TRAIN_TEST_ROOT_PATH + TARGET_PATH + str(stock_code) +
                                                "_validate" + ".csv")
            if one_stock_validate_df.empty:
                print(stock_code + ".csv is empty!")
                continue
            elif one_stock_validate_df.shape[1] != fea_num:
                print(stock_code + ".csv(validate) feature num miss!!!")
                continue
            else:
                if full_validate_df.empty:
                    full_validate_df = one_stock_validate_df
                else:
                    full_validate_df = full_validate_df.append(one_stock_validate_df)
                print(stock_code + ".csv concat ok!")
        except Exception as e:
            print(stock_code + " error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(e)
            continue

    # 删除label有问题的行
    full_train_df = full_train_df.loc[~(full_train_df["label"] == 404)]
    full_validate_df = full_validate_df.loc[~(full_validate_df["label"] == 404)]

    # 从验证集中筛掉不对的日期
    end_date_index = DATE_LIST.index(VALIDATE_Y_DATE)
    start_date_index = end_date_index - TRAIN_USE_SEASON_NUM
    target_date = DATE_LIST[start_date_index]
    full_validate_df = full_validate_df.loc[full_validate_df["end_date_1"] == int(target_date)]

    # 写csv
    full_train_df.to_csv(TRAIN_TEST_ROOT_PATH + TARGET_PATH + "full_train_set.csv", index=False, index_label=False)
    full_validate_df.to_csv(TRAIN_TEST_ROOT_PATH + TARGET_PATH + "full_validate_set.csv",
                            index=False, index_label=False)


# 参数为要参与拼接的表
selected_stock_traverse(table_to_concat=TABLE_TO_CONCAT)
print("each data ok!")
train_data_concator()
print("data all ok!")
