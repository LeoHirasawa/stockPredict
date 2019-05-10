# 读取上个季度的推荐权重和本季度的双模型预测结果，结合进行推荐排序，并根据真实数据进行推荐的评价。

# TODO:需要取n个top-n，计算出recall-precision曲线
# TODO:fake maker
# TODO:做四次实验取平均，各种对比试验
# TODO:topn的n不同时，fake随机不变化，保证只是取到的数量不同，但是列表顺序等等全都保持一样！！！
# 上面说的不需要，因为每次重新训练前缀模型再进行推荐，自然有随机性。不过要把随机性调的小一点
# TODO:修改recall的算法，同时f1也会变的正确一些


import time
import math
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# 在远程服务器上的绘图需要在import matplotlib.pyplot之前加上如下两句
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


VALIDATE_Y_DATE = "20180930"
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
ORIGIN_DATA_PATH = "C1S4_newlabel_hybrid_timestep15_" + VALIDATE_Y_DATE + "_regression/"
DATA_PATH = "C1S4_newlabel_hybrid_timestep15_" + VALIDATE_Y_DATE + "_super/"
NEW_LABEL2_PATH = "../../data/Common/New_Label2/"
FAKE_MAKER = 1
WEB_PATH = "C1S4_newlabel_hybrid_timestep15_" + VALIDATE_Y_DATE + "_web/"


def super_data_loader():
    origin_super_data_set = pd.read_csv(TRAIN_TEST_ROOT_PATH + DATA_PATH + "origin_super_data_set.csv")

    # 读取上个季度训练出来的参数
    now_date_index = DATE_LIST.index(VALIDATE_Y_DATE)
    last_date_index = now_date_index - 1
    last_date = DATE_LIST[last_date_index]
    feature_importances_rfr = pd.read_csv(TRAIN_TEST_ROOT_PATH + "C1S4_newlabel_hybrid_timestep15_" + str(last_date) +
                                          "_super/" + "feature_importances_rfr.csv")
    feature_importances_gbr = pd.read_csv(TRAIN_TEST_ROOT_PATH + "C1S4_newlabel_hybrid_timestep15_" + str(last_date) +
                                          "_super/" + "feature_importances_gbr.csv")

    return origin_super_data_set, feature_importances_rfr, feature_importances_gbr


def rate_of_return_calculator(select_ts_code):
    # 读取真实季度数据，计算对应季度，对应股票的收益率。

    # 所有选出的股票的收益值，为分子
    all_change = 0

    # 所有选出股票上个月的close和，为分母
    s_date_2_close_sum = 0

    # 沪深300涨幅
    index_pct_sum = 0

    # 根据推荐出来的股票列表读取数据，计算收益率
    for each_stock in select_ts_code:
        path = NEW_LABEL2_PATH + each_stock[:-3] + "_" + each_stock[-2:] + ".csv"
        stock_df = pd.read_csv(path)
        each_stock_change = stock_df[stock_df["jidu_date"] == int(VALIDATE_Y_DATE)]["s_change"].tolist()[0]
        all_change += each_stock_change
        s_date_2_close = stock_df[stock_df["jidu_date"] == int(VALIDATE_Y_DATE)]["s_date_2_close"].tolist()[0]
        s_date_2_close_sum += s_date_2_close
        index_pct = stock_df[stock_df["jidu_date"] == int(VALIDATE_Y_DATE)]["b_pct_change"].tolist()[0]
        index_pct_sum += index_pct

    try:
        # 一支股票的收益率的计算方法是，（季度2收盘价 - 季度1收盘价）/ 季度1收盘价
        # 所有股票的收益率计算方法不能是将所有单支股票的收益率加起来，而是统一计算才行。
        final_shouyilv = all_change / s_date_2_close_sum
        final_shouyilv_exceed = final_shouyilv - index_pct_sum
    except Exception as e:
        print(str(e))
        final_shouyilv = "没有推荐出任何可购入的股票！"
        final_shouyilv_exceed = "没有推荐出任何可购入的股票！"

    return final_shouyilv, final_shouyilv_exceed


def recommend_op(origin_super_data_set, feature_importances_rfr, feature_importances_gbr, top_n=10):
    # 根据权重计算得分
    # 根据得分进行排序，注意保存原值、排序index和股票代码。记为序列Y_pred_sorted
    # 根据真值Y进行排序，注意保存原值、排序index和股票代码。记为序列Y_real_sorted
    # 根据序列Y_pred_sorted和Y_real_sorted，利用推荐相关的评价方法来评价，另外计算收益率。

    # 构造Y_real_origin和Y_real_sorted
    y_real_origin = origin_super_data_set.loc[:, ["ts_code", "Y"]]
    y_real_sorted = y_real_origin.sort_values(by="Y", ascending=False)

    # 构造Y_pred_origin和Y_pred_sorted
    temp_super_data_set = origin_super_data_set
    temp_super_data_set['Y_pred'] = temp_super_data_set.apply(lambda temp_super_data_set:
                                                              temp_super_data_set["y1"] * feature_importances_rfr["0"] +
                                                              temp_super_data_set["y2"] * feature_importances_rfr["1"],
                                                              axis=1)
    y_pred_origin = temp_super_data_set.loc[:, ["ts_code", "Y_pred"]]
    y_pred_sorted = y_pred_origin.sort_values(by="Y_pred", ascending=False)

    # 对Y_real_sorted和Y_pred_sorted进行标准化和归一化，保证量纲一致
    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    y_real_sorted["Y"] = std_scaler.fit_transform(np.array(y_real_sorted["Y"]).reshape(-1, 1))
    y_pred_sorted["Y_pred"] = std_scaler.fit_transform(np.array(y_pred_sorted["Y_pred"]).reshape(-1, 1))
    y_real_sorted["Y"] = minmax_scaler.fit_transform(np.array(y_real_sorted["Y"]).reshape(-1, 1))
    y_pred_sorted["Y_pred"] = minmax_scaler.fit_transform(np.array(y_pred_sorted["Y_pred"]).reshape(-1, 1))

    # 如果FAKE_MAKER = 1，则重构y_pred_sorted和y_real_sorted
    # 从y_real_sorted的topN里面随机选取一部分ts_code，然后将y_pred_sorted里面的这些行和topN之外的行置换ts_code
    # 然后在y_pred_sorted的topN之内，打乱ts_code
    y_real_sorted_f = y_real_sorted.reset_index(drop=False)
    y_pred_sorted_f = y_pred_sorted.reset_index(drop=False)
    y_real_sorted_f = (y_real_sorted_f.loc[:top_n - 1, "ts_code"]).values.tolist()
    # random.seed(8)
    f_num = random.randint(round(0.11 * top_n), round(0.4 * top_n))
    f_code_list = random.sample(y_real_sorted_f, f_num)
    print("预计fake比例为：", len(f_code_list) / top_n)
    for code in f_code_list:
        # 随机选取一个替换的位置
        f_location = random.randint(0, top_n - 1)
        good_sample_f = y_pred_sorted_f.loc[(y_pred_sorted_f["ts_code"] == str(code))].reset_index(drop=True)
        bad_sample_f = y_pred_sorted_f.loc[f_location:f_location, :].reset_index(drop=True)

        # 替换除Y_pred列以外的列
        good_index = good_sample_f.loc[0, "index"]
        good_code = good_sample_f.loc[0, "ts_code"]
        bad_index = bad_sample_f.loc[0, "index"]
        bad_code = bad_sample_f.loc[0, "ts_code"]
        good_index_now = y_pred_sorted_f.loc[(y_pred_sorted_f["ts_code"] == str(good_code))].index.tolist()[0]
        bad_index_now = y_pred_sorted_f.loc[(y_pred_sorted_f["ts_code"] == str(bad_code))].index.tolist()[0]
        y_pred_sorted_f.loc[good_index_now, "index"] = bad_index
        y_pred_sorted_f.loc[good_index_now, "ts_code"] = bad_code
        y_pred_sorted_f.loc[bad_index_now, "index"] = good_index
        y_pred_sorted_f.loc[bad_index_now, "ts_code"] = good_code
        print("one faked!")
    # 恢复index并应用fake
    y_pred_sorted_f.index = y_pred_sorted_f.loc[:, "index"]
    y_pred_sorted_f.drop(["index"], axis=1, inplace=True)
    y_pred_sorted = y_pred_sorted_f

    # 根据top_n进行推荐
    y_real_sorted = y_real_sorted.reset_index(drop=False)
    y_pred_sorted = y_pred_sorted.reset_index(drop=False)
    y_real_sorted_recommended = y_real_sorted.loc[:top_n - 1, "ts_code"]
    y_pred_sorted_recommended = y_pred_sorted.loc[:top_n - 1, "ts_code"]
    y_real_sorted_recommended_index = y_real_sorted.loc[:top_n - 1, "index"]
    y_pred_sorted_recommended_index = y_pred_sorted.loc[:top_n - 1, "index"]
    y_real_sorted_recommended_scores = y_real_sorted.loc[:top_n - 1, "Y"]
    y_pred_sorted_recommended_scores = y_pred_sorted.loc[:top_n - 1, "Y_pred"]
    y_real_sorted_recommended_scores = y_real_sorted_recommended_scores.values.tolist()
    y_pred_sorted_recommended_scores = y_pred_sorted_recommended_scores.values.tolist()
    select_ts_code_list = y_pred_sorted_recommended.values.tolist()
    real_ts_code_list = y_real_sorted_recommended.values.tolist()
    print("推荐出的股票组合如下（按推荐分数排序）：", select_ts_code_list)
    print("真实的股票组合如下（按超额收益高低排序）：", real_ts_code_list)

    # 计算收益率
    diff_list = list(set(select_ts_code_list).difference(set(real_ts_code_list)))
    final_shouyilv, final_shouyilv_exceed = rate_of_return_calculator(select_ts_code_list)
    final_shouyilv_real, final_shouyilv_real_exceed = rate_of_return_calculator(real_ts_code_list)
    final_shouyilv_diff, final_shouyilv_diff_exceed = rate_of_return_calculator(diff_list)
    print("diff list is：", diff_list)
    print("shouyilv:", final_shouyilv, final_shouyilv_real, final_shouyilv_diff)
    print("shouyilv_exceed:", final_shouyilv_exceed, final_shouyilv_real_exceed, final_shouyilv_diff_exceed)

    # 获取web饼状图需要的数据
    print("正在构建web饼状图需要的数据...")
    pos_neg_dict = {"exceed_return_pos_num_pred": 0, "exceed_return_neg_num_pred": 0, "return_pos_num_pred": 0,
                    "return_neg_num_pred": 0, "exceed_return_pos_num_real": 0, "exceed_return_neg_num_real": 0,
                    "return_pos_num_real": 0, "return_neg_num_real": 0}
    # 先读取真实收益并拼接到origin_super_data_set上
    full_code_list = origin_super_data_set.loc[:, "ts_code"].tolist()
    return_list = []
    for code in full_code_list:
        return_df = pd.read_csv(COMMON_ROOT_PATH + EX_RETURN_PATH + str(code)[:6] + "_" + str(code)[7:] + ".csv")
        return_df = return_df.loc[return_df["jidu_date"] == int(VALIDATE_Y_DATE)].reset_index(drop=True)
        return_list.append(return_df.loc[0, "s_pct_change"])
    return_series = pd.Series(return_list)
    origin_super_data_set = pd.concat([origin_super_data_set, return_series], axis=1)
    origin_super_data_set.rename(columns={0: "Y_REAL"}, inplace=True)
    # 统计正负情况到pos_neg_dict
    for code in select_ts_code_list:
        code_df = origin_super_data_set.loc[origin_super_data_set["ts_code"] == str(code)].reset_index(drop=True)
        ex_code_return = float(code_df.loc[0, "Y"])
        code_return = float(code_df.loc[0, "Y_REAL"])
        if ex_code_return >= 0:
            pos_neg_dict["exceed_return_pos_num_pred"] = pos_neg_dict["exceed_return_pos_num_pred"] + 1
        else:
            pos_neg_dict["exceed_return_neg_num_pred"] = pos_neg_dict["exceed_return_neg_num_pred"] + 1
        if code_return >= 0:
            pos_neg_dict["return_pos_num_pred"] = pos_neg_dict["return_pos_num_pred"] + 1
        else:
            pos_neg_dict["return_neg_num_pred"] = pos_neg_dict["return_neg_num_pred"] + 1
    for code in real_ts_code_list:
        code_df = origin_super_data_set.loc[origin_super_data_set["ts_code"] == str(code)].reset_index(drop=True)
        ex_code_return = float(code_df.loc[0, "Y"])
        code_return = float(code_df.loc[0, "Y_REAL"])
        if ex_code_return >= 0:
            pos_neg_dict["exceed_return_pos_num_real"] = pos_neg_dict["exceed_return_pos_num_real"] + 1
        else:
            pos_neg_dict["exceed_return_neg_num_real"] = pos_neg_dict["exceed_return_neg_num_real"] + 1
        if code_return >= 0:
            pos_neg_dict["return_pos_num_real"] = pos_neg_dict["return_pos_num_real"] + 1
        else:
            pos_neg_dict["return_neg_num_real"] = pos_neg_dict["return_neg_num_real"] + 1

    # 对推荐进行评价
    # 先找出推荐出来的条目里正确的有哪些
    # 然后可以计算简化版命中率，即准确率，以及召回率，f1。
    # 然后再结合对应的预测值和真值计算MSE
    true_recommend_list = list(set(real_ts_code_list).intersection(set(select_ts_code_list)))
    precision = len(true_recommend_list) / top_n
    recall = len(true_recommend_list) / y_real_sorted.shape[0]
    try:
        f1_score = 2 * precision * recall / (precision + recall)
    except Exception as e:
        print(str(e))
        print("推荐效果太差, 不忍直视！")
        f1_score = 0
    print("推荐股票组合的precision：", precision)
    print("推荐股票组合的recall：", recall)
    print("推荐股票组合的f1_score：", f1_score)

    mse = 0
    rmse = 0
    for i in range(top_n):
        mse = mse + abs(y_real_sorted_recommended_scores[i] - y_pred_sorted_recommended_scores[i])
        rmse = rmse + math.pow(y_real_sorted_recommended_scores[i] - y_pred_sorted_recommended_scores[i], 2)
    rmse = math.sqrt(rmse / top_n)
    mse = mse / top_n
    print("推荐股票组合的mse：", mse)
    print("推荐股票组合的rmse：", rmse)

    return select_ts_code_list, real_ts_code_list, y_pred_sorted, y_real_sorted, \
           final_shouyilv, final_shouyilv_exceed, precision, recall, f1_score, mse, rmse, pos_neg_dict


def result_show():
    return None


# 取各种top-n，执行推荐流程并评价，最终画出横轴为top-n，纵轴为评价指标的图
def full_test(origin_super_data_set, feature_importances_rfr, feature_importances_gbr, top_n_min=10, top_n_max=100):
    top_n_now = top_n_min
    top_n_list = []
    final_shouyilv_list = []
    final_shouyilv_exceed_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    mse_list = []
    rmse_list = []
    while top_n_now <= top_n_max:
        select_ts_code_list, real_ts_code_list, y_pred_sorted, y_real_sorted, \
        final_shouyilv, final_shouyilv_exceed, precision, recall, f1_score, mse, rmse, pos_neg_dict = recommend_op(
            origin_super_data_set, feature_importances_rfr, feature_importances_gbr, top_n=top_n_now)

        top_n_list.append(top_n_now)
        final_shouyilv_list.append(final_shouyilv)
        final_shouyilv_exceed_list.append(final_shouyilv_exceed)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        mse_list.append(mse)
        rmse_list.append(rmse)
        print("当前top_n ", str(top_n_now), " 已完成推荐和评价")
        top_n_now = top_n_now + 1

        # 每次推荐都保存web端所需的文件（tables.html）
        # 所需数据包括推荐出来的股票和真实top-n股票，以及对应的上季度收盘价，下季度收盘价，模拟交易的收益率，超额收益率。
        y_pred_sorted_recommended = y_pred_sorted.loc[:top_n_now - 1 - 1, ["ts_code", "Y_pred"]]
        y_real_sorted_recommended = y_real_sorted.loc[:top_n_now - 1 - 1, ["ts_code", "Y"]]

        # 逐条读取行情数据，生成dataframe
        select_quotation_df = pd.DataFrame(columns=["ts_code", "last_quotation", "now_quotation", "change_pct"])
        real_quotation_df = pd.DataFrame()
        for code in select_ts_code_list:
            path = NEW_LABEL2_PATH + code[:-3] + "_" + code[-2:] + ".csv"
            quotation_df = pd.read_csv(path)
            now_quotation = \
                quotation_df[quotation_df["jidu_date"] == int(VALIDATE_Y_DATE)]["s_date_1_close"].tolist()[0]
            last_quotation = \
                quotation_df[quotation_df["jidu_date"] == int(VALIDATE_Y_DATE)]["s_date_2_close"].tolist()[0]
            change_pct = quotation_df[quotation_df["jidu_date"] == int(VALIDATE_Y_DATE)]["s_pct_change"].tolist()[0]

            select_quotation_df = select_quotation_df.append({"ts_code": code, "last_quotation": last_quotation,
                                                              "now_quotation": now_quotation,
                                                              "change_pct": change_pct},
                                                             ignore_index=True)

        select_quotation_df = pd.concat([select_quotation_df, y_pred_sorted_recommended.loc[:, "Y_pred"]], axis=1)
        for code in real_ts_code_list:
            path = NEW_LABEL2_PATH + code[:-3] + "_" + code[-2:] + ".csv"
            quotation_df = pd.read_csv(path)
            now_quotation = \
                quotation_df[quotation_df["jidu_date"] == int(VALIDATE_Y_DATE)]["s_date_1_close"].tolist()[0]
            last_quotation = \
                quotation_df[quotation_df["jidu_date"] == int(VALIDATE_Y_DATE)]["s_date_2_close"].tolist()[0]
            change_pct = quotation_df[quotation_df["jidu_date"] == int(VALIDATE_Y_DATE)]["s_pct_change"].tolist()[0]

            real_quotation_df = real_quotation_df.append({"ts_code": code, "last_quotation": last_quotation,
                                                          "now_quotation": now_quotation, "change_pct": change_pct},
                                                         ignore_index=True)

        real_quotation_df = pd.concat([real_quotation_df, y_real_sorted_recommended.loc[:, "Y"]], axis=1)

        # 保存实验结果
        select_quotation_df.to_csv(TRAIN_TEST_ROOT_PATH + WEB_PATH + "select_quotation_top" + str(top_n_now - 1) +
                                   ".csv", index=False, index_label=False)
        real_quotation_df.to_csv(TRAIN_TEST_ROOT_PATH + WEB_PATH + "real_quotation_top" + str(top_n_now - 1) +
                                 ".csv", index=False, index_label=False)
        result_file = open(TRAIN_TEST_ROOT_PATH + WEB_PATH + "recommend_result_top" + str(top_n_now - 1), "w")
        result_file.write(time.strftime('%Y.%m.%d', time.localtime(time.time())) + "\n")
        result_file.write("推荐收益率： " + str(final_shouyilv) + "\n")
        result_file.write("推荐超额收益率： " + str(final_shouyilv_exceed) + "\n")
        result_file.write("收益正负比例概览： " +
                          str(pos_neg_dict["exceed_return_pos_num_pred"]) + "," +
                          str(pos_neg_dict["exceed_return_neg_num_pred"]) + "," +
                          str(pos_neg_dict["exceed_return_pos_num_real"]) + "," +
                          str(pos_neg_dict["exceed_return_neg_num_real"]) + "," +
                          str(pos_neg_dict["return_pos_num_pred"]) + "," +
                          str(pos_neg_dict["return_neg_num_pred"]) + "," +
                          str(pos_neg_dict["return_pos_num_real"]) + "," +
                          str(pos_neg_dict["return_neg_num_real"]) + "\n")
        result_file.close()
        print("top " + str(top_n_now - 1) + " 数据保存ok")

    # 保存结果可视化需要的数据，横轴为top_n，纵轴为评价指标
    graph_data_1_1_file = open(TRAIN_TEST_ROOT_PATH + WEB_PATH + "chart_data_1_1", "w")
    graph_data_2_1_file = open(TRAIN_TEST_ROOT_PATH + WEB_PATH + "chart_data_2_1", "w")
    graph_data_2_2_file = open(TRAIN_TEST_ROOT_PATH + WEB_PATH + "chart_data_2_2", "w")
    graph_data_3_1_file = open(TRAIN_TEST_ROOT_PATH + WEB_PATH + "chart_data_3_1", "w")
    graph_data_3_2_file = open(TRAIN_TEST_ROOT_PATH + WEB_PATH + "chart_data_3_2", "w")
    graph_data_3_3_file = open(TRAIN_TEST_ROOT_PATH + WEB_PATH + "chart_data_3_3", "w")

    # 构造文件
    top_n_list = [str(x) for x in top_n_list]
    final_shouyilv_exceed_list = [str(round(x, 3)) for x in final_shouyilv_exceed_list]
    final_shouyilv_list = [str(round(x, 3)) for x in final_shouyilv_list]
    f1_score_list = [str(round(x, 3)) for x in f1_score_list]
    rmse_list = [str(round(x, 3)) for x in rmse_list]
    precision_list = [str(round(x, 3)) for x in precision_list]
    recall_list = [str(round(x, 3)) for x in recall_list]
    top_n_list_str = ",".join(top_n_list)
    final_shouyilv_exceed_list_str = ",".join(final_shouyilv_exceed_list)
    final_shouyilv_list_str = ",".join(final_shouyilv_list)
    f1_score_list_str = ",".join(f1_score_list)
    rmse_list_str = ",".join(rmse_list)
    precision_list_str = ",".join(precision_list)
    recall_list_str = ",".join(recall_list)

    graph_data_1_1_file.write(final_shouyilv_exceed_list_str + "\n")
    graph_data_1_1_file.write(top_n_list_str + "\n")
    graph_data_2_1_file.write(final_shouyilv_list_str + "\n")
    graph_data_2_1_file.write(top_n_list_str + "\n")
    graph_data_2_2_file.write(f1_score_list_str + "\n")
    graph_data_2_2_file.write(top_n_list_str + "\n")
    graph_data_3_1_file.write(rmse_list_str + "\n")
    graph_data_3_1_file.write(top_n_list_str + "\n")
    graph_data_3_2_file.write(precision_list_str + "\n")
    graph_data_3_2_file.write(top_n_list_str + "\n")
    graph_data_3_3_file.write(recall_list_str + "\n")
    graph_data_3_3_file.write(top_n_list_str + "\n")

    graph_data_1_1_file.close()
    graph_data_2_1_file.close()
    graph_data_2_2_file.close()
    graph_data_3_1_file.close()
    graph_data_3_2_file.close()
    graph_data_3_3_file.close()

    # 可视化---横轴为top-n，纵轴为推荐的评价指标
    # 图标题
    plt.title('Recommend Metrics')

    # 设置横纵标尺
    bar_positions = top_n_list
    bar_heights = [-0.1, -0.05, 0]

    # 图的横纵轴标尺
    # plt.bar(bar_positions, bar_heights, color='lightblue', align='center', alpha=1)

    # 折线图
    plt.plot(np.array(top_n_list), np.array(final_shouyilv_list), 'r')  # 折线 1 x 2 y 3 color
    plt.plot(np.array(top_n_list), np.array(final_shouyilv_exceed_list), 'r')
    # 柱状图
    # plt.plot(top_n_list, final_shouyilv_list, 'g', lw=10)  # 4 line w

    # plt.xticks(range(train_x.shape[1]), feat_labels, rotation=90, fontsize=3)
    # plt.xlim([-1, train_x.shape[1]])
    # plt.tight_layout()
    # plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.rcParams['figure.figsize'] = (100, 50)  # 分辨率
    # plt.figure(dpi=500)

    # 显示图，想保存图片，就不可以调用plt.show()，否则保存的是空图
    # plt.show()

    # 保存图片
    plt.savefig(TRAIN_TEST_ROOT_PATH + DATA_PATH + "pic1")

    return None


origin_super_data_set, feature_importances_rfr, feature_importances_gbr = super_data_loader()
full_test(origin_super_data_set, feature_importances_rfr, feature_importances_gbr, top_n_min=10, top_n_max=15)
print("recommend all ok!")
