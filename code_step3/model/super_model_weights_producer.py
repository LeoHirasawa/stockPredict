# 读取多个数据集，训练得到推荐需要用到的超模型权重，并进行保存。

# 将1500条样本分成训练集和验证集的方法有问题，因为在预测的时候，有很多股票直接就没参加
# 正常来讲，训练出了参数，模型就是给下个季度用的，所以应该用下个季度数据的来参与预测

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


VALIDATE_Y_DATE = "20170930"
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


def super_data_producer():
    # 不读取真实数据，因为真实数据的y2和Y是相同的
    # 只能读取分类和回归得到的结果和真实的超额收益量，分别作为y1,y2和Y来构建数据集。
    # 不将该数据集分为训练集和验证集，而是全部参加训练。只为了得到权重，保存后供给下个季度的数据用来推荐。
    # （正常应该把下个季度的数据当作验证集）
    left = pd.read_csv(TRAIN_TEST_ROOT_PATH + DATA_PATH + "super_validate_data_left.csv")
    right = pd.read_csv(TRAIN_TEST_ROOT_PATH + DATA_PATH + "super_validate_data_right.csv")
    left.columns = ["y1"]
    right.columns = ["y2", "Y"]
    super_data_set = pd.concat([left, right], axis=1)

    # 保留一份未进行过标准化和归一化的原数据，以便在推荐后进行评价
    ultra_super_data_set = super_data_set
    origin_validate_data = pd.read_csv(TRAIN_TEST_ROOT_PATH + ORIGIN_DATA_PATH + "hybrid_validate_set.csv")
    stock_code_df = origin_validate_data.loc[:, ["ts_code"]]
    ultra_origin_super_data_set = pd.concat([stock_code_df, ultra_super_data_set], axis=1)

    # 对x_train和x_validate的左半部分进行标准化和归一化
    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    super_data_set["y1"] = std_scaler.fit_transform(np.array(super_data_set["y1"]).reshape(-1, 1))
    super_data_set["y2"] = std_scaler.fit_transform(np.array(super_data_set["y2"]).reshape(-1, 1))
    super_data_set["y1"] = minmax_scaler.fit_transform(np.array(super_data_set["y1"]).reshape(-1, 1))
    super_data_set["y2"] = minmax_scaler.fit_transform(np.array(super_data_set["y2"]).reshape(-1, 1))

    # 保存一份源数据，为了将样本与股票代码互相匹配，以备推荐
    origin_super_data_set = pd.concat([stock_code_df, super_data_set], axis=1)

    x_train = super_data_set.ix[:, :2]
    y_train = super_data_set.ix[:, 2:]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train.values, y_train.values.ravel(), origin_super_data_set, ultra_origin_super_data_set


def model_op(x_train, y_train):
    # 只进行训练，得到权重即可
    # 预测要留到推荐的时候，使用上个季度训练得到的参数来预测
    # 验证要在推荐之后，使用推荐相关的评价标准来进行

    # score()是r2评价标准的值
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)
    feature_importances_rfr = rfr.feature_importances_
    # score_rfr = rfr.score(x_test, y_test)
    # result_rfr = rfr.predict(x_test)

    gbr = GradientBoostingRegressor()
    gbr.fit(x_train, y_train)
    feature_importances_gbr = gbr.feature_importances_
    # score_gbr = gbr.score(x_test, y_test)
    # result_gbr = gbr.predict(x_test)

    return feature_importances_rfr, feature_importances_gbr


def data_to_csv(feature_importances_rfr, feature_importances_gbr, origin_super_data_set, ultra_origin_super_data_set):
    feature_importances_rfr_df = pd.DataFrame(np.array(feature_importances_rfr).reshape(1, 2))
    feature_importances_gbr_df = pd.DataFrame(np.array(feature_importances_gbr).reshape(1, 2))

    feature_importances_rfr_df.to_csv(TRAIN_TEST_ROOT_PATH + DATA_PATH + "feature_importances_rfr" + ".csv",
                                      index=False, index_label=False)
    feature_importances_gbr_df.to_csv(TRAIN_TEST_ROOT_PATH + DATA_PATH + "feature_importances_gbr" + ".csv",
                                      index=False, index_label=False)
    origin_super_data_set.to_csv(TRAIN_TEST_ROOT_PATH + DATA_PATH + "origin_super_data_set" + ".csv",
                                 index=False, index_label=False)
    ultra_origin_super_data_set.to_csv(TRAIN_TEST_ROOT_PATH + DATA_PATH + "ultra_origin_super_data_set" + ".csv",
                                       index=False, index_label=False)


x_train, y_train, origin_super_data_set, ultra_origin_super_data_set = super_data_producer()
feature_importances_rfr, feature_importances_gbr = model_op(x_train, y_train)
data_to_csv(feature_importances_rfr, feature_importances_gbr, origin_super_data_set, ultra_origin_super_data_set)
print("weights ok!")
