# 将某一列转型，改变该列的dtype值。
# all_stock_df["market"] = all_stock_df["market"].astype("int64")

# Dataframe.loc[] 是按标签选取行列，也可以配合索引使用。
# 选取所有market列值为"主板"且list_status列值为"L"的行
# selected_stock_df = all_stock_df.loc[(all_stock_df["market"] == "主板") & (all_stock_df["list_status"] == "L")]
# test_df.loc[0, "close"] = 100

# fina_indicator_date_list是一个list，该方法能把列值为列表里的值的行都筛选出来
# selected_daily_df = daily_df.loc[daily_df["trade_date"].isin(fina_indicator_date_list)]

# 删除某列值等于某值的行，~是取反的意思
# fina_indicator_df = fina_indicator_df.loc[~(fina_indicator_df["ann_date"] == fina_indicator_date)]

# 删除某列值等于np.nan的行，~是取反的意思
# train_set_hybrid = train_set_hybrid.loc[~(train_set_hybrid[str(fea_name)].isna())]

# numpy的reshape，-1表示留空，第一位表示行数，第二位表示列数，下面的形式表示把数据变成一列
# data["label"].values.reshape(-1, 1)

# 按照列名删除dataframe的列,inplace=True的话，直接在原数据上修改，否则不会修改原数据
# df.drop(['B','C'],axis=1,inplace=True)

# dataframe行去重，subset表示考虑的列值，keep表示重复时保留第几个，inplace=True表示直接覆盖原dataframe
# daily_df.drop_duplicates(subset=["trade_date"], keep='first', inplace=True)

# dataframe按index排序，如果axis=1则是按列名排序
# train_df = slide_train_df.reset_index(drop=True).sort_index(ascending=False, axis=0)

# series转dataframe
# df = se.to_frame().T

# 得到列名列表，str型。
# df.keys()

# Dataframe根据元素内容找到索引列表
# index_list = df[(df.BoolCol == 3) & (df.attr == 22)].index.tolist()

# Dataframe拼接，axis=1表示列拼接，axis=0表示行拼接
# validate_set_hybrid = pd.concat([validate_set, validate_set_sequential_data_df], axis=1)

# dataframe反转（reverse）数据
# csv_temp = csv_temp.iloc[::-1]

# 取出除了["A", "B"]的其他列，不过列会被按照列名的字母顺序自动排序
# raw_train_data.loc[:, raw_train_data.columns.difference(["A", "B"])]

# 多个dataframe之间进行各种运算
# temp_super_data_set['Y_pred'] = temp_super_data_set.apply(lambda temp_super_data_set:
#                                                           temp_super_data_set["y1"] * feature_importances_rfr["0"] +
#                                                           temp_super_data_set["y2"] * feature_importances_rfr["1"],
#                                                           axis=1)

