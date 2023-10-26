import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class DataTransformer :
    """
    Variables:
    need_normalized -> bool, if data needs to be normalized
    scale_type -> minmax, standard otherwise
    R_dis_dimmension -> the number of discrete columns
    separate_num -> the separate position of continuous and discrete columns


    raw_data : [5,A3,7]
    ordered_R : [0.5, [0,0,1,0], 0.7]
    separate_data : [[0,0,1,0], 0.5, 0.7]
    unprocessed_data : [[0.001,-0.9,1.02,0.2],0.47,0.73]


    Methods:
    _fit_discrete -> fit discrete columns with sklearn.OneHot
    _fit_continuous -> fit continuous columns with sklearn.MinMaxScaler or sklearn.StandardScaler
    fit -> fit given raw_data

    transform -> raw_data transforms to ordered_R
    inverse_transform -> ordered_R transforms to raw_data

    separate_continuous_discrete_columns -> ordered_R to continuous data and discrete data
    take_discrete_continuous_together -> unprocessed_data to separate_data
    separate_to_ordered_R -> separate_data to ordered_R

    // TODO: process the unprocessed_data should be pulled out to another method
    """
    class ColumnTransformerInfo():
        def __init__(self, transformer, column_name, dim):
            self.transformer = transformer
            self.column_name = column_name
            self.dim = dim

    def __init__(self, discrete_columns, need_normalized=True, scale_type="minmax"):
        self.columnTransformerInfoList = []
        self.R_dis_dimmension = []
        self.discrete_columns = discrete_columns
        self.need_normalized = need_normalized
        self.scale_type = scale_type
        self.separate_num = 0

    def _fit_discrete(self, raw_data, column_name):
        cur_transformer = OneHotEncoder()
        cur_transformer.fit(raw_data[[column_name]])
        cur_transformer_info = DataTransformer.ColumnTransformerInfo(cur_transformer, column_name,
                                                                     cur_transformer.categories_[0].shape[0])
        return cur_transformer_info

    def _fit_continuous(self, raw_data, column_name):
        if self.scale_type == "minmax":
            cur_transformer = MinMaxScaler()
        else:
            cur_transformer = StandardScaler()
        cur_transformer.fit(raw_data[[column_name]])
        cur_transformer_info = DataTransformer.ColumnTransformerInfo(cur_transformer, column_name, 1)
        return cur_transformer_info

    def fit(self, raw_data):
        cur_discrete_dim = 0
        for column in raw_data.columns:
            if column in self.discrete_columns:
                cur_column_transformer_info = self._fit_discrete(raw_data, column)
                cur_discrete_dim += cur_column_transformer_info.dim
                self.R_dis_dimmension.append(cur_column_transformer_info.dim)
            else:
                cur_column_transformer_info = self._fit_continuous(raw_data, column)
            self.columnTransformerInfoList.append(cur_column_transformer_info)
        self.separate_num = cur_discrete_dim

    def _transform_discrete(self, raw_data, transformer_info):
        column_name = transformer_info.column_name
        cur_data = transformer_info.transformer.transform(raw_data[[column_name]]).todense().A
        return cur_data

    def _transform_continuous(self, raw_data, transformer_info):
        column_name = transformer_info.column_name
        cur_data = transformer_info.transformer.transform(raw_data[[column_name]])
        if self.need_normalized is False:
            cur_data = raw_data[[column_name]].to_numpy()
        return cur_data

    def transform(self, raw_data):
        Routput = []
        for transformer_info in self.columnTransformerInfoList:
            if transformer_info.column_name in self.discrete_columns:
                output = self._transform_discrete(raw_data, transformer_info)
            else:
                output = self._transform_continuous(raw_data, transformer_info)
            Routput.append(output)
        R = np.concatenate(Routput, axis=1).astype(float)
        return R

    def _inverse_transform_continuous(self, cur_data, columnTransformerInfo):
        if self.scale_type == "minmax":
            cur_data = np.clip(cur_data, 0, 1)
        ans = columnTransformerInfo.transformer.inverse_transform(cur_data)
        if self.need_normalized is False:
            ans = cur_data
        return ans

    def _inverse_transform_discrete(self, cur_data, columnTransformerInfo):
        real_onehot_data = np.eye(columnTransformerInfo.dim)[np.argmax(cur_data,axis=1)]
        ans = columnTransformerInfo.transformer.inverse_transform(real_onehot_data)
        return ans

    def inverse_transform(self, raw_R, continuous_position=None):
        recovered_data_list = []
        recovered_data_columns = []
        if continuous_position is None:
            start = 0
            for transformer_info in self.columnTransformerInfoList:
                cur_r_data = raw_R[:, start: start + transformer_info.dim]
                if transformer_info.column_name in self.discrete_columns:
                    recovered_data = self._inverse_transform_discrete(cur_r_data, transformer_info)
                else:
                    recovered_data = self._inverse_transform_continuous(cur_r_data, transformer_info)
                recovered_data_list.append(recovered_data)
                recovered_data_columns.append(transformer_info.column_name)
                start += transformer_info.dim

        else:
            start_dis = 0
            start_con = continuous_position
            for transformer_info in self.columnTransformerInfoList:
                if transformer_info.column_name in self.discrete_columns:
                    cur_r_data = raw_R[:, start_dis: start_dis + transformer_info.dim]
                    recovered_data = self._inverse_transform_discrete(cur_r_data, transformer_info)
                    start_dis += transformer_info.dim
                else:
                    cur_r_data = raw_R[:, start_con: start_con + transformer_info.dim]
                    recovered_data = self._inverse_transform_continuous(cur_r_data, transformer_info)
                    start_con += transformer_info.dim
                recovered_data_list.append(recovered_data)
                recovered_data_columns.append(transformer_info.column_name)
        final_data = np.column_stack(recovered_data_list)
        final_data = (pd.DataFrame(final_data, columns=recovered_data_columns))
        return final_data


    def is_discrete(self, column_name):
        if column_name in self.discrete_columns:
            return True
        else:
            return False

    def separate_continuous_discrete_columns(self, r):
        continuous_data_list = []
        discrete_data_list = []
        start = 0
        for transformer_info in self.columnTransformerInfoList:
            cur_data = r[:, start: start + transformer_info.dim]
            if transformer_info.column_name in self.discrete_columns:
                discrete_data_list.append(cur_data)
            else:
                continuous_data_list.append(cur_data)
            start += transformer_info.dim
        if len(discrete_data_list) == 0:
            final_discrete_data = np.array([])
        else:
            final_discrete_data = np.column_stack(discrete_data_list)
        # final_discrete_data = pd.DataFrame(final_discrete_data, columns=discrete_columns_list)
        if len(continuous_data_list) == 0:
            final_continuous_data = np.array([])
        else:
            final_continuous_data = np.column_stack(continuous_data_list)
        # final_continuous_data = pd.DataFrame(final_continuous_data, columns=continuous_columns_list)
        return final_discrete_data, final_continuous_data

    def take_discrete_continuous_together(self, data_discrete, data_continuous):
        if len(self.discrete_columns) == len(self.columnTransformerInfoList):
            return data_discrete
        final_data = []
        start = 0
        for columnTransformerInfo in self.columnTransformerInfoList:
            if columnTransformerInfo.column_name in self.discrete_columns:
                cur_data = data_discrete[:, start:start + columnTransformerInfo.dim]
                real_onehot_data = np.eye(columnTransformerInfo.dim)[np.argmax(cur_data, axis=1)]
                final_data.append(real_onehot_data)
                start += columnTransformerInfo.dim
        data_continuous = np.clip(data_continuous,0,1)
        final_data.append(data_continuous)
        final_data = np.concatenate(final_data, axis=1).astype(float)
        return final_data

    def separate_to_ordered_R(self, separate_data):
        r = []
        start_discrete = 0
        start_continuous = self.separate_num
        for transform_info in self.columnTransformerInfoList:
            if transform_info.column_name in self.discrete_columns:
                cur_data = separate_data[:, start_discrete:start_discrete + transform_info.dim]
                r.append(cur_data)
                start_discrete += transform_info.dim
            else:
                cur_data = separate_data[:, start_continuous :start_continuous + transform_info.dim]
                r.append(cur_data)
                start_continuous += transform_info.dim
        r = np.concatenate(r, axis=1).astype(float)
        return r

    def locating_the_dis_column(self, given_location):
        # return the column start and end of the given location
        start = 0
        for i in range(len(self.R_dis_dimmension)):
            if given_location < start + self.R_dis_dimmension[i]:
                return start, start + self.R_dis_dimmension[i]
            else:
                start += self.R_dis_dimmension[i]
        return None, None