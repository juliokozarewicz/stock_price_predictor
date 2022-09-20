from pandas import to_datetime, read_csv, concat
from pandas_datareader import data
from a_config import tickers_dict


class Data_input:
    """
    Class responsible for inputting data to the model.
    """

    def data_input_forecast(self, folder, period):
        """
        Data for the prediction of independent variables
        """
        
        # variables
        ticker = tickers_dict[folder][period]['ticker']
        date_train_init = tickers_dict[folder][period]['date_train_init']
        date_train_end = tickers_dict[folder][period]['date_train_end']
        date_predict_init = tickers_dict[folder][period]['date_predict_init']
        date_predict_end = tickers_dict[folder][period]['date_predict_end']
        
        # sample
        if tickers_dict[folder][period]['freq'] == 'D':
            
            data_entry = data.DataReader(
                
                ticker, 
                start = date_train_init, 
                end = date_predict_end, 
                data_source = 'yahoo'
                
            )
        
        if tickers_dict[folder][period]['freq'] == 'MS':
            
            data_entry = data.get_data_yahoo(
                
                ticker, 
                start = date_train_init, 
                end = date_predict_end, 
                interval = 'm'
                
            )
        
        data_entry = data_entry.rename(
            
            columns = {
                
                'High' : 'high',
                'Low': 'low',
                'Open' : 'open',
                'Close' : 'close',
                'Volume' : 'volume',
                'Adj Close' : 'adj close'
                
            }
            
        )
        
        data_entry.index = data_entry.index.rename('index_date')
        
        data_entry = data_entry.reindex(
            
            columns = [
                
                'close',
                'volume',
                'open',
                'high',
                'low',
                'adj close'
                
            ]
            
        )
        
        data_entry.index = to_datetime(data_entry.index)
        
        # filter for variables
        data_all_fore = data_entry 
        
        dep_var = [tickers_dict[folder][period]['dependent_variable']]
        indep_var = tickers_dict[folder][period]['independent_variables']
        
        vars_slice = dep_var + indep_var
       
        data_all_fore = data_all_fore.loc[ : , vars_slice]
        
        return data_all_fore


    def data_input(self, folder, period):
        """
        Data input
        """
        
        # variables
        ticker = tickers_dict[folder][period]['ticker']
        date_train_init = tickers_dict[folder][period]['date_train_init']
        date_train_end = tickers_dict[folder][period]['date_train_end']
        date_predict_init = tickers_dict[folder][period]['date_predict_init']
        date_predict_end = tickers_dict[folder][period]['date_predict_end']
        
        # sample
        if tickers_dict[folder][period]['freq'] == 'D':
            
            data_entry = data.DataReader(
                
                ticker, 
                start = date_train_init, 
                end = date_predict_end, 
                data_source = 'yahoo'
                
            )
        
        if tickers_dict[folder][period]['freq'] == 'MS':
            
            data_entry = data.get_data_yahoo(
                
                ticker, 
                start = date_train_init, 
                end = date_predict_end, 
                interval = 'm'
                
            )
        
        data_entry = data_entry.rename(
            
            columns = {
                
                'High' : 'high',
                'Low': 'low',
                'Open' : 'open',
                'Close' : 'close',
                'Volume' : 'volume',
                'Adj Close' : 'adj close'
                
            }
            
        )
        
        data_entry.index = data_entry.index.rename('index_date')
        
        data_entry = data_entry.reindex(
            
            columns = [
                
                'close',
                'volume',
                'open',
                'high',
                'low',
                'adj close'
                
            ]
            
        )
        
        data_entry.index = to_datetime(data_entry.index)
        
        data_f_pred = read_csv(
            
            f"1_data/{folder}/{period}/data_base/{folder}_fpred.csv",
            sep=",",
            decimal="."
            
        )
        
        data_f_pred = data_f_pred.reindex(
            
            columns = [
                
                'index_date',
                'close',
                'volume',
                'open',
                'high',
                'low',
                'adj close'
                
            ]
            
        )
        
        data_f_pred["index_date"] = to_datetime(data_f_pred["index_date"])
        data_f_pred = data_f_pred.sort_values("index_date")
        data_f_pred = data_f_pred.set_index("index_date")
        
        data_entry = data_entry[ 
            
            (data_entry.index >= date_train_init) & 
            (data_entry.index < date_predict_init)
            
        ]
        
        data_entry = concat([data_entry, data_f_pred])
        
        # sample filter
        if tickers_dict[folder][period]['freq'] == 'D':
            
            data_original = data.DataReader(
                
                ticker, 
                start = date_train_init, 
                end = date_predict_end, 
                data_source = 'yahoo'
                
            )
        
        if tickers_dict[folder][period]['freq'] == 'MS':
            
            data_original = data.get_data_yahoo(
                
                ticker, 
                start = date_train_init, 
                end = date_predict_end, 
                interval = 'm'
                
            )
        
        data_original = data_original.rename(
            
            columns = {
                
                'High' : 'high',
                'Low': 'low',
                'Open' : 'open',
                'Close' : 'close',
                'Volume' : 'volume',
                'Adj Close' : 'adj close'
                
            }
            
        )
        
        data_original.index = data_original.index.rename('index_date')
        
        data_original = data_original.reindex(
            
            columns = [
                
                'close',
                'volume',
                'open',
                'high',
                'low',
                'adj close'
                
            ]
            
        )
        
        data_original.index = to_datetime(data_original.index)
        data_original = data_original[data_original.index >= date_train_init].iloc[ : , 0 ]
        
        # filter for variables
        data_train = data_entry[ 
            
            (data_entry.index >= date_train_init) & 
            (data_entry.index <= date_predict_end)
            
        ]
        
        # variables
        data_endog = data_train.iloc[ : , 0 : 1 ]
        
        data_exogs = data_train.iloc[ : , 1 :   ][
            
            tickers_dict[folder][period]['independent_variables']
            
        ]
        
        # variable name
        variable_ = list(data_endog.columns.values.tolist())[0]
        variable = variable_.replace("_", ' ').upper()
        
        return (data_endog, data_exogs, variable.lower(), data_original, data_train)
