from warnings import filterwarnings
from shutil import rmtree
from os import mkdir
from os.path import exists
from pandas import concat, read_csv, to_datetime, DataFrame
from a_config import tickers_dict, path_x13_arima
from b_data_input import Data_input
from c_pred_indep_var import Arima_indep
from d_descriptive_statistics import Time_serie_level
from e_x13arima_seas_adjust import X13_arima_desaz
from f_stationarity import Stationarity_diff
from g_dummy import Dummy_generator
from h_model_execute import Model_execute


# suppress warnings - sorry about that =(
filterwarnings("ignore")

for folder in tickers_dict.keys(): 
    
    # folders
    if not exists("1_data"):
        mkdir("1_data")
    
    if not exists(f"1_data/{folder}"):
        mkdir(f"1_data/{folder}")    

    for period in tickers_dict[folder].keys():
        
        # variables
        ticker = tickers_dict[folder][period]['ticker']
        date_train_init = tickers_dict[folder][period]['date_train_init']
        date_train_end = tickers_dict[folder][period]['date_train_end']
        date_predict_init = tickers_dict[folder][period]['date_predict_init']
        date_predict_end = tickers_dict[folder][period]['date_predict_end']
        
        # folders
        if not exists(f"1_data/{folder}/{period}"):
            mkdir(f"1_data/{folder}/{period}")
        
        if exists(f"1_data/{folder}/{period}/data_base"):
            rmtree(f"1_data/{folder}/{period}/data_base")
            mkdir(f"1_data/{folder}/{period}/data_base")
        
        else:
            mkdir(f"1_data/{folder}/{period}/data_base")
        
        if exists(f"1_data/{folder}/{period}/results"):
            rmtree(f"1_data/{folder}/{period}/results")
            mkdir(f"1_data/{folder}/{period}/results")
        
        else:
            mkdir(f"1_data/{folder}/{period}/results")
        
        # independent variables forecast
        db_indep_fore = Data_input()
        
        data_all_fore = db_indep_fore.data_input_forecast(
            
            folder,
            period
            
        )
        
        data_all = data_all_fore[(data_all_fore.index <= date_train_end)]
        
        # auto arima model
        auto_arima = Arima_indep(
            
            data_all,
            tickers_dict[folder][period]['dependent_variable'],
            folder,
            period,
            date_predict_init,
            date_predict_end,
            tickers_dict[folder][period]['freq'],
            tickers_dict[folder][period]['p_value_accepted']
            
        )
        
        auto_arima.auto_arima_model(
            
            tickers_dict[folder][period]['model_parameters'][6]
            
        )
        
        #data model input
        db = Data_input()
        data_endog, data_exogs, variable, data_original, data_train = db.data_input(folder, period)
        
        # Time_serie_level (descriptive statistics)
        descriptive_statistics = Time_serie_level(
                
                data_endog,
                folder,
                period,
                tickers_dict[folder][period]['ylabel'],
                date_train_end,
                tickers_dict[folder][period]['style_graph'],
                tickers_dict[folder][period]['color1'],
                tickers_dict[folder][period]['color2'],
                tickers_dict[folder][period]['color3'],
                tickers_dict[folder][period]['color4'],
                tickers_dict[folder][period]['color5'],
            
        )
        
        descriptive_statistics.time_serie_plot()
        
        if tickers_dict[folder][period]['freq'] == 'MS':
            descriptive_statistics.moving_average_m()

        if tickers_dict[folder][period]['freq'] == 'D':
            descriptive_statistics.moving_average_d()

        descriptive_statistics.acf_pacf_plot()
        descriptive_statistics.periodogram_plot()
        descriptive_statistics.descriptive_stat()
        
        # seasonality
        # x13-arima-seats
        if tickers_dict[folder][period]['freq'] == 'MS':
            
            x13_desaz = X13_arima_desaz(
                
                data_endog, 
                data_exogs,
                folder,
                period,
                tickers_dict[folder][period]['ylabel'],
                path_x13_arima,
                tickers_dict[folder][period]['freq'],
                date_train_init,
                date_train_end,
                date_predict_end,
                tickers_dict[folder][period]['style_graph'],
                tickers_dict[folder][period]['color1'],
                tickers_dict[folder][period]['color2'],
                tickers_dict[folder][period]['color3'],
                tickers_dict[folder][period]['color4'],
                tickers_dict[folder][period]['color5'],
                
            )
            
            x13_desaz.x13_results()
            x13_desaz.x13_seasonal_adjustment()
            x13_desaz.independent_desaz_x13()
        
        else:
            
            data_d = concat([data_endog, data_exogs], axis=1)
            
            data_d.to_csv(
                
                f"1_data/{folder}/{period}/data_base/{folder}_seasonal_adjustment_{period}.csv"
                
            )
        
        # stationarity
        try:
            data_non_seasonal = read_csv(
                
                f"1_data/{folder}/{period}/data_base/{folder}_seasonal_adjustment_{period}.csv",
                sep=",",
                decimal="."
                
            )
            
            data_non_seasonal["index_date"] = to_datetime(data_non_seasonal["index_date"])
            data_non_seasonal = data_non_seasonal.sort_values("index_date")
            data_non_seasonal = data_non_seasonal.set_index("index_date")
        
        except Exception as erro:
            print(erro)
            exit()
        
        stationarity = Stationarity_diff(
            
            data_non_seasonal,
            folder,
            period,
            tickers_dict[folder][period]['p_value_accepted']
            
        )
        
        stationarity.adf_teste()
        stationarity.diff_data()
        stationarity.independent_var_stationarity()
        
        # model execute 
        # open stationary data
        try:
            data_stationarity = read_csv(
                
                f"1_data/{folder}/{period}/data_base/{folder}_stationary_{period}.csv",
                sep=",",
                decimal="."
             
            )
            
            data_stationarity["index_date"] = to_datetime(data_stationarity["index_date"])
            data_stationarity = data_stationarity.sort_values("index_date")
            data_stationarity = data_stationarity.set_index("index_date")
            
            data_stationarity = data_stationarity[
                
                (data_stationarity.index >= date_train_init) & 
                (data_stationarity.index <= date_predict_end)
                
            ]
        
        except Exception as erro:
            print(erro)
            exit()
        
        
        # endogenous
        data_non_seasonal_endog = read_csv(
            
            f"1_data/{folder}/{period}/data_base/{folder}_seasonal_adjustment_{period}.csv", 
            sep=",",
            decimal="."
            
        )
        
        data_non_seasonal_endog["index_date"] = to_datetime(data_non_seasonal_endog["index_date"])
        data_non_seasonal_endog = data_non_seasonal_endog.sort_values("index_date")
        data_non_seasonal_endog = data_non_seasonal_endog.set_index("index_date")
        
        data_non_seasonal_endog = data_non_seasonal_endog[
            
            (data_non_seasonal_endog.index >= date_train_init) &
            (data_non_seasonal_endog.index <= date_predict_end)
            
        ]
        
        data_dummy = DataFrame()
        
        # dummy variable
        for dummy in tickers_dict[folder][period]['dummy'].keys():
            
            if tickers_dict[folder][period]['dummy'][dummy]['type'] == 'range':
                
                dm_range = Dummy_generator(
                    
                    tickers_dict[folder][period]['date_train_init'],
                    tickers_dict[folder][period]['date_predict_end'],
                    tickers_dict[folder][period]['freq']
                    
                )
                
                data_dummy[dummy] = dm_range.dummy_generator_range(
                    
                    dummy,
                    tickers_dict[folder][period]['dummy'][dummy]['start'],
                    tickers_dict[folder][period]['dummy'][dummy]['end']
                    
                )
        
        # data frame model
        try:
            
            data_model = concat(
                
                [
                    
                    data_non_seasonal_endog.iloc[ : , 0 ],
                    data_stationarity.iloc[ : , 1: ], 
                    data_dummy
                    
                ], 
                
                axis=1
                
            )
        
        except:
            
            data_model = concat(
                
                [
                    
                    data_non_seasonal_endog.iloc[ : , 0 ],
                    data_stationarity.iloc[ : , 1: ]
                    
                ], 
                
                axis=1
                
            )
        
        data_model = data_model[ (data_model.index <= date_train_end) ].dropna()
        
        # final data frame for the forecast
        data_exogs_fore = concat(
            
            [
                
                data_stationarity[(data_stationarity.index > date_train_end)],
                data_dummy[(data_dummy.index > date_train_end)]
                
            ],
            
            axis=1
            
        )
        
        # model execute
        model = Model_execute(
            
            data_original,
            data_model,
            data_exogs_fore,
            folder,
            period,
            tickers_dict[folder][period]['ylabel'],
            tickers_dict[folder][period]['style_graph'],
            tickers_dict[folder][period]['color1'],
            tickers_dict[folder][period]['color2'],
            tickers_dict[folder][period]['color3'],
            tickers_dict[folder][period]['color4'],
            tickers_dict[folder][period]['color5'],
            
        ) 
        
        model.model_execute(
            
            tickers_dict[folder][period]['model_parameters'][0],
            tickers_dict[folder][period]['model_parameters'][1],
            tickers_dict[folder][period]['model_parameters'][2],
            tickers_dict[folder][period]['model_parameters'][3],
            tickers_dict[folder][period]['model_parameters'][4],
            tickers_dict[folder][period]['model_parameters'][5],
            tickers_dict[folder][period]['model_parameters'][6],
            
        )
        
        model.ts_residuals_plot()
        model.dist_residual_analysis()
        model.acf_pacf_residuals()
        
        if tickers_dict[folder][period]['freq'] == 'MS':
            model.adjust_predict_m(date_predict_init)
        
        if tickers_dict[folder][period]['freq'] == 'D':
            model.adjust_predict_d(date_predict_init)

