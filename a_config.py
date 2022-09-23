# tickers
# --------------------------------------------------------------------------
tickers_dict = {

    "abev3": {
        
        "daily": {
            
            "ticker": "ABEV3.SA",
            "ylabel": "R$ - Brazil Real",
            "date_train_init": "2019-01-01",
            "date_train_end": "2022-08-31",
            "date_predict_init": "2022-09-01",
            "date_predict_end": "2022-09-30",
            "model_parameters": (3,1,9,1,1,0,5), #3,1,9,1,1,0,5
            "dependent_variable": 'close',
            "independent_variables": ['volume','high','low', 'open'],
            "freq": "D",
            "style_graph": "default",
            "color1": "royalblue",
            "color2": "goldenrod",
            "color3": "crimson",
            "color4": "black",
            "color5": "red",
            "p_value_accepted": 0.05,
            
            "dummy": {
               
                "covid": {
                    "type": "range",
                    "start": "2020-02-01",
                    "end": "2020-12-31"
                }
                
            }
            
        },
        
        "monthly": {
            
            "ticker": "ABEV3.SA",
            "ylabel": "R$ - Brazil Real",
            "date_train_init": "2010-01-01",
            "date_train_end": "2021-12-31",
            "date_predict_init": "2022-01-01",
            "date_predict_end": "2022-12-31",
            "model_parameters": (3,1,9,2,1,2,12), #3,1,9,2,1,2,12
            "dependent_variable": 'close',
            "independent_variables": ['volume', 'high', 'low', 'open'],
            "freq": "MS",
            "style_graph": "default",
            "color1": "royalblue",
            "color2": "goldenrod",
            "color3": "crimson",
            "color4": "black",
            "color5": "red",
            "p_value_accepted": 0.05,
            
            "dummy": {}
            
        }
        
    },

    "petr4": {
        
        "daily": {
            
            "ticker": "PETR4.SA",
            "ylabel": "R$ - Brazil Real",
            "date_train_init": "2015-01-01",
            "date_train_end": "2022-08-31",
            "date_predict_init": "2022-09-01",
            "date_predict_end": "2022-09-30",
            "model_parameters": (3,1,4,1,1,2,5),
            "dependent_variable": 'close',
            "independent_variables": ['volume'],
            "freq": "D",
            "style_graph": "default",
            "color1": "seagreen",
            "color2": "goldenrod",
            "color3": "crimson",
            "color4": "black",
            "color5": "red",
            "p_value_accepted": 0.05,
            
            "dummy": {
                
                "covid": {
                    "type": "range",
                    "start": "2020-02-01",
                    "end": "2020-12-31"
                }
                
            }
            
        },
        
        "monthly": {
            
            "ticker": "PETR4.SA",
            "ylabel": "R$ - Brazil Real",
            "date_train_init": "2015-01-01",
            "date_train_end": "2021-12-31",
            "date_predict_init": "2022-01-01",
            "date_predict_end": "2022-12-31",
            "model_parameters": (3,1,1,5,0,2,12),
            "dependent_variable": 'close',
            "independent_variables": ['volume'],
            "freq": "MS",
            "style_graph": "default",
            "color1": "seagreen",
            "color2": "goldenrod",
            "color3": "crimson",
            "color4": "black",
            "color5": "red",
            "p_value_accepted": 0.05,
            
            "dummy": {}
            
        }
        
    }

}

# --------------------------------------------------------------------------

# x13 arima path
path_x13_arima = "C:/Program Files (x86)/x12arima"
#path_x13_arima = "/home/x13as/"
