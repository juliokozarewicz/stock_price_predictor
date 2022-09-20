from pandas import read_csv, DataFrame, date_range, to_datetime
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX


class Arima_indep:
    """
    Study of data stationarity.
    
    Required settings:
        - data (independent variables)
        - dep_variable (formatted dependent variable)
        - indep_variable (formatted independent variables)
        - period
        - date_predict_init
        - date_predict_end
        - frequency
        - p_value_accepted (p-value number accepted)
    
    """

    def __init__(
        
        self,
        data,
        dep_variable,
        indep_variables,
        period,
        date_predict_init,
        date_predict_end,
        freq,
        p_value_accepted = 0.05
        ):
        """
        Settings for the outputs.
        """
        
        # config
        self.data_all = data
        self.dep_var = dep_variable
        self.folder = indep_variables
        self.period = period
        self.p_value_accepted = p_value_accepted
        self.date_predict_init = date_predict_init
        self.date_predict_end = date_predict_end
        self.freq = freq
        self.folder_ = indep_variables.replace(" ", "_").lower()


    def auto_arima_model(self, s):
        """
        Function that uses auto arima to find the best parameters for the model, 
        s is an integer giving the periodicity (number of periods in season), 
        often it is 4 for quarterly data or 12 for monthly data.
        """
        
        df_exog_pred = DataFrame()
        
        for col in self.data_all.columns.to_list():
            
            # Best model with auto arima
            model_select = auto_arima(
                
                self.data_all[col],
                information_criterion = 'aic',
                seasonal = True,
                error_action = "ignore",
                supress_warnings = True,
                trace = False,
                m = s,
                start_p = 1,
                start_q = 1,
                start_P = 1,
                start_Q = 1,
                
            )
            
            model_select = str(model_select)
            
            p = int(model_select[7])
            q = int(model_select[9])
            d = int(model_select[11])
            P = int(model_select[14])
            D = int(model_select[16])
            Q = int(model_select[18])
            
            # model
            model = SARIMAX(
                
                self.data_all[col],
                order = (p, d, q), 
                seasonal_order = (P, D, Q, s), 
                trend = "c"
                
            )
            
            self.model_fit = model.fit(disp=False)
            
            model_result = self.model_fit.summary()
            
            forecast_number = date_range(
                
                start = self.date_predict_init,
                end = self.date_predict_end,
                freq = self.freq)
            
            forecast = self.model_fit.get_forecast(len(forecast_number))
            
            predict = forecast.predicted_mean
            
            df_exog_pred[col] = DataFrame(predict)
        
        df_exog_pred['index_date'] = to_datetime(forecast_number)
        
        df_exog_pred = df_exog_pred.sort_values("index_date")
        
        df_exog_pred = df_exog_pred.set_index('index_date')
        
        df_exog_pred[self.dep_var] = ''
        
        df_exog_pred = df_exog_pred[ df_exog_pred.index <= self.date_predict_end ]
        
        df_exog_pred.to_csv(
            
            f"1_data/{self.folder}/{self.period}/data_base/{self.folder}_fpred.csv",
            sep = ",",
            decimal = ".",
            index_label = "index_date"
            
        )
        
        return
