from pandas import DataFrame, read_csv, concat, to_datetime
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
from matplotlib.dates import DateFormatter
from matplotlib.pyplot import fill_between
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import r2_score
from numpy import arange


class Model_execute:
    """
    Class responsible for estimating the model.
    
    Required settings:
    - data_original (original dependent folder dataset)
    - data (input data)
    - data_exogs_fore (data for test or forecast)
    - folder (formatted dependent folder - "NAME VARIABLE")
    - period (string)
    - ylabel (string)
    
    Optional settings:
    - style (graphic style)
    - color1 (color setting)
    - color2 (color setting)
    - color3 (color setting)
    - color4 (color setting)
    - color5 (color setting)
    
    """

    def __init__(
        
        self,
        data_original,
        data,
        data_exogs_fore,
        folder,
        period,
        ylabel,
        style_graph="seaborn", 
        color1="royalblue", 
        color2="crimson", 
        color3="darkorange", 
        color4="black", 
        color5="red"
        
        ):
        """
        Settings for the outputs.
        """
        
        # data frame
        self.data_endog = data.iloc[ : , 0 : 1 ]
        self.data_exogs = data.iloc[ : , 1 :   ]
        self.data_exogs_fore = data_exogs_fore.iloc[ : , 1 :   ]
        self.data_original = data_original
        
        # configs
        self.folder = folder
        self.period = period
        self.ylabel = ylabel
        
        # style
        self.style_graph = style_graph
        self.color1 = color1
        self.color2 = color2
        self.color3 = color3
        self.color4 = color4
        self.color5 = color5

    def model_execute(self, p, d, q, P, D, Q, s):
        """
        Model estimation.
        
        p = Order of the AR term
        q = Order of the MA term
        d = Number of differencing required to make the time series stationary
        P = Seasonal order of the AR term
        D = Seasonal order of the MA term
        Q = Seasonal difference number
        """
        
        self.model = SARIMAX(
            
            endog=self.data_endog,
            exog=self.data_exogs,
            order=(p, d, q), 
            seasonal_order=(P, D, Q, s),
            trend="c"
            
        )
        
        # parameters
        self.D_term = D 
        self.s_term = s
        
        # model
        self.model_fit = self.model.fit(disp=False) 
        self.resid = DataFrame(self.model_fit.resid, columns=[f"{self.folder}"])
        self.resid = self.resid.iloc[ ( self.D_term * self.s_term ) + 1 : , : ]
        
        model_result = self.model_fit.summary()
        
        with open(f"1_data/{self.folder}/{self.period}/results/{self.folder}_9_model_summary_{self.period}.txt", 'w') as desc_stat:
            desc_stat.write(str(model_result))
        
        return


    def ts_residuals_plot(self):
        """
        Residuals time serie plot.
        """
        
        # style
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=( 12 , 6), dpi=300)
        
        # set
        plt.title(f"RESIDUALS - {self.period.upper()} {self.folder.upper()}")
        
        # plot
        ts_plot = plt.plot(
            
            self.resid,
            linestyle="solid",
            color=self.color2, 
            linewidth = 2
            
        )
        
        # plot config
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False) 
        plt.gcf().autofmt_xdate() # year
        date_format = mpl_dates.DateFormatter('%b. %Y') # month, year
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.xlabel("")
        
        plt.tight_layout()
        
        #save
        plt.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_10_residuals_(time_serie)_{self.period}.jpg")
        
        return 


    def dist_residual_analysis(self):
        """
        Analysis of model residuals.
        """
        
        fig, ax = plt.subplots(1, 1)
        plt.style.use(self.style_graph)
        plt.rcParams["figure.figsize"] = [12, 6]
        plt.rcParams["figure.dpi"] = 300
        
        resid_plot_fd = self.resid.plot(
            
            color=self.color2,
            kind='hist',
            legend=False
            
        )
        
        plt.title(f"RESIDUALS - {self.period.upper()} {self.folder.upper()}")
        
        # config
        plt.grid(False)
        resid_plot_fd.spines['top'].set_visible(False)
        resid_plot_fd.spines['right'].set_visible(False)
        resid_plot_fd.spines['bottom'].set_visible(False)
        resid_plot_fd.spines['left'].set_visible(False)
        
        plt.tight_layout()
        
        # save
        plt.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_11_residuals_(frequency_distribution)_{self.period}.jpg")
        
        return


    def acf_pacf_residuals(self):
        """
        Residuals ACF and PACF.
        """
        
        fig, ax = plt.subplots(2, 1, sharex=False, figsize=( 12 , 6), dpi=300)
        
        acf = plot_acf(
            
            self.resid.values.squeeze(),
            title = f"ACF (RESIDUALS) - {self.period.upper()} {self.folder.upper()}",
            color = self.color1,
            vlines_kwargs = {"colors": self.color1},
            use_vlines = True,
            alpha = 0.02,
            ax = ax[0],
            zero = False
            
        )
        
        pacf = plot_pacf(
            
            self.resid.values.squeeze(),
            title = f"PACF (RESIDUALS) - {self.period.upper()} {self.folder.upper()}",
            color = self.color2,
            vlines_kwargs = {"colors": self.color2},
            use_vlines = True,
            alpha = 0.02,
            ax = ax[1],
            zero = False
            
        )
        
        # config
        ax[0].set_ylim(-0.5, 0.5) 
        ax[1].set_ylim(-0.5, 0.5)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['left'].set_visible(False) 
        ax[0].get_xaxis().set_ticks([])
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False) 
        
        plt.tight_layout()
        
        # save
        plt.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_12_residuals_(acf_and_pacf)_{self.period}.jpg")
        
        return
    
    
    def adjust_predict_m(self, date_predict_init):
        """
        Observed x fitted + predict plot.
        """
        
        # *** fit model ***
        init_fitted = ( self.D_term * self.s_term ) + 1
        self.data_endog[f"{self.folder}_fitted"] = self.model_fit.predict(start=init_fitted, dynamic=False)
        
        # *** r-squared ***
        r2_fit = init_fitted - len(self.data_endog)
        
        r2 = r2_score(
            
            self.data_endog.iloc[ r2_fit : -1 , 0 ],
            self.data_endog.iloc[ r2_fit : -1 , 1 ]
            
        )
        
        r2 = r2 * 100
        
        # *** forecast ***
        self.data_exogs_fore = self.data_exogs_fore.fillna(0)
        
        predict = self.model_fit.get_prediction(
            
            start=len(self.data_endog) - 1, 
            end=len(self.data_endog) + (len(self.data_exogs_fore) - 1), 
            exog=self.data_exogs_fore
            
        )
        
        predict_mean = predict.summary_frame()["mean"]
        index_newdf = DataFrame(self.data_exogs_fore.index)
        predict_mean = DataFrame(predict_mean.iloc[0:-1].values)
        df_data_fore = concat([index_newdf, predict_mean], axis=1)
        df_data_fore["index_date"] = to_datetime(df_data_fore["index_date"])
        df_data_fore = df_data_fore.set_index('index_date')
        
        # standard error
        std_error = predict.summary_frame()['mean_se'].iloc[ 1: , ].values
        std_error = DataFrame(std_error)
        std_error = concat([index_newdf, std_error], axis=1)
        std_error["index_date"] = to_datetime(std_error["index_date"])
        std_error = std_error.set_index('index_date')
        
        # *** confidence interval ***
        conf_95 = predict.conf_int(alpha=0.05)
        conf_95 = DataFrame(conf_95.values)
        conf_95 = concat([index_newdf, conf_95], axis=1)
        conf_95["index_date"] = to_datetime(conf_95["index_date"])
        conf_95 = conf_95.set_index('index_date')
        
        conf_50 = predict.conf_int(alpha=0.5)
        conf_50 = DataFrame(conf_50.values)
        conf_50 = concat([index_newdf, conf_50], axis=1)
        conf_50["index_date"] = to_datetime(conf_50["index_date"])
        conf_50 = conf_50.set_index('index_date')
        
        # concat
        df_data_all = concat(
            
            [
                
                self.data_endog,
                df_data_fore,
                std_error,
                conf_95
                
            ],
            
            axis = 1
            
        )
        
        df_data_all.columns = [
            
            f"{self.folder}_observed",
            f"{self.folder}_fitted",
            f"{self.folder}_predicted", 
            f"std_error",
            f"ci_95_lower", 
            f"ci_95_upper"
            
        ]
        
        # save dataframe
        df_data_all = df_data_all.iloc[ 1: , : ]
        
        df_data_all.to_csv(
            
            f"1_data/{self.folder}/{self.period}/data_base/{self.folder}_observed_fitted_predicted_{self.period}.csv",
            sep=",", 
            index_label="index_date"
            
        )
        
        # plot config
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12 , 6), dpi=300)
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        
        # fitted
        fitted = df_data_all[f"{self.folder}_fitted"]
        fitted[fitted.index == date_predict_init] = df_data_all[df_data_all.index == date_predict_init][f"{self.folder}_predicted"]
        plt.plot(fitted, color=self.color2)
        
        # forecast
        plt.plot(df_data_all[f"{self.folder}_predicted"], color=self.color3)
        
        # plot original
        observed = self.data_original.plot(
            
            title=f"FORECAST - {self.period.upper()} {self.folder.upper()}",
            xlabel="",
            ylabel="",
            color=self.color1,
            figsize=(12, 6)
            
        )
        
        # plot confidence interval
        predict_conf_95 = fill_between(
            
            conf_95.index,
            conf_95.iloc[ : , 0 ],
            conf_95.iloc[ : , 1 ],
            color=self.color4,
            alpha=0.05
            
        )
        
        predict_conf_50 = fill_between(
            
            conf_50.index,
            conf_50.iloc[ : , 0 ],
            conf_50.iloc[ : , 1 ],
            color=self.color4,
            alpha=0.1
            
        )
        
        # plot legends
        plt.legend(
            
            [
                
                f"fitted model (R² = {r2:.2f}%)",
                f"forecast",
                f"observed",
                f"conf. int. 95%", 
                f"conf. int. 50%"
                
            ],
            
            frameon = False
            
        )
        
        # config
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.gcf().autofmt_xdate()
        date_format = mpl_dates.DateFormatter('%b. %Y')
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.xlabel("")
        plt.ylabel(self.ylabel)
        plt.tight_layout()
        
        # save fig
        plt.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_13_observed_fitted_predict_{self.period}.jpg")
        
        return


    def adjust_predict_d(self, date_predict_init):
        """
        Observed x fitted + predict plot.
        """
        
        # plot config
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12 , 6), dpi=300)
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        
        # *** fit model ***
        init_fitted = ( self.D_term * self.s_term ) + 1
        self.data_endog[f"{self.folder}_fitted"] = self.model_fit.predict(start=init_fitted, dynamic=False)
        
        # *** r-squared ***
        r2_fit = init_fitted - len(self.data_endog)
        
        r2 = r2_score(
            
            self.data_endog.iloc[ r2_fit : -1 , 0 ],
            self.data_endog.iloc[ r2_fit : -1 , 1 ]
            
        )
        
        r2 = r2 * 100
        
        # *** forecast ***
        self.data_exogs_fore = self.data_exogs_fore.fillna(0)
        
        predict = self.model_fit.get_prediction(
            
            start=len(self.data_endog) - 1, 
            end=len(self.data_endog) + (len(self.data_exogs_fore) - 1), 
            exog=self.data_exogs_fore
            
        )
        
        predict_mean = predict.summary_frame()["mean"]
        index_newdf = DataFrame(self.data_exogs_fore.index)
        predict_mean = DataFrame(predict_mean.iloc[0:-1].values)
        df_data_fore = concat([index_newdf, predict_mean], axis=1)
        df_data_fore["index_date"] = to_datetime(df_data_fore["index_date"])
        df_data_fore = df_data_fore.set_index('index_date')
        
        # standard error
        std_error = predict.summary_frame()['mean_se'].iloc[ 1: , ].values
        std_error = DataFrame(std_error)
        std_error = concat([index_newdf, std_error], axis=1)
        std_error["index_date"] = to_datetime(std_error["index_date"])
        std_error = std_error.set_index('index_date')
        
        # *** confidence interval ***
        conf_95 = predict.conf_int(alpha=0.05)
        conf_95 = DataFrame(conf_95.values)
        conf_95 = concat([index_newdf, conf_95], axis=1)
        conf_95["index_date"] = to_datetime(conf_95["index_date"])
        conf_95 = conf_95.set_index('index_date')
        
        conf_50 = predict.conf_int(alpha=0.5)
        conf_50 = DataFrame(conf_50.values)
        conf_50 = concat([index_newdf, conf_50], axis=1)
        conf_50["index_date"] = to_datetime(conf_50["index_date"])
        conf_50 = conf_50.set_index('index_date')
        
        # concat
        df_data_all = concat(
            
            [
                
                self.data_endog,
                df_data_fore,
                std_error,
                conf_95
                
            ],
            
            axis = 1
            
        )
        
        df_data_all.columns = [
            
            f"{self.folder}_observed",
            f"{self.folder}_fitted",
            f"{self.folder}_predicted", 
            f"std_error",
            f"ci_95_lower", 
            f"ci_95_upper"
            
        ]
        
        # save dataframe
        df_data_all = df_data_all.iloc[ 1: , : ]
        
        df_data_all.to_csv(
            
            f"1_data/{self.folder}/{self.period}/data_base/{self.folder}_observed_fitted_predicted_{self.period}.csv",
            sep=",", 
            index_label="index_date"
            
        )
        
        # plot original
        num_plot = int( ( len(self.data_original ) // 6) * -1)
        self.data_original = self.data_original.iloc[ num_plot: ]
        
        observed = self.data_original.iloc[ num_plot: ].plot(
            
            title=f"FORECAST - {self.period.upper()} {self.folder.upper()}",
            xlabel="",
            ylabel="",
            color=self.color1,
            figsize=(12, 6)
            
        )
        
        # plot fitted
        fitted = self.data_endog[self.data_endog.index >= self.data_original.index[0]]
        fitted = self.data_endog.iloc[ num_plot: , 1 ]
        fitted2 = df_data_all[df_data_all.index == date_predict_init]
        fitted2 = concat([fitted, fitted2[f'{self.folder.lower()}_predicted']])
        date_slice_plot = self.data_original.index[0]
        fitted2 = fitted2[(fitted2.index >= date_slice_plot)]
        
        fitted2 = fitted2.plot(xlabel="", ylabel="", color=self.color2)
        
        # plot confidence interval
        #predict_conf_95 = fill_between(
        #    
        #    conf_95.index,
        #    conf_95.iloc[ : , 0 ],
        #    conf_95.iloc[ : , 1 ],
        #    color=self.color4,
        #    alpha=0.05
        #    
        #)
        #
        #predict_conf_50 = fill_between(
        #    
        #    conf_50.index,
        #    conf_50.iloc[ : , 0 ],
        #    conf_50.iloc[ : , 1 ],
        #    color=self.color4,
        #    alpha=0.1
        #    
        #)
        
        # predicted plot
        predicted = df_data_all[f"{self.folder.lower()}_predicted"].plot(color=self.color3)
       
        # plot legends
        plt.legend(
            
            [
                
                f"observed",
                f"fitted model (R² = {r2:.2f}%)",
        #        f"conf. int. 95%", 
        #        f"conf. int. 50%",
                f"forecast"
                
            ],
            
            frameon = False
            
        )
        
        # plot config
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.gcf().autofmt_xdate()
        date_format = mpl_dates.DateFormatter('%b. %Y')
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.xlabel("")
        plt.ylabel(self.ylabel)
        plt.tight_layout()
        
        # save fig
        plt.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_13_observed_fitted_predict_{self.period}.jpg")
        
        return
