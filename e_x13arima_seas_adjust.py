from statsmodels.tsa.x13 import x13_arima_analysis as x13a
from pandas import DataFrame, read_csv, concat, date_range, Series
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
from sys import platform


class X13_arima_desaz:
    """
    X-13 ARIMA-SEATS, successor to X-12-ARIMA and X-11, is a set of statistical 
    methods for seasonal adjustment and other descriptive analysis of time 
    series data that are implemented in the U.S. Census Bureau's.
    
    Required settings:
    - data_endog (dependent variable)
    - data_exogs (independent variables)
    - variable (formatted dependent variable - "NAME VARIABLE")
    - period (string)
    - ylabel (string)
    - path (Directory of the folder where x13 arima seats are located)
    - freq (Period frequency)
    - date_train_init,
    - date_predict_end,
    - date_predict_end,
    
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
        data_endog,
        data_exogs,
        folder,
        period,
        ylabel,
        path,
        freq,
        date_train_init,
        date_train_end,
        date_predict_end,
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
        self.data_endog = data_endog[(data_endog.index <= date_train_end)]
        self.data_exogs = data_exogs
        
        # configs
        self.folder = folder
        self.variable_ = folder.replace(" ", "_").lower()
        self.freq = freq
        self.period = period
        self.ylabel = ylabel
        self.date_train_init = date_train_init
        self.date_train_end = date_train_end
        self.date_predict_end = date_predict_end
        
        # style
        self.style_graph = style_graph
        self.color1 = color1
        self.color2 = color2
        self.color3 = color3
        self.color4 = color4
        self.color5 = color5
        
        # X13-ARIMA-SEATS CONFIG
        self.path = path
        self.x13_desaz = x13a(self.data_endog, x12path=self.path)


    def x13_results(self):
        """
        Results obtained with X13-ARIMA-SEATS (dependent variable)
        """
        
        # style
        fig, ax = plt.subplots(4, 1, sharex=True, figsize=( 12 , 6), dpi=300)
        plt.style.use(self.style_graph)
        plt.rcParams.update({'font.size': 12})
       
        # x13 results 
        x13_seasonal = DataFrame(self.x13_desaz.seasadj.values,
                                 index=self.data_endog.index.values,
                                 columns=[self.folder])
        
        x13_trend = DataFrame(self.x13_desaz.trend.values,
                                 index=self.data_endog.index.values,
                                 columns=[self.folder])
        
        x13_irregular = DataFrame(self.x13_desaz.irregular.values,
                                 index=self.data_endog.index.values,
                                 columns=[self.folder])
        
        # plot
        x13_original = ax[0].plot(self.data_endog,
                                 color=self.color1)
        
        # config
        ax[0].set_ylabel("original")
        x13_desazonal = ax[1].plot(x13_seasonal, color=self.color1)
        
        ax[1].set_ylabel("seas. adjusted")
        x13_trend = ax[2].plot(x13_trend, color=self.color1)
        
        ax[2].set_ylabel("trend")
        x13_irreg = ax[3].plot(x13_irregular, color=self.color1)
        
        ax[3].set_ylabel("irregular")
        ax[0].set_title(f"X13-ARIMA RESULTS - {self.period.upper()} {self.folder.upper()}")
        
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['left'].set_visible(False) 
        ax[0].get_yaxis().set_ticks([])
        
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False) 
        ax[1].get_yaxis().set_ticks([])
        
        ax[2].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['bottom'].set_visible(False)
        ax[2].spines['left'].set_visible(False) 
        ax[2].get_yaxis().set_ticks([])
        
        ax[3].spines['top'].set_visible(False)
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['bottom'].set_visible(False)
        ax[3].spines['left'].set_visible(False) 
        ax[3].get_yaxis().set_ticks([])
        
        plt.gcf().autofmt_xdate() # year
        date_format = mpl_dates.DateFormatter('%b. %Y') # month, year
        plt.gca().xaxis.set_major_formatter(date_format) 
        
        plt.tight_layout()
        
        # save
        plt.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_5_x13_results_{self.period}.jpg")
        
        return


    def x13_seasonal_adjustment(self):
        """
        X13 Seasonal adjustment (dependent variable).
        """
        
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=( 12 , 6), dpi=300)
        
        x13_seasonal = DataFrame(self.x13_desaz.seasadj.values,
                                 index=self.data_endog.index.values,
                                 columns=[self.folder])
        
        x13_seasonal_plot_raw = plt.plot(self.data_endog,
                                         color=self.color1,
                                         label="original")
        
        x13_seasonal_plot = plt.plot(x13_seasonal,
                                     color=self.color2,
                                     label="seasonal adjustment")
        
        # config
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False) 
        plt.gcf().autofmt_xdate() # year
        date_format = mpl_dates.DateFormatter('%b. %Y') # month, year
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.ylabel(self.ylabel)
        plt.legend(loc=0, frameon=False)
        plt.title(f"X13-ARIMA SEASONAL ADJUSTMENT - {self.period.upper()} {self.folder.upper()}")
        
        plt.tight_layout()
        
        # save
        plt.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_6_x13_seasonal_adjustment_{self.period}.jpg")
        
        # new data frame
        x13_seasonal.to_csv(f"1_data/{self.folder}/{self.period}/data_base/{self.folder}_seasonal_adjustment_{self.period}.csv",
                               index_label="index_date", sep=",")
        
        return


    def independent_desaz_x13(self):
        """
        Deseasonalization of independent variables.
        """
        
        df_seas_raw = read_csv(
            
            f"1_data/{self.folder}/{self.period}/data_base/{self.folder}_seasonal_adjustment_{self.period}.csv",
            sep = ","
            
        )
        
        list_exog_col = self.data_exogs.columns.to_list()
        
        data_exog_fore = self.data_exogs
        
        for col in list_exog_col:
            
            df_desaz_indep = DataFrame(data_exog_fore[col])
            
            desaz_indep_x13 = x13a(
                
                df_desaz_indep,
                x12path = self.path,
                
            ).seasadj.values
            
            df_seas_raw = concat(
                
                [
                    
                    df_seas_raw,
                    
                    DataFrame(
                        
                        desaz_indep_x13,
                        columns=[col]
                        
                    )
                    
                ], 
                
                axis=1
                
            )
        
        index = date_range(
            
            start = self.date_train_init,
            end = self.date_predict_end,
            freq = self.freq
            
        )
        
        df_seas_raw = concat( [DataFrame(index), df_seas_raw.iloc[ : , 1: ]], axis=1)
        
        df_seas_raw = df_seas_raw.rename(columns={df_seas_raw.columns[0]:"index_date"})
        
        df_seas_raw.to_csv(
            
            f"1_data/{self.folder}/{self.period}/data_base/{self.folder}_seasonal_adjustment_{self.period}.csv",
            sep = ",",
            index = False
            
        )
        
        return
