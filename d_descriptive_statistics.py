from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy import absolute as abs
from numpy.fft import fft
from numpy import std, var


class Time_serie_level:
    """
    Elaboration of descriptive statistics results.

    Required settings:
    - data (input_data)
    - folder (formatted dependent variable - "NAME VARIABLE")
    - period
    - ylabel
    - date_train_end

    Optional settings:
    - style_graph (graphic style)
    - color1 (color setting)
    - color2 (color setting)
    - color3 (color setting)
    - color4 (color setting)
    - color5 (color setting)

    """

    def __init__(
        
        self,
        data,
        folder,
        period,
        ylabel,
        date_train_end,
        style_graph = "seaborn", 
        color1 = "royalblue", 
        color2 = "crimson", 
        color3 = "darkorange", 
        color4 = "black", 
        color5 = "red"
        
        ):
        """
        Settings for the outputs.
        """
        
        # data frame
        self.data_endog = data[(data.index <= date_train_end)].iloc[ : , 0 ]
        
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


    def time_serie_plot(self):
        """
        Time series plot.
        """
        
        # style
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=( 12 , 6), dpi=300)
        
        # set
        plt.title(f"TIME SERIE (LEVEL) - {self.period.upper()} {self.folder.upper()}")
        
        # plot
        ts_plot = plt.plot(self.data_endog, linestyle="solid",
                                        color=self.color1, 
                                        linewidth = 2)
        
        # plot config
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False) 
        plt.gcf().autofmt_xdate() # year
        date_format = mpl_dates.DateFormatter('%b. %Y') # month, year
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.xlabel("")
        plt.ylabel(self.ylabel)
        plt.tight_layout()
        
        # save
        plt.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_1.0_time_serie_{self.period}.jpg")
        
        return


    def moving_average_d(self):
        """
        Plotting moving averages.
        """
        
        # style
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=( 12 , 6), dpi=400)
        
        # set
        plt.title(f"MOVING AVERAGE - {self.period.upper()} {self.folder.upper()}")
        
        # ma 20
        ma20 = self.data_endog.rolling(20).mean()
        plt.plot(ma20, linestyle="solid", color="orangered", linewidth=1)
        
        # ma 60
        ma60 = self.data_endog.rolling(60).mean()
        plt.plot(ma60, linestyle="solid", color="orange", linewidth=1)
        
        # ma 120
        ma120 = self.data_endog.rolling(120).mean()
        plt.plot(ma120, linestyle="solid", color="deepskyblue", linewidth=1)
        
        # ma 240
        ma240 = self.data_endog.rolling(240).mean()
        plt.plot(ma240, linestyle="solid", color="magenta", linewidth=1)
        
        # plot
        ts_plot = plt.plot(self.data_endog, linestyle="solid", color='black', 
                           linewidth = 1)
        
        # plot legends
        plt.legend(
            
            [
                
                "ma 20",
                "ma 60",
                "ma 120", 
                "ma 240",
                F"{self.folder}"
                
            ],
            
            frameon = False
            
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
        plt.ylabel(self.ylabel)
        plt.tight_layout()
        
        # save
        plt.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_1.1_moving_average_{self.period}.jpg")
        
        return


    def moving_average_m(self):
        """
        Plotting moving averages.
        """
        
        # style
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        fig, ax = plt.subplots(1, 1, sharex=False, figsize=( 12 , 6), dpi=400)
        
        # set
        plt.title(f"MOVING AVERAGE - {self.period.upper()} {self.folder.upper()}")
        
        # ma 3
        ma3 = self.data_endog.rolling(3).mean()
        plt.plot(ma3, linestyle="solid", color="orangered", linewidth=1)
        
        # ma 6
        ma6 = self.data_endog.rolling(6).mean()
        plt.plot(ma6, linestyle="solid", color="orange", linewidth=1)
        
        # ma 12
        ma12 = self.data_endog.rolling(12).mean()
        plt.plot(ma12, linestyle="solid", color="deepskyblue", linewidth=1)
        
        # ma 24
        ma24 = self.data_endog.rolling(24).mean()
        plt.plot(ma24, linestyle="solid", color="magenta", linewidth=1)
        
        # plot
        ts_plot = plt.plot(self.data_endog, linestyle="solid", color='black', 
                           linewidth = 1)
        
        # plot legends
        plt.legend(
            
            [
                
                "ma 3",
                "ma 6",
                "ma 12", 
                "ma 24",
                F"{self.folder}"
                
            ],
            
            frameon = False
            
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
        plt.ylabel(self.ylabel)
        plt.tight_layout()
        
        # save
        plt.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_1.1_moving_average_{self.period}.jpg")
        
        return


    def acf_pacf_plot(self):
        """
        ACF and PACF plots.
        """
        
        # style
        fig, ax = plt.subplots(2, 1, sharex=False, figsize=(12, 6), dpi=300)

        # plot
        acf = plot_acf(
            
            self.data_endog.values.squeeze(),
            title = f"ACF (LEVEL) - {self.period.upper()} {self.folder.upper()}",
            color = self.color1,
            vlines_kwargs = {"colors": self.color1},
            alpha = 0.05,
            ax = ax[0],
            zero = True
          
        )
        
        pacf = plot_pacf(
            
            self.data_endog.values.squeeze(),
            title = f"PACF (LEVEL) - {self.period.upper()} {self.folder.upper()}",
            color = self.color2,
            vlines_kwargs = {"colors": self.color2},
            alpha = 0.05,
            ax = ax[1],
            zero = True
         
        )
       
        
        ax[0].set_ylim(-1.1, 1.1) 
        ax[1].set_ylim(-1.1, 1.1)
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
        fig.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_2_acf_pacf_level_{self.period}.jpg")
        
        return


    def periodogram_plot(self):
        """
        Periodogram plot.
        """
        
        # selection
        self.data_endog = self.data_endog
        
        # style
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 6), dpi=300)
        plt.rcParams.update({'font.size': 12})
        plt.style.use(self.style_graph)
        
        # config
        plt.title(f"PERIODOGRAM (LEVEL) - {self.period.upper()} {self.folder.upper()}")
        
        ps = abs(fft(self.data_endog, n=len(self.data_endog / 4 )))  ** 2
        
        periodogram = plt.plot(ps[1:], color=self.color1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False) 
        
        plt.tight_layout()
        
        # save
        fig.savefig(f"1_data/{self.folder}/{self.period}/results/{self.folder}_3_periodogram_level_{self.period}.jpg")
        
        return


    def descriptive_stat(self):
        """
        Descriptive data analysis
        """
       
        # define variables
        mean = self.data_endog.mean()
        median = self.data_endog.median()
        std_sample = std(self.data_endog)
        variance = var(self.data_endog)
        lowest = self.data_endog.min()
        highest = self.data_endog.max()
        
        # frame
        results_txt = (
            
            f"{'-' * 50}\n"
            f"Descriptive analysis:\n\n"
            f"Variable: {self.folder} (level)\n"
            f"Mean: {mean:.2f}\n"
            f"Median: {median:.2f}\n"
            f"Sample std: {std_sample:.2f}\n"
            f"Variance: {variance:.2f}\n"
            f"Lowest: {lowest:.2f}\n"
            f"Highest: {highest:.2f}\n"
            f"{'-' * 50}\n"
          
        )
        
        # export
        with open(f"1_data/{self.folder}/{self.period}/results/{self.folder}_4_descriptive_statistics_level_{self.period}.txt", 'w') as desc_stat:
            desc_stat.write(results_txt)
        
        return
