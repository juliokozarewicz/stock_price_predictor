from pandas import date_range, DataFrame, to_datetime


class Dummy_generator:
    """
    Dummy variables generator for time series models.
    
    Required settings:
    - dataset start date
    - dataset end date
    - freq
    
    """

    def __init__(self, start, end, freq):
        """
        Settings for the outputs.
        """
        
        self.freq = freq
        
        self.index = date_range(
            
            start = start,
            end = end,
            freq = freq
            
        )
        
        self.data_dummy = DataFrame(self.index)
        self.data_dummy.columns = ["index_date"]
        self.data_dummy["index_date"] = to_datetime(self.data_dummy["index_date"])
        self.data_dummy = self.data_dummy.set_index('index_date')
        
        return


    def dummy_generator_range(self, variable_name, start_dummy, end_dummy):
        """
        Dummy variables generator for time series models.
        
        Required settings:
        - variable name
        - dataset start date
        - dataset end date
        
        """
        
        self.data_dummy[f"{variable_name}"] = 0
        
        self.data_dummy[ (self.data_dummy.index >= start_dummy) &
                         (self.data_dummy.index <= end_dummy) ] = 1
        
        return self.data_dummy
    
