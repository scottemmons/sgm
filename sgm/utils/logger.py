"""
Author: redacted
Email: redacted
Date: August 30, 2019
Purpose: Automate logging of data to file
"""

import pandas as pd
import os

class Logger:
    """
    Not to be instantiated directly. Instead, use child class DFLogger or
    RowLogger.
    """
    def __init__(self, filename, columns, append=False):
        """
        Parameters:
        filename, string specifying path to write csv
        columns, a list of strings naming the columns of the data
        append, a Boolean specifying whether or not to append to results already at filename
        """
        self.fname = filename
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self.initColumns(columns)
        self.initAppend(append)

    def initColumns(self, columns):
        """
        Description:
        Reset the data using the provided column names.
        Parameters:
        columns, a list of strings naming the columns of the data
        """
        self.col = columns

    def initAppend(self, append):
        """
        Description:
        Initialize append functionality which, if append, adds logged data to
        data already at self.fname.
        Parameters:
        append, a Boolean specifying whether or not to append to results already at self.fname.
        """
        self.append = append

    def add(self, *args):
        """
        Description:
        Add a row.
        Parameters:
        args, a list of values in the order of the column names
        """
        assert len(args) == len(self.col)

class DFLogger(Logger):
    """
    Hold pandas DataFrame in memory and write entire DataFrame to CSV upon
    addition of data.
    """
    def __init__(self, filename, columns=None, append=False, autowrite=False):
        """
        Parameters:
        filename, string specifying path to write csv
        columns, a list of strings naming the columns of the data
        append, a Boolean specifying whether or not to append to results already at filename
        """
        super(DFLogger, self).__init__(filename, columns=columns, append=append)

        if append:
            self.data = pd.read_csv(filename)
            self.col = list(self.data.columns)

        self.autowrite = autowrite

    def initColumns(self, columns):
        """
        Description:
        Reset the data using the provided column names.
        Parameters:
        columns, a list of strings naming the columns of the data
        """
        super(DFLogger, self).initColumns(columns)
        self.data = pd.DataFrame(columns=columns)

    def add(self, *args):
        """
        Description:
        Add a row to the data.
        Parameters:
        args, a list of values in the order of the column names
        """
        super(DFLogger, self).add(*args)
        series = pd.Series(args, index=self.col)
        self.data = self.data.append(series, ignore_index=True)
        if self.autowrite:
            self.write()

    def write(self):
        """
        Description:
        Write all data to file.
        """
        self.data.to_csv(path_or_buf=self.fname, index=False, columns=self.col)

class RowLogger(Logger):
    """
    Append each additional row to end of file, holding no data in memory.
    """
    def __init__(self, filename, columns=None, append=False):
        """
        Parameters:
        filename, a string specifying path to write csv
        columns, a list of strings naming the columns of the data
        append, a Boolean specifying whether or not to append to results already at filename
        """
        super(RowLogger, self).__init__(filename, columns=columns, append=append)

    def initAppend(self, append):
        """
        Description:
        Initialize append functionality which, if append, adds logged data to
        data already at self.fname.
        Parameters:
        append, a Boolean specifying whether or not to append to results already at self.fname.
        """
        if append and os.path.exists(self.fname):
            with open(self.fname, 'r') as f:
                line = f.readline()
                if line[-1] == "\n":
                    line = line[:-1]
                pieces = line.split(",")
            assert len(pieces) == len(self.col)
            for i in range(len(pieces)):
                assert pieces[i] == self.col[i]
        if not append:
            self.writeRow(self.col, 'w')

    def add(self, *args):
        """
        Description:
        Add a row to the log file.
        Parameters:
        args, a list of values in the order of the column names
        """
        super(RowLogger, self).add(*args)
        self.writeRow(args, 'a')

    def writeRow(self, row, mode):
        """
        Description:
        Write one comma-separated row to self.fname.
        Parameters:
        row, a list of values to write
        mode, a string specifying how to open self.fname, e.g., 'w' or 'a'
        """
        with open(self.fname, mode) as f:
            f.write(",".join(row))
            f.write("\n")
