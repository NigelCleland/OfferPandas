#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import datetime
import itertools
import os

import pandas as pd
from pandas import DataFrame
import numpy as np

from dateutil.parser import parse

import OfferPandas

def load_offerframe(fName, *args, **kargs):
    """ This is a publically exposed generic function used to create
    the Frame object containing csv data. It is the primary method
    through which data should be read into the Frames
    """
    df = pd.read_csv(fName, *args, **kargs)
    frame = Frame(df)

    frame = frame._column_mapping()
    frame = frame._remove_data_whitespace()
    frame = frame._market_node()
    frame = frame._map_locations()
    frame = frame._parse_dates()
    frame = frame._create_identifier()
    frame = frame._stack_frame()

    return frame


class Frame(DataFrame):
    """A Frame is a customised DataFrame object which is specific
    to Energy and Reserve market offer data. It is a base class which
    creates some custom formatting options and included useful utilities
    such as renaming columns, creating some helper columns and mapping
    locations as well as the primary DataFrame methods still easily available

    The best method of creating a Frame is through the load_offerframe method
    which is also publically expsosed, this creates the dataframe as well
    as a number of convenience formatting options.

    If creating manually, you must pass an existing Pandas DataFrame with
    the offer data in the standard WITS format.
    """

    def __new__(cls, *args, **kargs):

        arr = DataFrame.__new__(cls)

        if type(arr) is Frame:
            return arr
        else:
            return arr.view(Frame)

    def _column_mapping(self):
        """ Update the Column Mapping to improve the naming structure,
        Major changes include stripping white space and moving towards
        a title case as well setting a consistent naming schema for
        the Grid Points across Energy IL and Reserve

        Returns
        -------
        Frame: A Frame object for method chaining.
        """
        # Title and strip all white space from the columns
        column_mapping = {x: x.strip().title() for x in self.columns}
        self.rename(columns=column_mapping, inplace=True)

        # Update the grid names to a consistent naming structure
        nodal_names = ("Grid_Point", "Grid_Injection_Point", "Grid_Exit_Point")
        nodal_mapping = {x: "Bus_Id" for x in nodal_names}
        self.rename(columns=nodal_mapping, inplace=True)

        return self

    def _remove_data_whitespace(self):
        """ The data can have unnecessary beginning and trailing white space
        which must be removed in order to make the analysis work properly.

        There is a type chech on the data columns, will only apply the method
        to columns which are of the object type.

        Returns
        -------
        Frame: A Frame object for method chaining.
        """

        # Apply the stripping to columns which are an Object type
        # Use a throwaway lambda function to do this
        strip = lambda x: x.strip()
        for col in self.columns:
            if self[col].dtype == "O":
                self[col] = self[col].apply(strip)

        return self

    def _map_locations(self):
        """ Map the OfferFrame with location data from a reference CSV file
        """
        # Get the Map data as a DataFrame object
        file_path = OfferPandas.__path__[0]
        map_path = '_static/nodal_metadata.csv'
        full_path = os.path.join(file_path, map_path)
        map_data = pd.read_csv(full_path)

        # Remove the blank space and replace by underscores to merge data
        column_mapping = {x: x.replace(' ', '_') for x in map_data.columns}
        map_data.rename(columns=column_mapping, inplace=True)

        # Merging the data will spit it back as a general data frame so
        # We need to call Frame again
        map_points = ["Node", "Bus_Id"]
        return Frame(self.merge(map_data, left_on=map_points,
                                          right_on=map_points))

    def _create_identifier(self):
        """ Create the Trading Period Identifier to make merging easier
        between different data sets. This identifier is of the form,
        yyyymmddpp and is stored as an integer where possible
        """

        date_lam = lambda x: x.strftime('%Y%m%d')
        period_lam = lambda x: "%02d" % x

        dates = self["Trading_Date"].apply(date_lam)
        periods = self["Trading_Period"].apply(period_lam)
        self["Trading_Period_ID"] = dates + periods
        self["Trading_Period_ID"] = self["Trading_Period_ID"].astype(int)
        return self

    def _parse_dates(self):
        """ Parse the dates using the general parse function from dateutils
        Then apply this as a mapping, this is done as parsing many dates can
        be very slow, this method is much quicker
        """

        unique_dates = self["Trading_Date"].unique()
        date_mapping = {x: parse(x) for x in unique_dates}
        self["Trading_Date"] =  self["Trading_Date"].map(date_mapping)
        return self


    def _market_node(self):
        """ Create a Market Node Identifier for the generation data
        If the data is IL, attributed by the lack of a Unit column
        then the market node id is taken to be the Bus Id

        Example:

        GXP: HLY2201, Station: HLY: Unit 5
        Market_Node = HLY2201 HLY5

        """

        if "Unit" in self.columns:
            station = self["Station"] + self["Unit"].astype(str)
            identifier = self["Bus_Id"] + " " + station
            self["Node"] = identifier

        else:
            self["Node"] = self["Bus_Id"]

        return self

    def _stack_frame(self):
        """ General Function to move from a horizontal format to a vertical
        format which is easier to work with for analysis.
        It accomplishes this by parsing the column names to figure out what
        type of data something is and then creating column entries based upon
        this whilst retaining meta information from the original columns.

        This returns a DataFrame which is significantly larger, but should
        be easier to work with and do analysis.

        """

        arr = pd.concat(self._yield_frame(), ignore_index=True)
        max_names = ("Power", "Max")
        arr.rename(columns={x: "Quantity" for x in max_names}, inplace=True)
        return Frame(arr)

    def _yield_frame(self):

        general_columns = [x for x in self.columns if  "Band" not in x]
        fdict = self._classify_bands()

        for key in fdict:
            allcols = general_columns + fdict[key].values()
            single = self[allcols].copy()
            single["Product_Type"] = key[0]
            single["Reserve_Type"] = key[1]
            single["Band"] = key[2]
            single.rename(columns={v:k for k, v in fdict[key].items()},
                          inplace=True)
            yield single


    def _classify_bands(self):
        """ Function to classify a band based on the column name
        It returns a dictionary which is then merged back into the offer
        frame data

        Examples:
        ---------

        Input: Band1_PLSR_6S_Max
        Output:
            Band: 1, Product_Type: PLSR, Reserve_Type: FIR, Parameter: Quantity

        Input: Band5_Power
        Output:
            Band: 5, Product_Type: Energy, Reserve_Type: Energy,
            Parameter: Quantity

        Returns
        -------
        band_listing: A dictionary indexed by tuples and paramters with the
                      column name as the value.

        """

        def band_classifier(band):
            split = band.split('_')
            return int(split[0][4:]), split[-1]

        def reserve_classifier(band):
            """ Determines whether the band is a FIR or SIR band"""
            if "6S" in band:
                return "FIR"
            elif "60S" in band:
                return "SIR"
            else:
                return "Energy"

        def product_classifier(band):
            if "Plsr" in band:
                return "PLSR"
            elif "Twdsr" in band:
                return "TWDSR"
            elif "6S" in band or "60S" in band:
                return "IL"
            else:
                return "Energy"

        band_columns = [x for x in self.columns if "Band" in x]
        band_listing = defaultdict(dict)
        for band in band_columns:

            number, param =  band_classifier(band)
            rt, pt = (reserve_classifier(band), product_classifier(band))

            band_listing[(pt, rt, number)][param] = band

        return band_listing

    def efilter(self, *args, **kargs):
        """ A general purpose filter method which can take either
        a dictionary of arguments or specific command line implementations
        E.g. {"Company_Name": "Mighty River"} is equivalent to the keyword
        argument Company_Name="Mighty River" and should return the
        same result

        Furthermore, these methods should be able to be mixed up, although
        only one dictionary may be passed as an argument

        Can also handle iterable arguments interchangeabley with iterables
        to do a multiple match.

        Returns
        -------
        Frame: Filters applied, this is a new object, leaves original object
               intact
        """

        arr = self.copy()
        if args:
            for key, value in args[0].iteritems():
                if hasattr(value, "__iter__"):
                    arr = pd.concat((arr[arr[key] == each] for each in value),
                                    ignore_index=True)
                else:
                    arr = arr[arr[key] == value]

        if kargs:
            for key, value in kargs.iteritems():
                if hasattr(value, "__iter__"):
                    arr = pd.concat((arr[arr[key] == each] for each in value),
                                    ignore_index=True)
                else:
                    arr = arr[arr[key] == value]

        return Frame(arr)


    def rfilter(self, *args, **kargs):
        """ Implements a general purpose range filter for looking at numerical
        quantities, note, to specify a lower or upper range use arbitrarily
        low or high values, for example, 0 and 1000000. Not perfect but is
        simpler than implementing a custom ge/gt/lt/le filters which just
        clutter the code base.

        Example Usage:
        --------------

        Frame.rfilter({Price: (10,40)}) # Will get all offers with prices
                                          between $10 and $40.

        This will also work with dates, but not with string type matching
        e.g. a range of companies

        Returns
        -------
        Frame: Filters applied, this is a new object, leaves original object
               intact

        """
        arr = self.copy()

        if args:
            for key, value in args[0].iteritems():
                arr = arr[(arr[key] >= values[0]) & (arr[key] <= values[1])]

        if kargs:
            for key, values in kargs.iteritems():
                arr = arr[(arr[key] >= values[0]) & (arr[key] <= values[1])]

        return Frame(arr)


    def nfilter(self, *args, **kargs):
        """ A negative filter, useful for excluding specific options, can
        be used in conjunction with either argument or key word arguments.

        Can also handle multiple matchings via a simpler iterable.s

        Example Usage:
        --------------

        Frame.nfilter(Company="MRPL") # Exclude MRPL from analysis
        Frame.nfilter({"Company": "MRPL"}) # Same as above

        # Exclude all MERI offers and all hydro offers in general.
        Frame.nfilter(Company="MERI", {"Generation_Type": "Hydro"})

        Returns
        -------
        Frame: Filters applied, this is a new object, leaves original object
               intact

        """

        arr = self.copy()

        if args:
            for key, value in args[0].iteritems():
                if hasattr(value, "__iter__"):
                    arr = pd.concat((arr[arr[key] != each] for each in value),
                                    ignore_index=True)
                else:
                    arr = arr[arr[key] != value]

        if kargs:
            for key, value in kargs.iteritems():
                if hasattr(value, "__iter__"):
                    arr = pd.concat((arr[arr[key] != each] for each in value),
                                  ignore_index=True)
                else:
                    arr = arr[arr[key] != value]

        return Frame(arr)


if __name__ == '__main__':
    pass



