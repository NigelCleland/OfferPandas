#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import DataFrame
import numpy as np
from collections import defaultdict
import datetime
import itertools
import os

class OfferFrame(DataFrame):
    """docstring for OfferFrame"""


    def __new__(cls, *args, **kargs):

        arr = DataFrame.__new__(cls, *args, **kargs)

        if type(arr) is OfferFrame:
            return arr
        else:
            return arr.view(OfferFrame)

    def transmogrify(self):
        arr = self._retitle_columns()._scrub_whitespace()._map_frame()
        arr = arr._convert_date()
        arr = arr._create_timestamp().stack_frame()._market_node()
        return arr

    def _retitle_columns(self):

        self.rename(columns={x: x.strip().title()
                    for x in self.columns}, inplace=True)

        grid_names = ("Grid_Point", "Grid_Injection_Point", "Grid_Exit_Point")
        self.rename(columns={x: "Node" for x in grid_names}, inplace=True)

        return self

    def _scrub_whitespace(self):
        for col in self.columns:
            if self[col].dtype == "O":
                self[col] = self[col].apply(lambda x: x.strip())

        return self


    def _map_frame(self):
        location = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "_static/nodal_metadata.csv")
        mapping = pd.read_csv(location)
        mapping.rename(columns={x: x.replace(' ', '_')
                        for x in mapping.columns}, inplace=True)
        return OfferFrame(self.merge(mapping, left_on="Node", right_on="Node"))


    def stack_frame(self):
        arr = pd.concat(self._stack_frame(), ignore_index=True)
        max_names = ("Power", "Max")
        arr.rename(columns={x: "Quantity" for x in max_names}, inplace=True)
        return OfferFrame(arr)


    def _stack_frame(self):

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
        band_columns = [x for x in self.columns if "Band" in x]
        fdict = defaultdict(dict)
        for band in band_columns:
            number, param = (int(band.split('_')[0][4:]), band.split('_')[-1])
            rt, pt = (self._reserve_type(band), self._product_type(band))
            fdict[(pt, rt, number)][param] = band
        return fdict



    def _reserve_type(self, band):
        return "FIR" if "6S" in band else "SIR" if "60S" in band else "Energy"


    def _product_type(self, band):
        if "Plsr" in band:
            return "PLSR"
        elif "Twdsr" in band:
            return "TWDSR"
        elif "6S" in band or "60S" in band:
            return "IL"
        else:
            return "Energy"


    def _convert_date(self):
        self["Trading_Date"] = pd.to_datetime(self["Trading_Date"])
        return self


    def _create_timestamp(self):
        num_min = {x: datetime.timedelta(minutes=int(x*30-15))
                  for x in self["Trading_Period"].unique()}

        minutes = self["Trading_Period"].map(num_min)
        self["Timestamp"] = self["Trading_Date"] + minutes
        return self

    def efilter(self, **kargs):
        arr = self.copy()
        for key, value in kargs.iteritems():
            arr = arr[arr[key] == value]
        return OfferFrame(arr)

    def rfilter(self, **kargs):
        arr = self.copy()
        for key, values in kargs.iteritems():
            arr = arr[(arr[key] >= values[0]) & (arr[key] <= values[1])]
        return OfferFrame(arr)

    def nfilter(self, **kargs):
        arr = self.copy()
        for key, value in kargs.iteritems():
            self = arr[arr[key] != value]

        return OfferFrame(self)


    def price_stack(self):
        return self.groupby(["Timestamp", "Price"])["Quantity"].sum()


    def incrementalise(self):
        self["Incr Quantity"] = 1
        return OfferFrame(pd.concat(self._single_increment(), axis=1).T)


    def _market_node(self):
        lammn = lambda x: " ".join([x[0], "".join([x[1], str(x[2])])])
        self["Market_Node_ID"] = self[["Node", "Station", "Unit"]].apply(
                        lammn, axis=1)
        return self


    def _single_increment(self):
        """ Note assume a single timestamp """
        for index, series in self.iterrows():
            power = series["Quantity"]
            series["Incr Quantity"] = 1
            while power > 0:
                series["Incr Quantity"] = min(1, power)
                yield series.copy()
                power -= 1


    def _bathtub(self, capacity):

        arr = self.sort("Price")
        capline = np.arange(1, capacity+1,1)
        rline = np.zeros(len(capline))
        filt_old = np.zeros(len(capline))
        rdict = {}
        for index, row in arr.iterrows():
            price = row["Price"]

            reserve = capline * row["Percent"] / 100.
            rmap = np.where(reserve <= row["Quantity"], reserve, row["Quantity"])

            rline = rline + rmap
            filt = np.where(rline <= capline[::-1], rline, capline[::-1])
            entry = filt - filt_old
            rdict[price] = entry[1:] - entry[:-1]
            filt_old = filt.copy()

        # Create a DF
        df = pd.DataFrame(rdict)
        df.index = np.arange(1, len(df)+1,1)
        df = pd.DataFrame({"Reserve_Quantity": df.stack()})
        df["Cumulative_Quantity"] = df.index.map(lambda x: x[0])
        df["Reserve_Price"] = df.index.map(lambda x: x[1])
        df["Market_Node_ID"] = arr["Market_Node_ID"].unique()[0]
        #df["Timestamp"] = arr["Timestamp"].unique()[0]
        return df

    def _merge_incr(self, reserve):

        #self["Cumulative_Quantity"] = self["Incr Quantity"].cumsum()
        indices = ("Market_Node_ID", "Cumulative_Quantity")
        if len(reserve) > 0:
            bath = reserve._bathtub(self["Max_Output"].max())
            return self.merge(bath, left_on=indices, right_on=indices,
                         how='outer')
        else:
            return self

    def create_fan(self, reserveoffer, reserve_type="FIR", product_type="PLSR"):
        return pd.concat(self._create_fan(reserveoffer, reserve_type, product_type), ignore_index=True)

    def _create_fan(self, reserveoffer, reserve_type, product_type):

        arr = self.incrementalise()
        for index, stamp, mnode in set(arr[["Timestamp", "Market_Node_ID"]].itertuples()):

            energy = arr.efilter(Timestamp=stamp, Market_Node_ID=mnode).nfilter(Quantity=0)
            reserve = reserveoffer.efilter(Timestamp=stamp, Market_Node_ID=mnode, Reserve_Type=reserve_type, Product_Type=product_type).nfilter(Quantity=0)

            energy["Cumulative_Quantity"] = energy["Incr Quantity"].cumsum()
            #bathframe = reserve._bathtub(energy["Max_Output"].max())

            yield energy._merge_incr(reserve)










