#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import DataFrame
import numpy as np
from collections import defaultdict
import datetime
import itertools

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
        mapping = pd.read_csv("_static/nodal_metadata.csv")
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

        def band_classifier(band):
            split = band.split('_')
            return int(split[0][4:]), split[-1]

        def reserve_classifier(band):
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

    def create_map(self, func, items):
        return {x: func(x) for x in items}

    def _convert_date(self):
        self["Trading_Date"] = pd.to_datetime(self["Trading_Date"])
        return self


    def _create_timestamp(self):
        create_time = lambda x: datetime.timedelta(minutes=int(x*30-15))
        periods = np.arange(1,51,1)
        minute_map = self.create_map(create_time, periods)

        minutes = self["Trading_Period"].map(minute_map)
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
        return OfferFrame(pd.concat(self._single_increment(), axis=1).T)


    def _market_node(self):

        def create_node(series):
            station = "".join([series["Station"], str(series["Unit"])])
            node = " ".join([series["Node"], station])
            return node

        self["Market_Node_ID"] = self.apply(create_node, axis=1)
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
        """ A series of operations to create a Bathtub constraint upon
        a reserve offer for a particular unit.
        The maximum capacity of the unit needs to be passed to the function.
        This function is called by the fan curve function.

        """
        arr = self.sort("Price")
        market_node = arr["Market_Node_ID"].unique()[0]

        # Set up vectors for marginal calculations
        capacity_line = np.arange(1, capacity+1,1)
        reserve_line = np.zeros(len(capacity_line))
        previous_offer = np.zeros(len(capacity_line))

        offer_curves = {}

        for index, row in arr.iterrows():
            price = row["Price"]

            # Proportionality Constraint Line
            incr_reserve = capacity_line * row["Percent"] / 100.

            # Maximum Offer Line
            partial_offer = np.where(incr_reserve <= row["Quantity"],
                            incr_reserve, row["Quantity"])

            # Add them together
            reserve_line = reserve_line + partial_offer

            # Combined Capacity Line
            full_offer = np.where(reserve_line <= capacity_line[::-1],
                                  reserve_line, capacity_line[::-1])

            # Get a Marginal Offer for moving up the stacks
            marginal_offer = full_offer - previous_offer
            offer_curves[price] = marginal_offer[1:] - marginal_offer[:-1]
            previous_offer = full_offer.copy()

        # Create a DF
        df = pd.DataFrame(offer_curves)

        df.index = np.arange(1, len(df)+1,1)

        bathtub = pd.DataFrame({"Reserve_Quantity": df.stack()})
        # Filter the index to get the cumulaive quantity and price
        bathtub["Cumulative_Quantity"] = bathtub.index.map(lambda x: x[0])
        bathtub["Reserve_Price"] = bathtub.index.map(lambda x: x[1])

        bathtub["Market_Node_ID"] = market_node
        #bathtub["Timestamp"] = arr["Timestamp"].unique()[0]
        return bathtub

    def _merge_incr(self, reserve):

        #self["Cumulative_Quantity"] = self["Incr Quantity"].cumsum()
        indices = ("Market_Node_ID", "Cumulative_Quantity")
        if len(reserve) > 0:
            bath = reserve._bathtub(self["Max_Output"].max())
            return self.merge(bath, left_on=indices, right_on=indices,
                              how='outer')
        else:
            return self


    def create_fan(self, reserveoffer, reserve_type="FIR",
                   product_type="PLSR"):
        """ Creates a Fan Curve representation of the data which can be
        used to view the trade off between energy and reserve offers
        at either a composite or an individual station level.

        """
        return pd.concat(self._create_fan(reserveoffer, reserve_type,
                         product_type), ignore_index=True)


    def _create_fan(self, reserveoffer, reserve_type, product_type):
        """ Creates the fan curve
        """

        arr = self.incrementalise()
        for index, stamp, mnode in set(arr[["Timestamp",
                                            "Market_Node_ID"]].itertuples()):

            energy = arr.efilter(Timestamp=stamp,
                                 Market_Node_ID=mnode).nfilter(Quantity=0)

            reserve = reserveoffer.efilter(Timestamp=stamp,
                                           Market_Node_ID=mnode,
                                           Reserve_Type=reserve_type,
                                           Product_Type=product_type
                                           ).nfilter(Quantity=0)

            energy["Cumulative_Quantity"] = energy["Incr Quantity"].cumsum()
            #bathframe = reserve._bathtub(energy["Max_Output"].max())

            yield energy._merge_incr(reserve)










