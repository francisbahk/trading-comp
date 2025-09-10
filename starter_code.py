import pandas as pd
from datetime import datetime
import numpy as np
import scipy.stats as sci
from typing import List
import math


class Strategy:

    def __init__(self) -> None:
        self.capital: float = (
            1000000  # shouldn't it be 1 million if we are ignoring 100x multiplier everywhere else
        )
        self.portfolio_value: float = 0

        self.start_date: datetime = datetime(2024, 1, 1)
        self.end_date: datetime = datetime(2024, 3, 30)

        self.options: pd.DataFrame = pd.read_csv("data/cleaned_options_data.csv")
        self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])

        self.underlyinghour = pd.read_csv("data/underlying_data_hour.csv")
        self.underlyinghour.columns = self.underlyinghour.columns.str.lower()

        self.underlyingminute = pd.read_csv(
            "data/spx_minute_level_data_jan_mar_2024.csv"
        )
        self.underlyingminute.columns = self.underlyingminute.columns.str.lower()
        self.underlyingminute["date"] = self.underlyingminute["date"].astype(str)
        self.underlyingminute["day"] = self.underlyingminute["date"].apply(
            lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:]
        )
        self.underlyingminute["hour"] = self.underlyingminute["ms_of_day"].apply(
            lambda x: self.convert_ms_to_hhmm(x)[0]
        )
        self.underlyingminute["minute"] = self.underlyingminute["ms_of_day"].apply(
            lambda x: self.convert_ms_to_hhmm(x)[1]
        )
        self.underlyingminute["datetocompare"] = (
            pd.to_datetime(self.underlyingminute["date"])
            + pd.to_timedelta(self.underlyingminute["hour"], unit="h")
            + pd.to_timedelta(self.underlyingminute["minute"], unit="m")
        )

        self.dictminute = self.create_underlyingminute_dictionary()
        self.dicthour = self.create_underlyinghour_dictionary()

        # portfolio delta, vega
        self.port_delta = 0
        self.port_vega = 0

        # time difference (to keep track of daylight savings time)
        self.time_diff = 5

        # self.orders: pd.DataFrame = self.user_strategy.generate_orders()
        # self.orders["expiration_date"] = self.orders["option_symbol"].apply(
        #     lambda x: self.get_expiration_date(x)
        # )

    def convert_ms_to_hhmm(self, milliseconds):
        total_seconds = milliseconds // 1000
        total_minutes = total_seconds // 60
        hours = total_minutes // 60
        remaining_minutes = total_minutes % 60
        return [hours, remaining_minutes]

    def create_underlyingminute_dictionary(self):  # SUPPORTS MINUTE DATA!
        dictionary = dict(
            zip(
                pd.to_datetime(self.underlyingminute["datetocompare"]),
                self.underlyingminute["price"],
            )
        )
        final_dict = {pd.Timestamp(key): value for key, value in dictionary.items()}
        # final_dict = {key.to_pydatetime(): value for key, value in dictionary.items()}
        final_final = {
            key.replace(tzinfo=None): value for key, value in final_dict.items()
        }
        return final_final

    def create_underlyinghour_dictionary(self):
        dictionary = dict(
            zip(
                pd.to_datetime(self.underlyinghour["date"]), self.underlyinghour["open"]
            )
        )
        final_dict = {pd.Timestamp(key): value for key, value in dictionary.items()}
        # final_dict = {key.to_pydatetime(): value for key, value in dictionary.items()}
        final_final = {
            key.replace(tzinfo=None): value for key, value in final_dict.items()
        }
        return final_final

    def get_underlying_price_close(
        self, current_date_accurate
    ):  # SUPPORTS MINUTE DATA!
        """
        Input: Current Date to the nearest minute
        Output: Gets the open price of the underlying at the closest time before or equal to current_date_accurate.

        Need to make this fast it seems... the way I do it is match dates first,
        then iterate through hours until you get the closest hour rounded down.
        Can't think of a faster way.
        """
        if current_date_accurate.hour == 21:
            self.time_diff = (
                5  # no DST (if switch DST -> no DST, will be wrong first date)
            )
        if current_date_accurate.hour == 13:
            self.time_diff = 4  # confirmed DST
        if current_date_accurate.hour - self.time_diff >= 16:
            compare = current_date_accurate.replace(
                hour=14,
                minute=0,
                second=0,
                microsecond=0,
            )
            spot_price = self.dictminute[compare]

        elif (
            current_date_accurate.hour == 14 and current_date_accurate.minute == 30
        ):  ## use open since minute data doesn't have at 9:30 (14:30 with 5 hour time diff)
            compare = current_date_accurate.replace(
                hour=14, minute=30, second=0, microsecond=0
            )
            spot_price = self.dicthour[compare]

        else:
            spot_price = self.dictminute[
                current_date_accurate.replace(
                    hour=current_date_accurate.hour - self.time_diff,
                    second=0,
                    microsecond=0,
                )
            ]
        return spot_price

    # def get_underlying_price_close(self, current_date_accurate):
    #     """
    #     Input: Current Date to the nearest minute
    #     Output: Gets the open price of the underlying at the closest time before or equal to current_date_accurate.

    #     Need to make this fast it seems... the way I do it is match dates first,
    #     then iterate through hours until you get the closest hour rounded down.
    #     Can't think of a faster way.
    #     """
    #     if current_date_accurate.hour == 21:
    #         self.time_diff = (
    #             5  # no DST (if switch DST -> no DST, will be wrong first date)
    #         )
    #     if current_date_accurate.hour == 13:
    #         self.time_diff = 4  # confirmed DST
    #     if current_date_accurate.hour - self.time_diff >= 16:
    #         compare = current_date_accurate.replace(
    #             hour=15,
    #             minute=30,
    #             second=0,
    #             microsecond=0,
    #         )
    #         spot_price = self.dict[compare]
    #     elif current_date_accurate.minute >= 30:
    #         # print("DATE RIGHT NOW: " + str(current_date_accurate))
    #         compare = current_date_accurate.replace(
    #             hour=current_date_accurate.hour - self.time_diff,
    #             minute=30,
    #             second=0,
    #             microsecond=0,
    #         )

    #         spot_price = self.dict[compare]
    #     else:
    #         spot_price = self.dict[
    #             current_date_accurate.replace(
    #                 hour=current_date_accurate.hour - (self.time_diff + 1),
    #                 minute=30,
    #                 second=0,
    #                 microsecond=0,
    #             )
    #         ]
    #     return spot_price

    def black_scholes(self, S, K, r, sigma, t):
        """
        Black Scholes Model
        Input: S (Current Stock Price), K (Strike Price), r (risk free rate = 0.03), t (time to maturity), sigma (volatility of stock)
        Output: fair option prices of calls and puts
        """

        N = sci.norm.cdf

        d_1 = (np.log(S / K) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
        d_2 = d_1 - sigma * np.sqrt(t)

        call = S * N(d_1) - K * np.exp(-r * t) * N(d_2)
        put = K * np.exp(-r * t) * N(-d_2) - S * N(-d_1)

        return call, put

    def delta(self, S, K, r, sigma, t, action):
        """
        Calculates delta of an option; params are same as Black-Scholes,
        except for action, which is "C" or "P"
        """

        N = sci.norm.cdf

        d_1 = (np.log(S / K) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))

        if action == "C":
            return N(d_1)

        return N(d_1) - 1

    def vega(self, S, K, r, sigma, t):

        d_1 = (np.log(S / K) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))

        return S * np.sqrt(t) * sci.norm.pdf(d_1)

    def find_vol(self, target_value, S, K, r, t, call_put):
        ind = 0
        if call_put == "P":
            ind = 1
        MAX_ITERATIONS = 200
        PRECISION = 1.0e-4
        sigma = 0.1
        for i in range(0, MAX_ITERATIONS):
            price = (self.black_scholes(S, K, r, sigma, t))[ind]
            vega = self.vega(S, K, r, sigma, t)
            diff = target_value - price  # our root
            if abs(diff) < PRECISION:
                return sigma
            sigma = sigma + diff / vega  # f(x) / f'(x)
        return sigma  # value wasn't found, return best guess so far

    def get_underlying_opening(self, day):
        """
        Returns opening price of underlying on day specified by day
        """

        matching_rows = self.underlyinghour[
            self.underlyinghour["date"].str.startswith(day)
        ]
        opening_price = matching_rows.iloc[0]["open"]
        return opening_price

        # open_price = self.underlying[self.underlying["date"] == day]["open"].iloc[0]
        # return open_price

    def parse_option_symbol(self, symbol) -> List:
        """
        example: SPX   240419C00800000
        Returns: [Expiration date, action, strike_price]
        output example: [24, 04, 19, C, 800]
        """
        numbers: str = symbol.split(" ")[3]
        date: str = numbers[:6]
        date_yymmdd: str = "20" + date[0:2] + "-" + date[2:4] + "-" + date[4:6]
        action: str = numbers[6]
        strike_price: float = float(numbers[7:]) / 1000
        return [datetime.strptime(date_yymmdd, "%Y-%m-%d"), action, strike_price]

    def margin_ok(self, row, order):
        """
        Returns True if we have margin requirements to satisfy
        order of a particular row, False otherwise
        """

        expiry, c_p, strike = self.parse_option_symbol(row["symbol"])
        sz = order["order_size"]  # size of the order

        if c_p == "P":  # put options
            if order["action"] == "B":  # buying a put
                return (
                    self.capital > (row["ask_px_00"] + 0.1 * strike) * sz
                )  # based on ask price
            else:  # selling a put
                return (
                    self.capital > (row["bid_px_00"] + 0.1 * strike) * sz
                )  # based on bid price
        else:  # call options
            open_price = self.get_underlying_opening(row["day"])
            if order["action"] == "B":  # buying a call
                return (
                    self.capital > (row["ask_px_00"] + 0.1 * open_price) * sz
                )  # based on ask price
            else:  # selling a call
                return (
                    self.capital > (row["bid_px_00"] + 0.1 * open_price) * sz
                )  # based on bid price

    def get_expiration_date(self, symbol) -> str:
        """
        Input: SPX   240419C00800000
        Output: outputs 2024-04-19
        """

        numbers: str = symbol.split(" ")[3]
        date: str = numbers[:6]
        date_yymmdd: str = "20" + date[0:2] + "-" + date[2:4] + "-" + date[4:6]
        return date_yymmdd

    def price_after_fees(self, row):
        """
        Input: Takes in row of options (which has price) and takes into account the 0.1% slippage and $0.50 transaction costs
        Output: Price after fees
        """

        real_bid = row["bid_px_00"] * 0.999 - 0.5
        real_ask = row["ask_px_00"] * 1.001 + 0.5

        return real_bid, real_ask

    def string_to_date(self, string):
        """
        Takes string in the form 2024-01-02, converts to datetime object
        """
        year, month, day = string.split("-")
        return datetime(int(year), int(month), int(day))

    def get_current_date(self, row):
        """
        Get current date of an order (to the nearest day)
        """
        date_time = row["ts_recv"]
        date = date_time[: date_time.index("T")]
        return self.string_to_date(date)

    def time_to_expiration(self, row):
        """
        Input: row of options
        Output: number of years to expire

        """

        current_date = self.get_current_date(row)
        expire_date1 = self.get_expiration_date(row["symbol"])
        expire_date = self.string_to_date(expire_date1)

        difference = expire_date - current_date

        return difference.days / 365

    def expectancy(self, fair_call, fair_put, call_put, real_bid, real_ask):
        """
        fair is [fair_call, fair_put]
        call_put is whether option is call or put
        real_prices is [real_bid, real_ask]

        Output: tuple of expected value of buying and selling
        """
        if call_put == "C":  # working with call options
            return (fair_call - real_ask, real_bid - fair_call)
        if call_put == "P":  # working with put options
            return (fair_put - real_ask, real_bid - fair_put)

    def get_current_date_accurate(self, row):
        """
        Output: gets the accurate date (hour, minutes) in datetime object
        """
        date_time = row["ts_recv"]
        first = date_time.split(".")[0] + "Z"
        accurate_date = datetime.strptime(first, "%Y-%m-%dT%H:%M:%SZ")
        return accurate_date

    # def get_underlying_price_close(self, current_date_accurate):
    #     """
    #     Input: Current Date to the nearest minute
    #     Output: Gets the open price of the underlying at the closest time before or equal to current_date_accurate.

    #     Need to make this fast it seems... the way I do it is match dates first,
    #     then iterate through hours until you get the closest hour rounded down.
    #     Can't think of a faster way.
    #     """

    #     closest_time = None

    #     current_date = current_date_accurate.date()

    #     matching_dates = self.underlying[
    #         self.underlying["date"].str.startswith(str(current_date))
    #     ]

    #     for index in range(len(matching_dates)):
    #         time_str = matching_dates.iloc[index]["date"]
    #         time = time_str.split(" ")[0] + " " + time_str.split(" ")[1].split("-")[0]
    #         underlying_time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

    #         if underlying_time <= current_date_accurate:
    #             closest_time = underlying_time
    #             open_price = matching_dates.iloc[index]["open"]
    #         else:
    #             break

    #     return open_price

    # def get_underlying_price_close(self, current_date_accurate):
    #     """
    #     Input: Current Date to the nearest minute
    #     Output: Gets the open price of the underlying at the closest time before or equal to current_date_accurate.
    #     """

    #     open_price = None
    #     closest_time = None

    #     underlying_dates = self.underlying["date"]
    #     num_rows = len(underlying_dates)

    #     for index in range(num_rows):
    #         time_str = underlying_dates.iloc[index]
    #         time = time_str.split(" ")[0] + " " + time_str.split(" ")[1].split("-")[0]
    #         underlying_time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

    #         if underlying_time <= current_date_accurate:
    #             closest_time = underlying_time
    #             open_price = self.underlying["open"].iloc[index]
    #         else:
    #             break

    #     return open_price

    # def get_underlying_price_close(self, current_date_accurate):
    #     """
    #     Input: Current Date to the nearest minute
    #     Output: Gets the open of the price with the current date rounded down to nearest hour (can make faster?)
    #     """

    #     open_price = None
    #     closest_time = None

    #     for index, time_str in enumerate(
    #         self.underlying["date"]
    #     ):  # THIS IS HORRIBLE its slowing down time a LOT. for loop throughout the whole underlying price close to get the closest.
    #         time = time_str.split(" ")[0] + " " + time_str.split(" ")[1].split("-")[0]
    #         underlying_time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    #         if underlying_time <= current_date_accurate:
    #             if closest_time is None or underlying_time > closest_time:
    #                 closest_time = underlying_time
    #                 open_price = self.underlying["open"].iloc[index]

    #     return open_price

    def get_spot_price(self, row):
        """
        Input: row of options
        Output: Spot price (closest to when the option is offered by the hour)
        """

        current_date = self.get_current_date_accurate(row)
        spot_price = self.get_underlying_price_close(current_date)
        return spot_price

    def handle_expiring_orders(self, curr_date, expiry_dates):
        """
        Handles the orders expiring on curr_date, making changes to
        capital. This function is only called a few times (on expiration dates),
        so can be slow
        """
        print("ENTERING METHOD, CAPITAL: " + str(self.capital))
        sells = 0
        buys = 0
        string_date = curr_date.strftime("%Y-%m-%d")
        matching_dates = self.underlyinghour[
            self.underlyinghour["date"].str.startswith(str(string_date))
        ]
        daily_open = matching_dates.iloc[0]["open"]
        for order in expiry_dates[curr_date]:
            expiry, call_put, strike = self.parse_option_symbol(order["option_symbol"])
            # since orders are expiring, remove their delta/vega
            # this order dictionary has greeks in it
            self.port_delta -= order["delta"]
            self.port_vega -= order["vega"]
            if call_put == "C" and daily_open > strike:  # call is ITM
                if order["action"] == "B":
                    self.capital += (daily_open - strike) * order["order_size"]
                    buys += 1
                if order["action"] == "S":
                    self.capital -= (daily_open - strike) * order["order_size"]
                    sells += 1
            if call_put == "P" and daily_open < strike:
                if order["action"] == "B":
                    self.capital += (strike - daily_open) * order["order_size"]
                    buys += 1
                if order["action"] == "S":
                    self.capital -= (strike - daily_open) * order["order_size"]
                    sells += 1
        # print ("FRACTION SALES: " + str(sells/(sells + buys)))
        print("LEAVING METHOD, CAPITAL: " + str(self.capital))
        del expiry_dates[curr_date]  # very important, so function is not called again

    def last_row_datetime(self):
        last_row = current_date = self.get_current_date(self.options.iloc[-1])
        print(last_row)
        return last_row

    def generate_orders(self) -> pd.DataFrame:
        last_timestamp = self.last_row_datetime()
        orders = []
        number_parsed = 0
        number_of_orders = 0
        expiry_dates = (
            {}
        )  # dictionary of keys (expiry dates) to list of orders on that date
        vol = 0.13
        vol_count = 0
        num_vol = 0
        trading_on = False  # whether we are using the order for trading (True), or finding vol (False)
        tot_ev = 0
        tot_beats = 1
        tot_ev_sq = 0
        for i, row in self.options.iterrows():
            buying = True  # whether we can buy order at current row
            selling = True  # whether we can sell order at current row
            if i % 10000 == 0:
                trading_on = False
                vol_count = 0
                num_vol = 0
            if (i - 100) % 10000 == 0:
                trading_on = True
                try:
                    vol = vol_count / num_vol
                except:
                    vol = vol  # just keep vol the same if divide by 0 error
                print("trading on, vol is: " + str(vol))
                print("capital is: " + str(self.capital))
                print("AVERAGE +EV: " + str(tot_ev / tot_beats))
                print("ORDER NUMBER: " + str(i))
                print("# ORDERS TAKEN: " + str(len(orders)))
            mid_price = (row["bid_px_00"] + row["ask_px_00"]) / 2
            curr_date = self.get_current_date(row)
            if curr_date in expiry_dates:
                print("I IS: " + str(i))
                self.handle_expiring_orders(curr_date, expiry_dates)
            real_bid, real_ask = self.price_after_fees(row)
            t = self.time_to_expiration(row)
            r = 0.03  # risk-free rate
            expiry, action, strike_price = self.parse_option_symbol(row["symbol"])
            if expiry > last_timestamp:
                continue
            try:
                spot_price = self.get_spot_price(row)
            except:
                print("ERROR: " + str(row["ts_recv"]))
                raise ValueError
            if not trading_on:
                this_vol = self.find_vol(
                    mid_price, spot_price, strike_price, r, t, action
                )
                if not math.isnan(this_vol):
                    vol_count += this_vol
                    num_vol += 1
                continue
            # VERY aggressive hedging strategy (would not be optimal IRL), but want to test it
            # we only do things that get our greeks closer to 0
            # print ("CURRENT DELTA: " + str(self.port_delta))
            # print ("CURRENT VEGA: " + str(self.port_vega))
            if self.port_delta > 0 and self.port_vega > 0:  # should only sell calls
                if action == "P":
                    continue
                buying = False
            if self.port_delta > 0 and self.port_vega < 0:  # should only buy puts
                if action == "C":
                    continue
                selling = False
            if self.port_delta < 0 and self.port_vega > 0:  # should only sell puts
                if action == "C":
                    continue
                buying = False
            if self.port_delta < 0 and self.port_vega < 0:  # should only buy calls
                if action == "P":
                    continue
                selling = False
            fair_call, fair_put = self.black_scholes(
                spot_price, strike_price, r, vol, t
            )  # NEED TO GET PROPER SPOT PRICE (done i think but too slow) AND VOLATILITY function?

            contract_delta = self.delta(spot_price, strike_price, r, vol, t, action)
            contract_vega = self.vega(spot_price, strike_price, r, vol, t)

            e_buying, e_selling = self.expectancy(
                fair_call, fair_put, action, real_bid, real_ask
            )
            this_beat = max(e_buying, e_selling)  # best expectancy
            if this_beat < 0:  # if both sides bad, do nothing
                continue
            tot_beats += 1
            tot_ev += this_beat
            tot_ev_sq += this_beat**2
            avg_beat = tot_ev / tot_beats
            sd_beats = np.sqrt((tot_ev_sq / tot_beats) - (avg_beat**2))

            if e_buying > 0 and e_buying > (avg_beat + 1.5 * sd_beats) and buying:
                order = {
                    "datetime": row["ts_recv"],
                    "option_symbol": row["symbol"],
                    "action": "B",
                    "order_size": row["ask_sz_00"],
                }

                number_parsed += 1

            elif e_selling > 0 and e_selling > (avg_beat + 1.5 * sd_beats) and selling:
                order = {
                    "datetime": row["ts_recv"],
                    "option_symbol": row["symbol"],
                    "action": "S",
                    "order_size": row["bid_sz_00"],
                }

                number_parsed += 1

            else:
                # no_action_count += 1
                # print("didn't do anything:" + str(no_action_count))
                number_parsed += 1

                continue

            if expiry in expiry_dates:
                if len(expiry_dates[expiry]) >= 10:
                    continue

            if self.margin_ok(row, order):
                if order["action"] == "B":  # need a way of updating our capital,
                    #     # not sure how to do this properly this is defo wrong
                    self.capital -= order["order_size"] * real_ask

                else:  # action is selling
                    self.capital += order["order_size"] * real_bid

                orders.append(order)

                # updating delta, vega in our portfolio
                total_delta = contract_delta * order["order_size"]
                total_vega = contract_vega * order["order_size"]
                if order["action"] == "B":
                    self.port_delta += total_delta
                    self.port_vega += total_vega
                elif order["action"] == "S":
                    self.port_delta -= total_delta
                    self.port_vega -= total_vega

                order_greeks = {**order, "delta": total_delta, "vega": total_vega}

                if expiry in expiry_dates:
                    expiry_dates[expiry].append(order_greeks)
                else:
                    expiry_dates[expiry] = [order_greeks]

                number_of_orders += 1

                print(
                    "# of orders: " + str(number_of_orders),
                    "# parsed: " + str(number_parsed),
                    "#total: " + str(i + 1),
                    "ratio of orders that we take:  " + str(number_of_orders / (i + 1)),
                    "capital: " + str(self.capital),
                ),  # We don't parse every order anymore

        return pd.DataFrame(orders)
