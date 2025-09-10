def create_underlying_dictionary(self):
    dictionary = dict(
        zip(pd.to_datetime(self.underlying["date"]), self.underlying["open"])
    )
    final_dict = {key.to_pydatetime(): value for key, value in dictionary.items()}
    final_final = {key.replace(tzinfo=None): value for key, value in final_dict.items()}
    return final_final


def get_underlying_price_close(self, current_date_accurate):
    """
    Input: Current Date to the nearest minute
    Output: Gets the open price of the underlying at the closest time before or equal to current_date_accurate.

    Need to make this fast it seems... the way I do it is match dates first,
    then iterate through hours until you get the closest hour rounded down.
    Can't think of a faster way.
    """

    if current_date_accurate.hour >= 21:
        compare = current_date_accurate.replace(
            hour=15,
            minute=30,
            second=0,
            microsecond=0,
        )
        spot_price = self.dict[compare]
    elif current_date_accurate.minute >= 30:
        # print("DATE RIGHT NOW: " + str(current_date_accurate))
        compare = current_date_accurate.replace(
            hour=current_date_accurate.hour - 5,
            minute=30,
            second=0,
            microsecond=0,
        )

        spot_price = self.dict[compare]
    else:
        spot_price = self.dict[
            current_date_accurate.replace(
                hour=current_date_accurate.hour - 6,
                minute=30,
                second=0,
                microsecond=0,
            )
        ]
    return spot_price
