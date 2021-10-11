import os
import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''

    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        02-01 > Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filtering out non-delivered orders unless specified
        """
        orders = self.data['orders'].copy()
        # Filter dataframe on delivered orders
        orders = orders.query("order_status == 'delivered'").copy()
        # handle datetime
        orders.order_purchase_timestamp = \
                    pd.to_datetime(orders.order_purchase_timestamp)
        orders.order_delivered_customer_date =\
                    pd.to_datetime(orders.order_delivered_customer_date)
        orders.order_estimated_delivery_date =\
                    pd.to_datetime(orders.order_estimated_delivery_date)
        # compute wait time
        orders['wait_time'] = \
                (orders.order_delivered_customer_date - orders.order_purchase_timestamp) / np.timedelta64(24, 'h')
        # compute expected wait time
        orders['expected_wait_time'] = \
                (orders.order_estimated_delivery_date - orders.order_purchase_timestamp) / np.timedelta64(24, 'h')
        # compute delay vs expected - carefully handles "negative" delays
        delay = ((orders.order_delivered_customer_date - orders.order_estimated_delivery_date) / np.timedelta64(24, 'h')).copy()
        delay[delay < 0] = 0
        orders['delay_vs_expected'] = delay
        return orders[[
            'order_id',
            'wait_time',
            'expected_wait_time',
            'delay_vs_expected',
            'order_status'
        ]]

    def get_review_score(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        # Function to map five stars and one star
        def dim_five_star(d):
            return 1 if d == 5 else 0
        def dim_one_star(d):
            return 1 if d == 1 else 0
        # Compute dim_is_xx_star 
        reviews = self.data['order_reviews'].copy()
        reviews["dim_is_five_star"] = reviews["review_score"].map(dim_five_star)
        reviews["dim_is_one_star"] = reviews["review_score"].map(dim_one_star)
        return reviews[[
            'order_id',
            'dim_is_five_star',
            'dim_is_one_star',
            'review_score'
        ]]

    def get_number_products(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_products
        """
        n_products = self.data['order_items'].copy()
        # Keep only relevant columns
        n_products = n_products[['order_id', 'order_item_id']]
        n_products.rename(columns={'order_item_id': 'number_of_products'}, inplace=True)
        # Group by order_id and keep the maximum value of order_item_id.
        # For example, if an order has 6 items purchased, you have 6 rows
        # for this order, with order_item_id = [1 to 6].
        # So we need to keep only the max value of order_item_id to get the
        # number of products for the orders
        return n_products.groupby(by="order_id").max().reset_index()

    def get_number_sellers(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_sellers
        """
        n_sellers = self.data['order_items'].copy()
        n_sellers = n_sellers[['order_id', 'seller_id']]
        n_sellers.rename(columns={'seller_id': 'number_of_sellers'}, inplace=True)
        return n_sellers.groupby(['order_id']).nunique().reset_index()

    def get_price_and_freight(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, price, freight_value
        """
        price = self.data['order_items'].copy()
        price = price[['order_id', 'price', 'freight_value']]
        return price.groupby('order_id').sum().reset_index()

    def get_distance_seller_customer(self):
        """
        02-01 > Returns a DataFrame with order_id
        and distance between seller and customer
        """
        # Optional
        # Hint: you can use the haversine_distance logic coded in olist/utils.py

    def get_training_data(self, is_delivered=True,
                          with_distance_seller_customer=False):
        """
        02-01 > Returns a clean DataFrame (without NaN), with the following columns:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status,
        dim_is_five_star, dim_is_one_star, review_score, number_of_products,
        number_of_sellers, price, freight_value, distance_customer_seller]
        """
        wait_time = Order().get_wait_time()
        reviews= Order().get_review_score()
        n_products = Order().get_number_products()
        n_sellers = Order().get_number_sellers()
        prices = Order().get_price_and_freight()
        df = wait_time.merge(reviews, on='order_id') \
                    .merge(n_products, on='order_id') \
                    .merge(n_sellers, on='order_id') \
                    .merge(prices, on='order_id')
        return df.dropna()