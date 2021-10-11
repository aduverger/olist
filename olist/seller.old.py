import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:

    def __init__(self):
        # Import only data once
        olist = Olist()
        self.data = olist.get_data()
        self.matching_table = olist.get_matching_table()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
       'seller_id', 'seller_city', 'seller_state'
        """
        return self.data['sellers'].copy().drop(columns=['seller_zip_code_prefix'])

    def get_seller_delay_wait_time(self):
        """
        Returns a DataFrame with:
       'seller_id', 'delay_to_carrier', 'seller_wait_time'
        """
        orders = self.data['orders'][[
            'order_id',
            'order_delivered_carrier_date',
            'order_purchase_timestamp',
            'order_delivered_customer_date'
        ]].copy()
        items = self.data['order_items'][['order_id', 'seller_id', 'shipping_limit_date']].copy()
        items = items.merge(orders, on='order_id')
        
        #compute to date
        items.loc[:,'shipping_limit_date'] = pd.to_datetime(items.loc[:,'shipping_limit_date'])
        items.loc[:,'order_delivered_carrier_date'] = pd.to_datetime(items.loc[:,'order_delivered_carrier_date'])
        items.loc[:,'order_purchase_timestamp'] = pd.to_datetime(items.loc[:,'order_purchase_timestamp'])
        items.loc[:,'order_delivered_customer_date'] = pd.to_datetime(items.loc[:,'order_delivered_customer_date'])

        def handle_delay(x):
                    if x > 0:
                        return x
                    return 0
        #compute wait_time
        items['wait_time'] = \
                    (items['order_delivered_customer_date'] -
                    items['order_purchase_timestamp']) / np.timedelta64(24, 'h')
        # compute delay_to_carrier
        items['delay_to_carrier'] = (items['order_delivered_carrier_date'] - \
                                    items['shipping_limit_date']) / np.timedelta64(24, 'h')
        items.loc[:, 'delay_to_carrier'] = items['delay_to_carrier'].apply(handle_delay)
        
        return items.groupby("seller_id", as_index=False).mean()

    def get_active_dates(self):
        """
        Returns a DataFrame with:
       'seller_id', 'date_first_sale', 'date_last_sale'
        """
        orders2 = self.data['orders'][['order_id', 'order_purchase_timestamp']].copy()
        items2 = self.data['order_items'][['order_id', 'seller_id']].copy()
        orders2.loc[:,'order_purchase_timestamp'] = \
                    pd.to_datetime(orders2.loc[:,'order_purchase_timestamp'])
        items2 = items2.merge(orders2, on='order_id')

        dates = items2[['seller_id']] \
                .groupby('seller_id', as_index=False) \
                .count().copy()
        # get min
        min_ = items2.groupby('seller_id', as_index=False) \
                .min() \
                .drop(columns=['order_id']) \
                .rename(columns={'order_purchase_timestamp': 'date_first_sale'})
        # get max
        max_ = items2.groupby('seller_id', as_index=False) \
                .max() \
                .drop(columns=['order_id']) \
                .rename(columns={'order_purchase_timestamp': 'date_last_sale'})
        
        return dates.merge(min_, on='seller_id').merge(max_, on='seller_id')

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars',
        'review_score'
        """
        items3 = self.data['order_items'][['order_id', 'seller_id']].copy()
        stars = self.order.get_training_data()[[
            'order_id',
            'dim_is_five_star',
            'dim_is_one_star',
            'review_score'
        ]].copy()
        return stars \
                .merge(items3, on="order_id") \
                .groupby(by="seller_id", as_index=False) \
                .mean() \
                .rename(columns={'dim_is_five_star': 'share_of_five_stars', 'dim_is_one_star': 'share_of_one_stars'})

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        items4 = self.data['order_items'][[
            'order_id',
            'order_item_id',
            'product_id',
            'seller_id'
        ]].copy()

        # n_orders
        quantity = items4[['seller_id', 'order_id']] \
            .groupby('seller_id', as_index=False) \
            .nunique() \
            .rename(columns={'order_id': 'n_orders'})

        # quantity
        quantity = quantity.merge(
            items4[['seller_id', 'product_id']] \
                .groupby('seller_id', as_index=False) \
                .count() \
                .rename(columns={'product_id': 'quantity'}),
            on='seller_id'
        )

        # quantity_per_order
        quantity['quantity_per_order'] = quantity['quantity'] / quantity['n_orders']
        
        return quantity

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        items5 = self.data['order_items'][[
            'price',
            'seller_id'
        ]].copy()
        return items5 \
            .groupby('seller_id', as_index=False) \
            .sum() \
            .rename(columns={'price': 'sales'})

    def get_training_data(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_state', 'seller_city', 'delay_to_carrier',
        'wait_time', 'share_of_five_stars', 'share_of_one_stars',
        'seller_review_score', 'n_orders', 'quantity', 'date_first_sale', 'date_last_sale', 'sales'
        """
        seller_features = Seller().get_seller_features()
        wait_time = Seller().get_seller_delay_wait_time()
        active_dates = Seller().get_active_dates()
        review_score = Seller().get_review_score()
        quantity = Seller().get_quantity()
        sales = Seller().get_sales()
        df = seller_features.merge(wait_time, on='seller_id') \
                            .merge(active_dates, on='seller_id') \
                            .merge(review_score, on='seller_id') \
                            .merge(quantity, on='seller_id') \
                            .merge(sales, on='seller_id')
        return df.dropna()
        
        


