import os
import pandas as pd


class Olist:

    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrame loaded from csv files
        """

        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/csv')        
        file_names = [csv_file for csv_file in os.listdir(csv_path) if csv_file.endswith('.csv')]
        key_names = [file_name.replace('.csv', '') \
                            .replace('_dataset', '') \
                            .replace('olist_', '') for file_name in file_names]
        data = {}
        for k, v in zip(key_names, file_names):
            data[k] = pd.read_csv(os.path.join(csv_path, v))
        return data

    def get_matching_table(self):
        """
        01-01 > This function returns a matching table between
        columns [ "order_id", "review_id", "customer_id", "product_id", "seller_id"]
        """
        data = Olist().get_data()
        orders = data['orders'][['order_id', 'customer_id']]
        order_reviews = data['order_reviews'][['review_id', 'order_id']]
        order_items = data['order_items'][['order_id', 'seller_id', 'product_id']]
        df = order_items.merge(order_reviews, how='outer', on='order_id') \
                        .merge(orders, how='outer', on='order_id')
        return df

    def ping(self):
        """
        You call ping I print pong.
        """
        print('pong')
