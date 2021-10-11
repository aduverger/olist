from olist.data import Olist
from olist.order import Order
from olist.seller import Seller
import numpy as np


class Product:

    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.matching_table = olist.get_matching_table()
        self.order = Order()

    def get_product_features(self):
        """
        Returns a DataFrame with:
       'product_id', 'product_category_name', 'product_name_length',
       'product_description_length', 'product_photos_qty', 'product_weight_g',
       'product_length_cm', 'product_height_cm', 'product_width_cm'
        """

        products = self.data['products']

        en_category = self.data['product_category_name_translation']
        df = products.merge(en_category, on='product_category_name')
        df.drop(['product_category_name'], axis=1, inplace=True)
        df.rename(columns={'product_category_name_english': 'category',
                  'product_name_lenght': 'product_name_length',
                  'product_description_lenght': 'product_description_length'},
                  inplace=True)


        return df

    def get_price(self):
        """
        Return a DataFrame with:
        'product_id', 'price'
        """
        order_items = self.data['order_items']
        # There are many different order_items per product_id, each with different prices. Take the mean of various prices
        return order_items[['product_id', 'price']].groupby('product_id').mean()

    def get_wait_time(self):
        """
        Returns a DataFrame with:
        'product_id', 'wait_time'
        """
        matching_table = self.matching_table
        orders_wait_time = self.order.get_wait_time()

        df = matching_table.merge(orders_wait_time,
                                 on='order_id')

        return df.groupby('product_id',
                          as_index=False).agg({'wait_time': 'mean'})

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'product_id', 'share_of_five_stars', 'share_of_one_stars',
        'review_score'
        """
        matching_table = self.matching_table
        orders_reviews = self.order.get_review_score()

        # Since the same products can appear multiple times in the same
        # order, create a product <> order matching table

        matching_table = matching_table[['order_id',
                                         'product_id']].drop_duplicates()
        df = matching_table.merge(orders_reviews,
                                  on='order_id')
        df = df.groupby('product_id',
                        as_index=False).agg({'dim_is_one_star': 'mean',
                                             'dim_is_five_star': 'mean',
                                             'review_score': 'mean',
                                             })
        df.columns = ['product_id', 'share_of_one_stars',
                      'share_of_five_stars', 'review_score']

        return df

    def get_revenues(self):
        """
        Returns a DataFrame with:
        'product_id', 'revenues'
        Revenue: 
            Olist takes a 10% cut on the product price (excl. freight) of each order delivered.
        """
        # get 10% cut
        revenues = Product().get_sales().copy()
        revenues.loc[:, 'sales'] = revenues['sales'].map(lambda x: x/10)
        revenues.rename(columns={'sales': 'revenues'}, inplace=True)
        # get subscription
        #dates = Seller().get_active_dates().copy()
        #revenues['subscription'] = dates['active_months'] * 80
        # sum cut and subscription
        #revenues['revenues'] = revenues['sales'] + revenues['subscription']
        return revenues[['product_id', 'revenues']]

    def get_costs(self):
        """
        Returns a DataFrame with:
        'product_id', 'share_of_five_stars', 'share_of_one_stars',
        'review_score'
        """
        matching_table = self.matching_table
        orders_reviews = self.order.get_review_score()

        # Since the same products can appear multiple times in the same
        # order, create a product <> order matching table

        matching_table = matching_table[['order_id',
                                         'product_id']].drop_duplicates()
        df = matching_table.merge(orders_reviews,
                                  on='order_id')
        
        # Get cost from bad reviews
        def review_costs(review):
            if review <= 2:
                return 75/review
            elif review == 3:
                return 30
            return 0
        
        df['costs'] = df['review_score'].map(review_costs)
        df = df.groupby(
            'product_id', as_index=False).sum()
        return df[['product_id', 'costs']]
        # Get cost from IT
        #total_cost = 500_000
        #seller_quantity = Seller().get_quantity()[['seller_id', 'n_orders']].copy()
        #order_cost = total_cost / np.sum(np.sqrt(seller_quantity['n_orders']))
        #product_quantity = Product().get_quantity()[['product_id', 'n_orders']].copy()
        #product_quantity['IT_costs'] = np.sqrt(product_quantity['n_orders']) * order_cost
        
        # Sum IT cost and review cost
        #df = df.merge(product_quantity, on='product_id')
        #df['costs'] = df['IT_costs'] + df['review_costs']
        #return df[['product_id', 'costs']]

    def get_profits(self):
        """Returns a DataFrame with:
        'product_id', 'profits'
        profits = revenues - costs
        """
        costs = Product().get_costs()
        revenues = Product().get_revenues()
        revenues['profits'] = revenues['revenues'] - costs['costs']
        return revenues[['product_id', 'profits']]

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'product_id', 'n_orders', 'quantity'
        """
        order_items = self.data['order_items']

        n_orders =\
            order_items.groupby('product_id')['order_id'].nunique().reset_index()
        n_orders.columns = ['product_id', 'n_orders']

        quantity = \
            order_items.groupby('product_id',
                                   as_index=False).agg({'order_id': 'count'})
        quantity.columns = ['product_id', 'quantity']

        return n_orders.merge(quantity, on='product_id')

    def get_sales(self):
        """
        Returns a DataFrame with:
        'product_id', 'sales'
        """
        return self.data['order_items'][['product_id', 'price']]\
            .groupby('product_id', as_index=False)\
            .sum()\
            .rename(columns={'price': 'sales'})

    def get_training_data(self):

        training_set =\
            self.get_product_features()\
                .merge(
                self.get_wait_time(), on='product_id'
               ).merge(
                self.get_price(), on='product_id'
               ).merge(
                self.get_review_score(), on='product_id'
               ).merge(
                self.get_costs(), on='product_id'
               ).merge(
                self.get_revenues(), on='product_id'
               ).merge(
                self.get_profits(), on='product_id'
               ).merge(
                self.get_quantity(), on='product_id'
               ).merge(
                self.get_sales(), on='product_id'
               )

        return training_set
