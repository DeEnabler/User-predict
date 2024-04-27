import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_trader_data(num_traders=20):
    np.random.seed(42)
    trader_data = []
    
    for i in range(num_traders):
        trader_id = i + 1
        num_transactions = np.random.randint(50, 100)
        
        for _ in range(num_transactions):
            price = round(np.random.uniform(10, 1000), 2)
            amount = np.random.randint(1, 100)
            date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            marketcap = round(np.random.uniform(1000000, 10000000), 2)
            
            trader_data.append({'trader_id': trader_id, 'price': price, 'amount': amount, 'date': date, 'marketcap': marketcap})
    
    df = pd.DataFrame(trader_data)
    return df

