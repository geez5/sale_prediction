from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class SalesPredictor:
    def __init__(self):
        self.model = LinearRegression()
        
    def prepare_features(self, df):
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['DayOfYear'] = df['Date'].dt.dayofyear
        return df
        
    def train(self, df):
        features = ['DayOfWeek', 'Month', 'Year', 'DayOfYear']
        X = df[features]
        y = df['Sales']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return metrics
        
    def predict(self, X):
        return self.model.predict(X)
