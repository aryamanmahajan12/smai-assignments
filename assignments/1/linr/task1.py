import matplotlib.pyplot as plt
import pandas as pd

csv_file = 'train.csv' 
val_file = 'val.csv'
test_file = 'test.csv'

data = pd.read_csv(csv_file)
val_data = pd.read_csv(val_file)
test_data = pd.read_csv(test_file)


x = data['x']  
y = data['y']  


x_val = val_data['x']  
y_val = val_data['y']  

x_test = test_data['x']  
y_test = test_data['y']  

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='b',label='train')
plt.scatter(x_val,y_val,color='red',label='validate')
plt.scatter(x_test,y_test,color='g',label= 'test') 
plt.xlabel('X Axis Label')  
plt.ylabel('Y Axis Label')  
plt.legend()
plt.title('X vs Y Scatter Plot')  
plt.show()
