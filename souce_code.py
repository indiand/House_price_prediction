import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle 
data = pd.read_csv("Delhi house data1.csv")
#print(data.head(5))
print(data.columns)
x = data[['Area ' , 'BHK' , 'Bathroom', 'Parking']]
y = data['Price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state= 42)
model = LinearRegression()
model.fit(x_train,y_train)
with open("model.pkl","wb") as f:
 pickle.dump(model,f)
print("Model is successfully created and saved") 
#Load model
with open("model.pkl","rb") as f :
    model = pickle.load(f)
area  = float(input("Enter the area(sqft) : " ))
bhk = int(input("Enter the BHK : "))
bathroom = int(input("Enter the Bathroom :" ))
parking = int(input("Enter the no.of Parking :" ))
#prediction
prediction = model.predict([[area,bhk,bathroom,parking]])
print("Predicted house price: ",round(prediction[0],2))                 
     