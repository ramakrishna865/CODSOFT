
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("advertising.csv")


print(data.head())

plt.figure(figsize=(12, 4))

# TV vs Sales
plt.subplot(131)
plt.scatter(data['TV'], data['Sales'], alpha=0.5)
plt.title('TV vs Sales')
plt.xlabel('TV Advertising')
plt.ylabel('Sales')

# Radio vs Sales
plt.subplot(132)
plt.scatter(data['Radio'], data['Sales'], alpha=0.5)
plt.title('Radio vs Sales')
plt.xlabel('Radio Advertising')
plt.ylabel('Sales')

# Newspaper vs Sales
plt.subplot(133)
plt.scatter(data['Newspaper'], data['Sales'], alpha=0.5)
plt.title('Newspaper vs Sales')
plt.xlabel('Newspaper Advertising')
plt.ylabel('Sales')

plt.tight_layout()
plt.show()

# Preprocess the data
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the predicted vs. actual sales
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

# You can also use the trained model to make predictions for new data
new_data = pd.DataFrame({'TV': [100], 'Radio': [50], 'Newspaper': [20]})
predicted_sales = model.predict(new_data)
print(f"Predicted Sales for New Data: {predicted_sales[0]}")

# Calculating the total advertising expenditure for each channel
total_expenditure = data[['TV', 'Radio', 'Newspaper']].sum()

# Creating a pie chart
plt.figure(figsize=(6, 6))
plt.pie(total_expenditure, labels=total_expenditure.index, autopct='%1.1f%%', startangle=140)
plt.title('Advertising Expenditure Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
# Creating a histogram for sales
plt.figure(figsize=(8, 5))
plt.hist(data['Sales'], bins=15, edgecolor='k', alpha=0.7)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')

plt.show()
