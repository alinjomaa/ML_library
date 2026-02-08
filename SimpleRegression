
class SimpleLinearRegression:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def fit(self, x_data, y_data):
        y_bar = sum(y_data) / len(y_data)
        x_bar = sum(x_data) / len(x_data)

        numerator = 0
        denominator = 0

        for i in range(len(x_data)):
            numerator += (x_data[i] - x_bar) * (y_data[i] - y_bar) # for each point i, it preforms this calculation and adds it to the numerator set above 
            denominator += (x_data[i] - x_bar) ** 2

        b1 = numerator / denominator
        b0 = y_bar - b1 * x_bar

        return b0, b1

    def predict(self, b0, b1, x_new):
        y = b0 + b1 * x_new
        return y


x_data = [1, 2, 3, 4, 5]
y_data = [2, 4, 6, 8, 10]


model = SimpleLinearRegression(x_data, y_data)


b0_fitted, b1_fitted = model.fit(x_data, y_data)


print(f"Fitted coefficients: b0 (intercept) = {b0_fitted}, b1 (slope) = {b1_fitted}")


x_new_value = 6
prediction = model.predict(b0_fitted, b1_fitted, x_new_value)
print(f"Prediction for x = {x_new_value}: y = {prediction}")
