import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("==========================================")
print(" SCHOLARSHIP ELIGIBILITY PREDICTION SYSTEM ")
print("==========================================")


X = np.array([
    [85, 200000, 90],
    [60, 450000, 75],
    [92, 180000, 95],
    [55, 600000, 65],
    [78, 300000, 85],
    [88, 250000, 92],
    [50, 700000, 60],
    [90, 220000, 93]
])


y = np.array([1, 0, 1, 0, 1, 1, 0, 1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy : {accuracy*100:.2f}%")
print("------------------------------------------")


new_student = [[82, 240000, 88]]
probability = model.predict_proba(new_student)[0][1] * 100
result = model.predict(new_student)[0]

print("STUDENT DETAILS")
print("Marks      : 82")
print("Income     : 240000")
print("Attendance : 88%")
print("------------------------------------------")

print(f"Eligibility Probability : {probability:.2f}%")

if result == 1:
    print("Final Decision : Eligible for Scholarship")
else:
    print("Final Decision : Not Eligible")

print("==========================================")
