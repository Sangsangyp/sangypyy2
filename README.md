import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# 1. ĐỌC DỮ LIỆU
import pandas as pd

DATA_PATH = r"C:/Users/ADMIN/Downloads/airfoil+self+noise/airfoil_self_noise.dat"

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)

df.columns = ["Frequency", "Angle", "Chord", "Velocity", "Thickness", "SSPL"]

print(df.head())

# Biểu đồ 1: Chord vs Velocity (màu theo SSPL)
plt.scatter(df["Chord"], df["Velocity"], c=df["SSPL"], cmap="viridis")
plt.colorbar(label="SSPL (dB)")
plt.xlabel("Chord (m)")
plt.ylabel("Velocity (m/s)")
plt.title("(b1) Chord vs Velocity – màu theo SSPL")
plt.grid(True)
plt.show()

# Biểu đồ 2: Velocity vs SSPL (màu theo SSPL)
plt.scatter(df["Velocity"], df["SSPL"], c=df["SSPL"], cmap="inferno")
plt.colorbar(label="SSPL (dB)")
plt.xlabel("Velocity (m/s)")
plt.ylabel("SSPL (dB)")
plt.title("(b2) Velocity vs SSPL – màu theo SSPL")
plt.grid(True)
plt.show()

#Huấn luyện
# ---------------------------------------------------------------
X = df[["Frequency", "Angle", "Chord", "Velocity", "Thickness"]]
y = df["SSPL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Dùng Linear Regression + chuẩn hóa
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

model.fit(X_train, y_train)

#Hiển thị kqua
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE trên tập test =", mse)

#Tói ưu mo hinh
mse_target = np.var(y_test) * 0.10   

print("Ngưỡng MSE mục tiêu (10% variance) =", mse_target)

if mse <= mse_target:
    print(" MSE đạt yêu cầu < 10%")
else:
    print(" MSE chưa đạt yêu cầu")
