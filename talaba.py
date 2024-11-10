import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Talabalar ma'lumotlari to'plamini yaratish
data = {
    'data_size': ['kichik', 'katta', 'kichik', 'ortacha', 'katta'],
    'complexity': ['yuqori', 'past', 'past', 'yuqori', 'ortacha'],
    'speed': ['ortacha', 'yuqori', 'juda yuqori', 'ortacha', 'yuqori'],
    'recommended_db': ['Relational malumotlar bazasi', 'NoSQL', 'Xotira asosida saqlash', 'Obyektga yonaltirilgan', 'NoSQL']
}

# Pandas DataFrame ga o‘tkazish
df = pd.DataFrame(data)

# LabelEncoder yordamida kiritilgan xususiyatlarni raqamli formatga aylantirish
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Xususiyatlar (X) va nishon (y)
X = df[['data_size', 'complexity', 'speed']]
y = df['recommended_db']

# Neyron tarmoq modelini yaratish (MLPClassifier)
model = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=1000)

# Modelni o‘rgatish
model.fit(X, y)

# Tavsiya olish funksiyasi
def get_recommendation():
    # Foydalanuvchi kiritmalarini olish
    data_size = data_size_var.get()
    complexity = complexity_var.get()
    speed = speed_var.get()

    # Kiritmalarni model uchun kerakli formatga aylantirish
    input_data = pd.DataFrame([[data_size, complexity, speed]], columns=['data_size', 'complexity', 'speed'])
    for column in input_data.columns:
        input_data[column] = label_encoders[column].transform(input_data[column])

    # Model yordamida tavsiya olish
    prediction = model.predict(input_data)
    recommended_db = label_encoders['recommended_db'].inverse_transform(prediction)

    # Natijani interfeysda ko'rsatish
    result_label.config(text=f"Tavsiya etilgan saqlash usuli: {recommended_db[0]}")

# Interfeysni yaratish
app = tk.Tk()
app.title("Bilimlar Bazasi Saqlash Usullari Tavsiya Dasturi")
app.geometry("400x300")

# Parametrlarni tanlash uchun interfeys elementlari
ttk.Label(app, text="Ma'lumot hajmi").pack(pady=5)
data_size_var = tk.StringVar()
data_size_combo = ttk.Combobox(app, textvariable=data_size_var)
data_size_combo['values'] = ('kichik', 'ortacha', 'katta')
data_size_combo.pack()

ttk.Label(app, text="Ma'lumotlar murakkabligi").pack(pady=5)
complexity_var = tk.StringVar()
complexity_combo = ttk.Combobox(app, textvariable=complexity_var)
complexity_combo['values'] = ('past', 'ortacha', 'yuqori')
complexity_combo.pack()

ttk.Label(app, text="Ishlash tezligi").pack(pady=5)
speed_var = tk.StringVar()
speed_combo = ttk.Combobox(app, textvariable=speed_var)
speed_combo['values'] = ('past', 'ortacha', 'yuqori', 'juda yuqori')
speed_combo.pack()

# Natijani ko'rsatish tugmasi
ttk.Button(app, text="Tavsiya olish", command=get_recommendation).pack(pady=20)

# Natija uchun label
result_label = ttk.Label(app, text="")
result_label.pack(pady=5)

# Dastur ishga tushirilishi
app.mainloop()
