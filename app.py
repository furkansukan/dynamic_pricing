import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Başlık: Uygulamanın başlığı, kullanıcıya dinamik fiyatlandırma uygulamasını tanıtmak amacıyla gösterilir.
st.title("Dinamik Fiyatlandırma Uygulaması")

# Veri Yükleme: Kullanıcıdan veri dosyasını yüklemelerini isteyen alan.
uploaded_file = st.file_uploader("Veri dosyasını yükleyin", type="csv")

if uploaded_file is not None:
    # Yüklenen dosya işleniyor ve pandas DataFrame'e dönüştürülüyor.
    data = pd.read_csv(uploaded_file)
else:
    # Eğer dosya yüklenmezse, varsayılan bir örnek veri dosyası yüklenecek.
    st.warning("Veri yüklenmedi, örnek veri kullanılacak.")
    # Varsayılan örnek veri dosyasını yükleme işlemi.
    data = pd.read_csv("dynamic_pricing.csv")

# İlk birkaç satırı görselleştirme. Kullanıcıya veri hakkında bilgi verir.
st.write(data.head())

# Dinamik Fiyatlandırma Hesaplama: Fiyatların dinamik olarak nasıl belirleneceğini hesaplamak için bazı işlemler yapıyoruz.
if not data.empty:
    # Talep ve Arz Çarpanlarını Hesaplama: Bu adımda talep ve arz seviyelerine göre fiyatları çarparız.
    high_demand_percentile = 75
    low_demand_percentile = 25

    # Talep çarpanı hesaplanıyor, talep yüksekse fiyat artışı uygulanıyor.
    data['demand_multiplier'] = np.where(
        data['Number_of_Riders'] > np.percentile(data['Number_of_Riders'], high_demand_percentile),
        data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], high_demand_percentile),
        data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], low_demand_percentile))

    high_supply_percentile = 75
    low_supply_percentile = 25

    # Arz çarpanı hesaplanıyor, arz düşükse fiyat artışı uygulanıyor.
    data['supply_multiplier'] = np.where(
        data['Number_of_Drivers'] > np.percentile(data['Number_of_Drivers'], low_supply_percentile),
        np.percentile(data['Number_of_Drivers'], high_supply_percentile) / data['Number_of_Drivers'],
        np.percentile(data['Number_of_Drivers'], low_supply_percentile) / data['Number_of_Drivers'])

    # Fiyat Düzenleme Hesaplama: Talep ve arz çarpanlarını, önceki maliyetle çarparak yeni fiyat hesaplanır.
    data['adjusted_ride_cost'] = data['Historical_Cost_of_Ride'] * (
            np.maximum(data['demand_multiplier'], 0.8) * np.maximum(data['supply_multiplier'], 0.8)
    )

    # Dinamik fiyatlandırma sonuçlarını gösteriyoruz.
    st.write("Dinamik Fiyatlandırma Sonuçları:")
    st.write(data[['Historical_Cost_of_Ride', 'adjusted_ride_cost']].head())

# Scatter Plot ve Trendline: Beklenen yolculuk süresi ile geçmiş maliyet arasında ilişkiyi görselleştiriyoruz.
sns.lmplot(x='Expected_Ride_Duration', y='Historical_Cost_of_Ride', data=data)
plt.title('Beklenen Yolculuk Süresi vs. Geçmiş Maliyet')
st.pyplot(plt)

# Box Plot: Araç tipine göre maliyet dağılımını görselleştiriyoruz.
sns.boxplot(x='Vehicle_Type', y='Historical_Cost_of_Ride', data=data)
plt.title('Araç Tipine Göre Maliyet Dağılımı')
st.pyplot(plt)

# 'Vehicle_Type' sütununu sayısal değerlere dönüştürme: Araç tipini sayısal değerlere çeviriyoruz (Premium -> 1, Economy -> 0).
data["Vehicle_Type"] = data["Vehicle_Type"].map({"Premium": 1, "Economy": 0})

# Korelasyon Isı Haritası: Sayısal verilerin birbirleriyle olan ilişkisini görselleştiriyoruz.
numeric_data = data.select_dtypes(include=[np.number])  # Sadece sayısal verileri seçiyoruz
corr_matrix = numeric_data.corr()  # Korelasyon matrisini hesaplıyoruz
plt.figure(figsize=(10, 8))  # Grafik boyutunu ayarlıyoruz
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)  # Isı haritasını çiziyoruz
plt.title('Korelasyon Matrisi')
st.pyplot(plt)

# Model Eğitimi ve Fiyat Tahminleri: Veriler ile makine öğrenimi modelini eğitiyoruz.
x = np.array(data[["Number_of_Riders", "Number_of_Drivers", "Vehicle_Type", "Expected_Ride_Duration"]])
y = np.array(data["adjusted_ride_cost"])

# Eğitim ve test verilerine ayırma: Verileri eğitim ve test olarak ayırıyoruz.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# RandomForestRegressor Modeli: Random Forest algoritması ile modelimizi eğitiyoruz.
model = RandomForestRegressor()
model.fit(x_train, y_train)

# Kullanıcıdan Girdi Al: Kullanıcıdan fiyat tahmini için gerekli parametreleri alıyoruz.
st.subheader("Fiyat Tahmini Yap")
user_number_of_riders = st.number_input("Rider Sayısı", min_value=1, max_value=100, value=10)
user_number_of_drivers = st.number_input("Driver Sayısı", min_value=1, max_value=100, value=10)
user_vehicle_type = st.selectbox("Araç Tipi", ["Economy", "Premium"])
expected_ride_duration = st.number_input("Beklenen Yolculuk Süresi (Dakika)", min_value=1, max_value=100, value=30)

# Araç Tipi Sayısal Değere Çevirme: Kullanıcının seçtiği araç tipi sayısal değere dönüştürülüyor.
vehicle_type_mapping = {"Premium": 1, "Economy": 0}
vehicle_type_numeric = vehicle_type_mapping[user_vehicle_type]

# Fiyat Tahmini Yapma: Kullanıcıdan alınan verilerle fiyat tahmini yapılır.
input_data = np.array([[user_number_of_riders, user_number_of_drivers, vehicle_type_numeric, expected_ride_duration]])
predicted_price = model.predict(input_data)

# Kullanıcıya tahmin edilen fiyatı gösteriyoruz.
st.write(f"Tahmin Edilen Fiyat: {predicted_price[0]:.2f} ₺")

# Gerçek ve Tahmin Edilen Değerler: Modelin başarısını test etmek için gerçek ve tahmin edilen değerleri karşılaştırıyoruz.
y_pred = model.predict(x_test)

# Gerçek ve Tahmin Edilen Değerler Karşılaştırma Grafiği: Gerçek ve tahmin edilen fiyatları karşılaştırıyoruz.
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Gerçek vs Tahmin')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='İdeal')

# Başlık ve etiketler ekliyoruz.
plt.title('Gerçek ve Tahmin Edilen Değerler')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.legend()
st.pyplot(plt)
