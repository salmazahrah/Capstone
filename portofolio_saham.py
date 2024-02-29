import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader import data as wb
from scipy.stats import norm
from scipy.optimize import minimize
import yfinance as yf
import altair as alt
import plotly.express as px

# set bentuk halaman web
st.set_page_config(layout='wide')

with st.sidebar:
    st.subheader("Penulis")
    st.write("Salma Fitria F Z")

# Tentukan tanggal awal
start_date = '2023-01-01'
end_date = '2024-01-01'

# Menggunakan fungsi DataReader dari yfinance
data01 = yf.download("ALHC", start=start_date, end=end_date)
data02 = yf.download("CELH", start=start_date, end=end_date)
data03 = yf.download("FLNC", start=start_date, end=end_date)


# Inisialisasi DataFrame
data00 = pd.DataFrame()
data00['ALHC'] = data01['Adj Close']
data00['CELH'] = data02['Adj Close']
data00['FLNC'] = data03['Adj Close']
data00.index = pd.to_datetime(data00.index)

# Return saham harian
r = data00.pct_change().dropna()

# Aktifkan Streamlit debug mode
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Estimasi Resiko Portofolio Saham")
#####################################
# Helper Function
def format_big_number(num):
    if num >= 1e6:
        return f"{num / 1e6:.2f} Juta"
    elif num >= 1e3:
        return f"{num / 1e3:.2f} Ribu"
    else:
        return f"{num:.2f}"

###### Tabbing plot saham
tab1, tab2, tab3, tab4 = st.tabs(['PORTOFOLIO','Daily Return', 'Matriks Korelasi', 'Simulasi'])

with tab1:
    
    with st.expander("Data Harga Closing Saham"):
        st.write(data00.head())
        st.caption("[finance.yahoo.com](https://finance.yahoo.com/quote/ALHC/history)")
        
    st.write("Grafik Harga Closing Saham")
    data = st.multiselect(
        "Pilih Saham",
        ['ALHC','CELH','FLNC'],
        key="multiselect_1"
        )
    selected_data = data00[data].reset_index()
    line_chart = px.line(selected_data, x='Date', y=data, title='Line Plot Harga Closing Saham Terpilih')
    st.plotly_chart(line_chart)
    
    st.write("Statistika Deskriptif")
    st.write(data00.describe())
        
    
with tab2:
    st.write("Dailyl Returns")
    data2 = st.multiselect(
        "Pilih Saham",
        ['ALHC','CELH','FLNC'],
        key="multiselect_2"
    )
    returns = r[data2].reset_index()
    line_chart = px.line(returns, x='Date', y=data2, title='Line Plot Return Harian Saham Terpilih')
    st.plotly_chart(line_chart)
    
with tab3:
    st.write("Korelasi Saham")
    ####### Correlation Matrix
    def create_correlation_matrix(data):
        correlation_matrix = data.corr()
        melted_matrix = pd.melt(correlation_matrix.reset_index(), id_vars='index')
        melted_matrix.columns = ['Saham 1', 'Saham 2', 'Korelasi']
        return melted_matrix
        return melted_matrix

    # Fungsi untuk membuat visualisasi matriks korelasi
    def create_correlation_chart(data):
        chart = alt.Chart(data).mark_rect().encode(
            x='Saham 1:N',
            y='Saham 2:N',
            color='Korelasi:Q',
            tooltip=['Saham 1:N', 'Saham 2:N', 'Korelasi:Q']
        ).properties(
            width=400,
            height=400
        )
        text_chart = alt.Chart(data).mark_text(baseline='middle').encode(
            x='Saham 1:N',
            y='Saham 2:N',
            text=alt.Text('Korelasi:Q', format=".2f"),
            color=alt.condition(
                alt.datum.Korelasi > 0.5,
                alt.value('white'),
                alt.value('black')
            )
        )
        return chart + text_chart

    # Membuat matriks korelasi dan melted DataFrame
    melted_matrix = create_correlation_matrix(data00)

    # Menampilkan visualisasi matriks korelasi dengan Altair di Streamlit
    st.altair_chart(create_correlation_chart(melted_matrix))

    st.markdown("__Matrix Korelasi__")
    st.write("""
        Dengan menggunakan matriks korelasi, kita dapat melihat hubungan antara harga penutupan saham
        1. Dapat dilihat bahwa nilai korelasi saham CELH dan FLNC bernilai positif, ini menunjukkan bahwa saham CELH bergerak dalam arah yang sama
        2. Dapat dilihat bahwa nilai korelasi saham ALHC dengan CELH dan FLNC bernilai negatif, ini menujukkan bahwa saham ALHC cenderung bergerak kearah yang berlawanan dengan saham CELH dan FLNC
    """)    
    
with tab4:
    st.write("Simulasi")
    ############### Proporsi Saham pada Portofolio
    # Kolom
    col1, col2, col3 = st.columns(3)
    with col1:
        p_alhc = st.number_input("Proporsi Saham ALHC", min_value=0.0, max_value=1.0, step=0.01, value=0.33)

    with col2:
        p_celh = st.number_input("Proporsi Saham CELH", min_value=0.0, max_value=1.0, step=0.01, value=0.51)

    with col3:
        p_flnc = st.number_input("Proporsi Saham FLNC", min_value=0.0, max_value=1.0, step=0.01, value=0.16)

    # Periksa apakah total proporsi adalah 1
    total_proporsi = p_alhc + p_celh + p_flnc

    # Tampilkan pesan jika total proporsi tidak sama dengan 1
    if total_proporsi != 1:
        st.warning(f"Total proporsi saat ini adalah {total_proporsi:.2f}. Pastikan total proporsi sama dengan 1.")

    data00['porto'] = ((p_alhc * data00['ALHC']) + (p_celh * data00['CELH']) + (p_flnc * data00['FLNC']))    
    # Fungsi untuk menjalankan simulasi Monte Carlo
    def monte_carlo_simulation(initial_price, drift, volatility, time_steps, num_simulations):
        dt = 1 / 252  # perhitungan per hari dalam setahun
        simulations = []

        for _ in range(num_simulations):
            daily_returns = np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * np.random.normal(0, 1, time_steps))
            stock_price = initial_price * np.cumprod(daily_returns)
            simulations.append(stock_price)

        return np.array(simulations)


    ## Parameter-model
    return_daily = data00['porto'].pct_change().dropna()
    return_daily.index = pd.to_datetime(return_daily.index)
    expected_return = return_daily.mean()
    var = return_daily.var()
    drift = expected_return - (0.5 * var) # tingkat drift harian (sesuai dengan data historis)
    volatility = return_daily.std() # volatilitas harian (sesuai dengan data historis)
    initial_price = data00['porto']  # harga saham awal
    time_steps = len(data00)  # jumlah langkah waktu (dalam satu tahun)
    num_simulations = 1000

    # Jalankan simulasi Monte Carlo
    simulations = monte_carlo_simulation(initial_price, drift, volatility, time_steps, num_simulations)

    ############# Menghitung VaR
    # selectbox menampilkan Tingkat Kepercayaan dropdown untuk dipilih
    alpha = st.selectbox(
        "Pilih Tingkat Kepercayaan yang akan digunakan (%)",
        ['1','5','10']
    )
    alpha = float(alpha)  # Mengonversi alpha menjadi float
    tingkat_kepercayaan = 100 - alpha

    # number_input untuk input number
    investasi = st.number_input(
        "Jumlah investasi awal",
        min_value=0,
        max_value=999999999,
        step=1,
        value=100000000
    )
    var_percentile = np.percentile(simulations[:, -1], alpha)
    hasil = investasi * var_percentile/100

    st.write(f"Estimasi resiko dengan tingkat kepercayaan {tingkat_kepercayaan}% setelah dilakukan 1000 kali iterasi sebesar {format_big_number(var_percentile)}%")

    st.write(f"Jika investasi awal sebesar {investasi} dengan tingkat kepercayaan {tingkat_kepercayaan}% di hari selanjutnya mendapatkan hasil sebesar {format_big_number(hasil)}")





 









