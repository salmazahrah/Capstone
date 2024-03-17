import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader import data as wb
from scipy.stats import norm
from scipy.stats import genextreme
from scipy.optimize import minimize
import yfinance as yf
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

# set bentuk halaman web
st.set_page_config(layout='wide')

# Aktifkan Streamlit debug mode
st.set_option('deprecation.showPyplotGlobalUse', False)

with st.sidebar:
    st.subheader("Penulis")
    st.write("Salma Fitria F Z")

# Tentukan tanggal awal
start_date = '2023-01-01'
end_date = '2024-01-01'

# Menggunakan fungsi DataReader dari yfinance
data01 = yf.download("BBRI.JK", start=start_date, end=end_date)
data02 = yf.download("BBNI.JK", start=start_date, end=end_date)
data03 = yf.download("BRIS.JK", start=start_date, end=end_date)


# Inisialisasi DataFrame
data00 = pd.DataFrame()
data00['BBRI.JK'] = data01['Adj Close']
data00['BBNI.JK'] = data02['Adj Close']
data00['BRIS.JK'] = data03['Adj Close']

data00.index = pd.to_datetime(data00.index)

# Return saham harian
r = data00.pct_change().dropna()
returns_bbri = r['BBRI.JK']
returns_bbni = r['BBNI.JK']
returns_bris = r['BRIS.JK']
    
########## Perhitungan Parameter
# BBRI
gev_bbriarams_mle_bbri = genextreme.fit(returns_bbri)
xi_bbri, mu_bbri, beta_bbri = gev_bbriarams_mle_bbri
# BBNI
gev_bbniarams_mle_bbni = genextreme.fit(returns_bbni)
xi_bbni, mu_bbni, beta_bbni = gev_bbniarams_mle_bbni
# BRIS
gev_brisarams_mle_bris = genextreme.fit(returns_bris)
xi_bris, mu_bris, beta_bris = gev_brisarams_mle_bris

########## Perhitungan iterasi dan Simulasi Monte Carlo
alpha = 0.05
z_score = 1.96  # Z-score untuk tingkat kepercayaan 95%
# BBRI
std_dev_bbri = np.std(returns_bbri)
mean_bbri = np.mean(returns_bbri)
iterations_bbri = 2*len(returns_bbri)
# iterations_bbri = ((100 * z_score * std_dev_bbri) / (5 * mean_bbri))**2
# BBNI
std_dev_bbni = np.std(returns_bbni)
mean_bbni = np.mean(returns_bbni)
iterations_bbni = 2*len(returns_bbni)
# iterations_bbni = ((100 * z_score * std_dev_bbni) / (5 * mean_bbni))**2
# BRIS
std_dev_bris = np.std(returns_bris)
mean_bris = np.mean(returns_bris)
iterations_bris = 2*len(returns_bris)
# iterations_bris = ((100 * z_score * std_dev_bris) / (5 * mean_bris))**2

########### Simulasi Monte Carlo
np.random.seed(42)
def monte_carlo_simulation_genextreme(xi, mu, beta, returns, iterations):
    VaR = np.percentile(
        genextreme.rvs(xi, loc=mu, scale=beta, size=(iterations, len(returns))),
        alpha * 100,
        axis=1
    )
    return np.mean(VaR)

# BBRI
BBRI = monte_carlo_simulation_genextreme(xi_bbri, mu_bbri, beta_bbri, returns_bbri, int(iterations_bbri))
# BBNI
BBNI = monte_carlo_simulation_genextreme(xi_bbni, mu_bbni, beta_bbni, returns_bbni, int(iterations_bbni))
# BRIS
BRIS = monte_carlo_simulation_genextreme(xi_bris, mu_bris, beta_bris, returns_bris, int(iterations_bris))


st.title("Estimasi Resiko Portofolio Saham")
#####################################
# Helper Function
def format_big_number(num):
    abs_num = abs(num)  # Ambil nilai absolut
    if abs_num >= 1e6:
        return f"{num / 1e6:.2f} Juta" if num >= 0 else f"-{abs_num / 1e6:.2f} Juta"
    elif abs_num >= 1e3:
        return f"{num / 1e3:.2f} Ribu" if num >= 0 else f"-{abs_num / 1e3:.2f} Ribu"
    else:
        return f"{num:.2f}"
    
###### Tabbing plot saham
tab1, tab2, tab3, tab4 = st.tabs(['PORTOFOLIO','Daily Return', 'Matriks Korelasi', 'Simulasi'])

with tab1:
    with st.expander("Data Harga Closing Saham"):
        st.write(data00.head())
        st.caption("[finance.yahoo.com](https://finance.yahoo.com/quote/BBRI.JK/history)")
        
    st.write("Grafik Harga Closing Saham")
    data = st.multiselect(
        "Pilih Saham",
        ['BBRI.JK','BBNI.JK','BRIS.JK'],
        key="multiselect_1"
        )
    selected_data = data00[data].reset_index()
    line_chart = px.line(selected_data, x='Date', y=data, title='Line Plot Harga Closing Saham Terpilih')
    st.plotly_chart(line_chart)
    
    st.write("Statistika Deskriptif")
    st.write(data00.describe())
            
with tab2:
    st.write("Dailyl Returns")
    st.write(r)
    data2 = st.multiselect(
        "Pilih Saham",
        ['BBRI.JK','BBNI.JK','BRIS.JK'],
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
        1. Dapat dilihat bahwa nilai korelasi saham BBNI.JK dengan BRIS.JK bernilai positif 0.24, ini menunjukkan bahwa saham BBNI.JK dan BRIS.JK memiiki hubungan yang lemah
        2. Dapat dilihat bahwa nilai korelasi saham BBRI.JK dengan BBNI.JK bernilai positif 0.43, ini menujukkan bahwa saham BBRI.JK dan BBNI.JK memiliki hubungan yang sedang
        3. Dapat dilihat bahwa nilai korelasi saham BBRI.JK dengan BBRIS.JK bernilai positif 0.70, ini menujukkan bahwa saham BBRI.JK dan BBNI.JK memiliki hubungan yang kuat
    """)    
    
with tab4:
    st.write("Simulasi")
    # Proporsi Saham pada Portofolio
    p_bbri = 0.33
    p_bbni = 0.51
    p_bris = 0.16

    data00['porto'] = ((p_bbri * data00['BBRI.JK']) + (p_bbni * data00['BBNI.JK']) + (p_bris * data00['BRIS.JK']))    
    r['PORTOFOLIO'] = data00['porto'].pct_change().dropna()
    returns_porto = r['PORTOFOLIO']

    gev_params_mle_porto = genextreme.fit(returns_porto)
    xi_porto, mu_porto, beta_porto = gev_params_mle_porto

    # Perhitungan iterasi
    std_dev_porto = np.std(returns_porto)
    mean_porto = np.mean(returns_porto)
    iterations_porto = 2*len(returns_porto)
    # iterations_porto = ((100 * z_score * std_dev_porto) / (5 * mean_porto))**2

    # Simulasi Monte Carlo
    np.random.seed(42)
    PORTOFOLIO = monte_carlo_simulation_genextreme(xi_porto, mu_porto, beta_porto, returns_porto, int(iterations_porto))

    # investasi_awal = 100000000
    # hasil_bbri = investasi_awal * BBRI
    # hasil_bbni = investasi_awal * BBNI
    # hasil_bris = investasi_awal * BRIS
    # hasil_porto = investasi_awal * PORTOFOLIO
    
    hasil_investasi = pd.DataFrame({
        'Saham': ['BBRI', 'BBNI', 'BRIS', 'PORTOFOLIO'],
        'Value_at_Risk (%)': [BBRI, BBNI, BRIS, PORTOFOLIO]
    })
    st.write("Tabel Hasil Investasi", hasil_investasi)
