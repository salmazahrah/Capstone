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
start_date = '2022-01-01'
end_date = '2024-01-01'

# Menggunakan fungsi DataReader dari yfinance
data01 = yf.download("BBRI.JK", start=start_date, end=end_date)
data02 = yf.download("BBNI.JK", start=start_date, end=end_date)
data03 = yf.download("BMRI.JK", start=start_date, end=end_date)


# Inisialisasi DataFrame
data00 = pd.DataFrame()
data00['BBRI.JK'] = data01['Adj Close']
data00['BBNI.JK'] = data02['Adj Close']
data00['BMRI.JK'] = data03['Adj Close']

data00.index = pd.to_datetime(data00.index)

# Return saham harian
r = data00.pct_change().dropna()
returns_bbri = r['BBRI.JK']
returns_bbni = r['BBNI.JK']
returns_bmri = r['BMRI.JK']
    
########## Perhitungan Parameter
# BBRI
gev_bbriarams_mle_bbri = genextreme.fit(returns_bbri)
xi_bbri, mu_bbri, beta_bbri = gev_bbriarams_mle_bbri
# BBNI
gev_bbniarams_mle_bbni = genextreme.fit(returns_bbni)
xi_bbni, mu_bbni, beta_bbni = gev_bbniarams_mle_bbni
# BMRI
gev_bmriarams_mle_bmri = genextreme.fit(returns_bmri)
xi_bmri, mu_bmri, beta_bmri = gev_bmriarams_mle_bmri

########## Perhitungan iterasi
alpha = 0.05
# z_score = 1.96  # Z-score untuk tingkat kepercayaan 95%
# BBRI
iterations_bbri = 2*len(returns_bbri)
# BBNI
iterations_bbni = 2*len(returns_bbni)
# BMRI
iterations_bmri = 2*len(returns_bmri)

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
# BMRI
BMRI = monte_carlo_simulation_genextreme(xi_bmri, mu_bmri, beta_bmri, returns_bmri, int(iterations_bmri))


st.title("Resiko Portofolio Saham")
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
tab1, tab2, tab3, tab4 = st.tabs(['Saham','Daily Return', 'Matriks Korelasi', 'Simulasi'])

with tab1:
    with st.expander("Data Harga Closing Saham"):
        st.write(data00.head())
        st.caption("[finance.yahoo.com](https://finance.yahoo.com/quote/BBRI.JK/history)")
        
    st.write("Grafik Harga Closing Saham")
    data = st.multiselect(
        "Pilih Saham",
        ['BBRI.JK','BBNI.JK','BMRI.JK'],
        key="multiselect_1"
        )
    selected_data = data00[data].reset_index()
    line_chart = px.line(selected_data, x='Date', y=data, title='Grafik Harga Closing Saham Terpilih')
    st.plotly_chart(line_chart)
    
    
            
with tab2:
    st.write("Daily Returns")
    st.write(r.head())
    st.write("Statistika Deskriptif")
    st.write(r.describe())
    data2 = st.multiselect(
        "Pilih Saham",
        ['BBRI.JK','BBNI.JK','BMRI.JK'],
        key="multiselect_2"
    )
    returns = r[data2].reset_index()
    line_chart = px.line(returns, x='Date', y=data2, title='Grafik Return Harian Saham Terpilih')
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
        Dengan menggunakan matriks korelasi, kita dapat melihat bahwa ketiga saham memiliki korelasi positif yang kuat
    """)    
    
with tab4:
    # Proporsi Saham pada Portofolio
    p_bbri = 0.41
    p_bbni = 0.30
    p_bmri = 0.29
    r['porto'] = ((p_bbri * r['BBRI.JK']) + (p_bbni * r['BBNI.JK']) + (p_bmri * r['BMRI.JK']))    
    returns_porto = r['porto']

    gev_params_mle_porto = genextreme.fit(returns_porto)
    xi_porto, mu_porto, beta_porto = gev_params_mle_porto
    iterations_porto = 2*len(returns_porto)

    # Simulasi Monte Carlo
    np.random.seed(42)
    PORTOFOLIO = monte_carlo_simulation_genextreme(xi_porto, mu_porto, beta_porto, returns_porto, int(iterations_porto))
        
    investasi_awal = 100000000
    hasil_bbri = investasi_awal * (1+BBRI)
    hasil_bbni = investasi_awal * (1+BBNI)
    hasil_bmri = investasi_awal * (1+BMRI)
    hasil_porto = investasi_awal * (1+PORTOFOLIO)
    
    hasil_investasi = pd.DataFrame({
        'Saham': ['BBRI', 'BBNI', 'BMRI', 'PORTOFOLIO'],
        'Value_at_Risk (%)': [round(BBRI*100, 2), round(BBNI*100, 2), round(BMRI*100, 2), round(PORTOFOLIO*100, 2)],
        'Investasi @100juta': [round(hasil_bbri), round(hasil_bbni), round(hasil_bmri), round(hasil_porto)]
    })
    st.write("Nilai Risiko dari Tingkat Pengembalian Harian Investasi", hasil_investasi)
    
