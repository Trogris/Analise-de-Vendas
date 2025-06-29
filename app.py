import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Análise de Vendas", layout="wide")
st.title("📊 Análise Inteligente de Vendas")

# Upload
file = st.file_uploader("📁 Faça upload do arquivo Excel (.xlsx)", type="xlsx")

if file:
    df = pd.read_excel(file)
    st.success("Arquivo carregado com sucesso!")

    if st.button("👀 Exibir primeiras linhas"):
        st.dataframe(df.head())

    if st.button("🔍 Análise exploratória"):
        st.subheader("📋 Estatísticas")
        st.write(df.describe())

        st.subheader("📊 Tipos de Dados")
        st.write(df.dtypes)

        st.subheader("🧹 Valores Nulos")
        st.write(df.isnull().sum())

        st.subheader("📈 Dispersão Preço x Total")
        fig1 = sns.scatterplot(data=df, x='Preço', y='Total')
        st.pyplot(fig1.figure)

        st.subheader("🔥 Correlação")
        corr = df.select_dtypes(include='number').corr()
        fig2 = plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(fig2)

    if st.button("⚙️ Treinar modelo de regressão"):
        drop_cols = ['ID', 'Data', 'Cliente']
        drop_cols = [col for col in drop_cols if col in df.columns]
        X = df.drop(columns=['Total'] + drop_cols)
        X = pd.get_dummies(X, drop_first=True)
        y = df['Total']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5

        st.success(f"✅ Modelo treinado com sucesso!")
        st.metric("🔍 RMSE (erro médio quadrático)", f"R$ {rmse:.2f}")

        importances = model.feature_importances_
        features = X.columns
        sorted_idx = importances.argsort()[::-1]
        top_features = features[sorted_idx][:10]
        top_importances = importances[sorted_idx][:10]

        fig3 = plt.figure(figsize=(10, 6))
        plt.barh(top_features[::-1], top_importances[::-1])
        plt.xlabel("Importância")
        plt.title("Top 10 variáveis que mais impactam o Total")
        st.pyplot(fig3)

        st.subheader("📝 Relatório Rápido")
        st.markdown(f"""
        - Valor médio de venda: R$ {df['Total'].mean():.2f}  
        - Desvio padrão: R$ {df['Total'].std():.2f}  
        - Principais influenciadores: **Quantidade**, **Preço**
        """)

else:
    st.info("Envie um arquivo Excel (.xlsx) para começar.")

# 🔄 Botão para reiniciar
if st.button("🔄 Reiniciar análise"):
    st.cache_data.clear()
    st.rerun()
