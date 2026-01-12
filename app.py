import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, RocCurveDisplay, mean_squared_error, r2_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.impute import SimpleImputer

# ------------------ Configuração visual ------------------
st.set_page_config(page_title="Análise Automática", layout="wide")

# Tema customizado via CSS
st.markdown(
    """
    <style>
    body {
        background-color: #F5F5F5;
        color: #333333;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E3F2FD;
        color: #1E88E5;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 Análise Automática — Classificação ou Regressão")

# ------------------ Carregamento de dados ------------------
uploaded = st.file_uploader("Selecione o arquivo CSV", type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("✅ Arquivo carregado com sucesso")
else:
    st.stop()

# ------------------ Seleção da variável alvo ------------------
target = st.selectbox("Selecione a variável alvo", df.columns)
X = df.drop(columns=[target])
y = df[target]

# ------------------ Pré-processamento ------------------
num_cols = X.select_dtypes(include=['float64','int64']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

numeric_tf = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_tf = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocess = ColumnTransformer([('num', numeric_tf, num_cols), ('cat', categorical_tf, cat_cols)])

# ------------------ Detectar tipo de problema ------------------
is_classification = (y.dtype == 'object') or (len(y.unique()) < 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Layout com abas ------------------
tab1, tab2, tab3, tab4 = st.tabs(["📂 Dados", "⚙️ Modelos", "📊 Gráficos", "📑 Relatório"])

relatorio = ""

with tab1:
    st.subheader("📂 Dados Carregados")
    st.write(df.head())
    st.write("Formato:", df.shape)

with tab2:
    st.subheader("⚙️ Treinamento e Comparação de Modelos")
    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=300),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss')
        }
        scores = {}
        for name, model in models.items():
            pipe = Pipeline([('prep', preprocess), ('model', model)])
            pipe.fit(X_train, y_train)
            y_proba = pipe.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_proba)
            scores[name] = auc
        st.write("ROC-AUC:", scores)
        best_model = max(scores, key=scores.get)
        relatorio = f"""
# 📑 Relatório de Interpretação — Classificação

## 📊 Desempenho dos Modelos
| Modelo               | ROC-AUC |
|----------------------|---------|
| Logistic Regression  | {scores['Logistic Regression']:.3f} |
| Random Forest        | {scores['Random Forest']:.3f} |
| XGBoost              | {scores['XGBoost']:.3f} |

**➡ Melhor modelo:** `{best_model}`

---

## 🎯 Conclusão
O modelo **{best_model}** apresentou o melhor desempenho e é indicado para prever a variável alvo.
"""
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=200, random_state=42)
        }
        scores = {}
        for name, model in models.items():
            pipe = Pipeline([('prep', preprocess), ('model', model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            scores[name] = {"RMSE": rmse, "R²": r2}
        st.write("Métricas:", scores)
        best_model = min(scores, key=lambda k: scores[k]["RMSE"])
        relatorio = f"""
# 📑 Relatório de Interpretação — Regressão

## 📊 Desempenho dos Modelos
| Modelo              | RMSE   | R²    |
|---------------------|--------|-------|
| Linear Regression   | {scores['Linear Regression']['RMSE']:.3f} | {scores['Linear Regression']['R²']:.3f} |
| Random Forest       | {scores['Random Forest']['RMSE']:.3f} | {scores['Random Forest']['R²']:.3f} |
| XGBoost             | {scores['XGBoost']['RMSE']:.3f} | {scores['XGBoost']['R²']:.3f} |

**➡ Melhor modelo:** `{best_model}`

---

## 🎯 Conclusão
O modelo **{best_model}** apresentou o melhor desempenho e é indicado para prever a variável alvo.
"""

with tab3:
    st.subheader("📊 Gráficos de Diagnóstico")
    if is_classification:
        # Curvas ROC
        fig, ax = plt.subplots()
        for name, model in models.items():
            pipe = Pipeline([('prep', preprocess), ('model', model)])
            pipe.fit(X_train, y_train)
            y_proba = pipe.predict_proba(X_test)[:,1]
            RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax)
        st.pyplot(fig)

        # Matriz de confusão (Random Forest)
        best_rf = Pipeline([('prep', preprocess), ('model', models["Random Forest"])])
        best_rf.fit(X_train, y_train)
        y_pred_rf = best_rf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_rf)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        st.pyplot(fig)

    else:
        # Gráfico de resíduos
        fig, ax = plt.subplots()
        y_pred = list(models.values())[0].fit(X_train, y_train).predict(X_test)
        ax.scatter(y_pred, y_test - y_pred)
        ax.axhline(0, color='red')
        ax.set_title("Resíduos vs Valores preditos")
        st.pyplot(fig)

        # Importância das variáveis (Random Forest)
        rf_model = Pipeline([('prep', preprocess), ('model', models["Random Forest"])])
        rf_model.fit(X_train, y_train)
        feature_names = num_cols.copy()
        if len(cat_cols) > 0:
            encoder = rf_model.named_steps['prep'].named_transformers_['cat'].named_steps['encoder']
            feature_names.extend(encoder.get_feature_names_out(cat_cols))
        importances = rf_model.named_steps['model'].feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([feature_names[i] for i in indices][::-1], importances[indices][::-1], color="#1E88E5")
        ax.set_xlabel("Importância")
        ax.set_title("Top 15 Variáveis mais Relevantes")
        st.pyplot(fig)

with tab4:
    st.download_button(
    label="⬇️ Baixar Relatório",
    data=relatorio,
    file_name="relatorio_analise.md",
    mime="text/markdown"
)