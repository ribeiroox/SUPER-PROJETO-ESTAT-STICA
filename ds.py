import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
from scipy.stats import spearmanr, f_oneway, ttest_ind
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

# ── Página ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Amazon Products — Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    [data-testid="stSidebar"] { background: #0d1b2a; }
    [data-testid="stSidebar"] * { color: #e8f0fe !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #7eb8f7 !important; font-size: 13px !important;
        text-transform: uppercase; letter-spacing: .06em;
    }
    [data-testid="stSidebar"] hr { border-color: #1e3a5f; }

    .mc { background:#0d1b2a; border:1px solid #1e3a5f; border-radius:10px;
          padding:.9rem 1rem; text-align:center; }
    .mc-lbl { font-size:11px; color:#7eb8f7; margin-bottom:4px; font-weight:600;
              text-transform:uppercase; letter-spacing:.05em; }
    .mc-val { font-size:24px; font-weight:700; color:#ffffff; }
    .mc-sub { font-size:11px; color:#4a7fa5; margin-top:3px; }

    .sec { font-size:17px; font-weight:700; color:#ffffff;
           background:#1e3a5f; padding:8px 14px; border-radius:8px;
           border-left:4px solid #4d9de0; margin-bottom:.6rem; }

    .sec2 { font-size:17px; font-weight:700; color:#ffffff;
            background:#1a3820; padding:8px 14px; border-radius:8px;
            border-left:4px solid #2ecc71; margin-bottom:.6rem; }

    .obj { background:#0a2540; border-left:4px solid #4d9de0; border-radius:6px;
           padding:.5rem .9rem; font-size:13px; color:#a8ccf0; margin-bottom:.6rem; }

    .ins { background:#0a2e14; border-left:4px solid #2ecc71; border-radius:6px;
           padding:.6rem .9rem; font-size:13.5px; color:#a8f0c0; margin-top:.4rem; }

    .warn { background:#2e1a0a; border-left:4px solid #f39c12; border-radius:6px;
            padding:.6rem .9rem; font-size:13.5px; color:#f0d0a8; margin-top:.4rem; }

    .resultado-card { border-radius:12px; padding:1.2rem; text-align:center;
                      border:2px solid; margin-bottom:.5rem; }
    .prob-bar-wrap { background:#0d1b2a; border-radius:8px; overflow:hidden;
                     height:22px; margin:4px 0; }
    .prob-bar-fill { height:100%; border-radius:8px; display:flex;
                     align-items:center; padding-left:8px;
                     font-size:12px; font-weight:600; color:#fff; }

    .badge { display:inline-block; background:#4d9de0; color:#fff;
             font-size:11px; padding:2px 9px; border-radius:999px; margin-right:4px; }

    .tab-header { font-size:15px; font-weight:600; color:#7eb8f7;
                  padding:4px 0; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Dados & modelos ──────────────────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    df = pd.read_csv("amazon.csv")
    df = df.rename(columns={
        'product_id':'id_produto','product_name':'nome_produto','category':'categoria',
        'discounted_price':'preco_com_desconto','actual_price':'preco_original',
        'discount_percentage':'percentual_desconto','rating':'nota_avaliacao',
        'rating_count':'total_avaliacoes','about_product':'sobre_produto',
        'user_id':'id_usuario','user_name':'nome_usuario','review_id':'id_comentario',
        'review_title':'titulo_comentario','review_content':'conteudo_comentario',
        'img_link':'link_imagem','product_link':'link_produto'
    })
    df = df.dropna()
    for col in ['preco_com_desconto','preco_original']:
        df[col] = df[col].str.replace('₹','',regex=False).str.replace('$','',regex=False).str.replace(',','',regex=False).astype(float)
    df['total_avaliacoes']    = df['total_avaliacoes'].str.replace(',','',regex=False).astype(float)
    df['nota_avaliacao']      = pd.to_numeric(df['nota_avaliacao'], errors='coerce')
    df['percentual_desconto'] = df['percentual_desconto'].str.replace('%','',regex=False).astype(float)
    cotacao = 0.0522
    df['preco_original']     = (df['preco_original']     * cotacao).round(2)
    df['preco_com_desconto'] = (df['preco_com_desconto'] * cotacao).round(2)
    df['categoria'] = df['categoria'].str.split('|').str[0].replace({
        "Electronics":"Eletrônicos","Computers&Accessories":"Computadores e Acessórios",
        "Home&Kitchen":"Casa e Cozinha","OfficeProducts":"Produtos de Escritório",
        "MusicalInstruments":"Instrumentos Musicais","HomeImprovement":"Melhoria Residencial",
        "Toys&Games":"Brinquedos e Jogos","Car&Motorbike":"Carros e Motocicletas",
        "Health&PersonalCare":"Saúde e Cuidados Pessoais"
    })
    df['satisfacao_cliente'] = np.select(
        [df['nota_avaliacao'] >= 4.2,
         (df['nota_avaliacao'] >= 3.5) & (df['nota_avaliacao'] < 4.2),
         df['nota_avaliacao'] < 3.5],
        ['Alta','Regular','Baixa'], default='Regular'
    )
    Q1, Q3 = df['preco_original'].quantile(0.25), df['preco_original'].quantile(0.75)
    df['is_outlier'] = df['preco_original'] > Q3 + 1.5*(Q3-Q1)
    df = df.dropna(subset=['nota_avaliacao'])
    return df

@st.cache_resource
def treinar_modelos(df):
    le = LabelEncoder()
    df = df.copy()
    df['categoria_enc'] = le.fit_transform(df['categoria'])
    features = ['preco_original','percentual_desconto','total_avaliacoes','categoria_enc']
    X = df[features]
    y = df['satisfacao_cliente']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_tr_sc, y_train)

    # Parâmetros Bayes manual
    classes = y.unique()
    total   = len(df)
    priori  = {c: len(df[df['satisfacao_cliente']==c])/total for c in classes}
    feats   = ['preco_original','percentual_desconto','total_avaliacoes']
    lk_params = {}
    for c in classes:
        sub = df[df['satisfacao_cliente']==c]
        lk_params[c] = {f: (sub[f].mean(), sub[f].std()) for f in feats}

    y_pred_dt  = dt.predict(X_test)
    y_pred_knn = knn.predict(X_te_sc)

    return {
        'dt': dt, 'knn': knn, 'scaler': scaler, 'le': le,
        'priori': priori, 'lk_params': lk_params, 'classes': list(classes),
        'X_test': X_test, 'y_test': y_test,
        'y_pred_dt': y_pred_dt, 'y_pred_knn': y_pred_knn,
        'acc_dt':  accuracy_score(y_test, y_pred_dt),
        'acc_knn': accuracy_score(y_test, y_pred_knn),
    }

def gaussiana(x, mu, sigma):
    if sigma == 0: return 1.0 if x == mu else 1e-10
    return (1/(sigma*math.sqrt(2*math.pi))) * math.exp(-0.5*((x-mu)/sigma)**2)

def classificar_bayes(preco, desconto, avaliacoes, priori, lk_params, classes):
    post = {}
    for c in classes:
        p  = priori[c]
        p *= gaussiana(preco,      *lk_params[c]['preco_original'])
        p *= gaussiana(desconto,   *lk_params[c]['percentual_desconto'])
        p *= gaussiana(avaliacoes, *lk_params[c]['total_avaliacoes'])
        post[c] = p
    soma = sum(post.values())
    if soma == 0: return {c: 1/len(classes) for c in classes}
    return {c: v/soma for c, v in post.items()}

df_raw = carregar_dados()
modelos = treinar_modelos(df_raw)

COR_MAP  = {"Alta":"#4d9de0","Regular":"#2ecc71","Baixa":"#e74c3c"}
TEMPLATE = "plotly_dark"
PLOT_CFG = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e8f0fe', title_font_size=14, margin=dict(t=48,b=8,l=8,r=8))

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filtros (Seção 1)")
    st.markdown("---")
    cats_disponiveis = sorted(df_raw['categoria'].unique())
    cats_sel = st.multiselect("Categoria", options=cats_disponiveis,
                              default=cats_disponiveis, placeholder="Selecione...")
    st.markdown("---")
    sat_sel = st.multiselect("Satisfação", options=["Alta","Regular","Baixa"],
                             default=["Alta","Regular","Baixa"])
    st.markdown("---")
    preco_min_total = float(df_raw['preco_original'].min())
    preco_max_total = float(df_raw['preco_original'].max())
    preco_range = st.slider("Faixa de preço (R$)",
                            min_value=preco_min_total, max_value=preco_max_total,
                            value=(preco_min_total, preco_max_total),
                            step=10.0, format="R$ %.0f")
    st.markdown("---")
    mostrar_outliers = st.radio("Outliers de preço",
                                options=["Incluir outliers","Apenas outliers","Excluir outliers"], index=0)
    st.markdown("---")
    st.markdown("**ℹ️ Outliers** = produtos com preço > Q3 + 1,5×IQR (R$ 501).")

# ── Aplicar filtros ──────────────────────────────────────────────────────────
df = df_raw.copy()
if cats_sel:  df = df[df['categoria'].isin(cats_sel)]
if sat_sel:   df = df[df['satisfacao_cliente'].isin(sat_sel)]
df = df[(df['preco_original'] >= preco_range[0]) & (df['preco_original'] <= preco_range[1])]
if mostrar_outliers == "Apenas outliers":   df = df[df['is_outlier']]
elif mostrar_outliers == "Excluir outliers": df = df[~df['is_outlier']]

# ── Cabeçalho ────────────────────────────────────────────────────────────────
st.markdown("## 📊 Amazon Products — Dashboard de Análise")
st.markdown("Amazon Products Sales (Kaggle) · Valores convertidos para BRL (1 INR = R$ 0,0522)")

badges_html = ""
if len(cats_sel) < len(cats_disponiveis):
    badges_html += "".join(f'<span class="badge">{c}</span>' for c in cats_sel)
if set(sat_sel) != {"Alta","Regular","Baixa"}:
    badges_html += "".join(f'<span class="badge">Satisfação: {s}</span>' for s in sat_sel)
if preco_range != (float(df_raw['preco_original'].min()), float(df_raw['preco_original'].max())):
    badges_html += f'<span class="badge">R$ {preco_range[0]:.0f}–{preco_range[1]:.0f}</span>'
if mostrar_outliers != "Incluir outliers":
    badges_html += f'<span class="badge">{mostrar_outliers}</span>'
if badges_html:
    st.markdown(f"🎛️ Filtros ativos: {badges_html}", unsafe_allow_html=True)

if len(df) == 0:
    st.warning("Nenhum produto encontrado com os filtros selecionados.")
    st.stop()

# ── Tabs principais ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📈 Seção 1 — Análise dos Dados", "🤖 Seção 2 — Classificação Probabilística"])

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Cards
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    n_out = df['is_outlier'].sum()
    rho_desc, _ = spearmanr(df['percentual_desconto'], df['nota_avaliacao'])
    cards = [
        ("Total de produtos",  f"{len(df):,}",                      "após filtros"),
        ("Categorias",         f"{df['categoria'].nunique()}",       "selecionadas"),
        ("Preço médio",        f"R$ {df['preco_original'].mean():,.0f}", "original"),
        ("Nota média",         f"{df['nota_avaliacao'].mean():.2f}", "de 5,0"),
        ("Desc. médio",        f"{df['percentual_desconto'].mean():.0f}%", "sobre preço original"),
        ("Outliers",           f"{n_out}",                          f"{n_out/len(df)*100:.0f}% do total"),
    ]
    for col, (lbl, val, sub) in zip([c1,c2,c3,c4,c5,c6], cards):
        with col:
            st.markdown(f'<div class="mc"><div class="mc-lbl">{lbl}</div>'
                        f'<div class="mc-val">{val}</div>'
                        f'<div class="mc-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bloco 1 — Satisfação & Volume ────────────────────────────────────────
    st.markdown('<div class="sec">1 · Satisfação do Cliente e Volume por Categoria</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Mostrar como os produtos se distribuem entre os três níveis de satisfação e verificar o balanceamento das classes (relevante para avaliar o desempenho dos classificadores).</div>', unsafe_allow_html=True)
        sat_c = (df['satisfacao_cliente'].value_counts()
                 .reindex(['Alta','Regular','Baixa']).reset_index())
        sat_c.columns = ['satisfacao','quantidade']
        sat_c['pct'] = (sat_c['quantidade'] / sat_c['quantidade'].sum() * 100).round(1)
        sat_c['label'] = sat_c.apply(lambda r: f"{r['quantidade']} ({r['pct']}%)", axis=1)
        fig1 = px.bar(sat_c, x='satisfacao', y='quantidade',
                      color='satisfacao', color_discrete_map=COR_MAP,
                      text='label', template=TEMPLATE,
                      labels={'satisfacao':'Satisfação','quantidade':'Produtos'},
                      title='Distribuição da variável alvo: Satisfação do Cliente')
        fig1.update_traces(textposition='outside', textfont_size=12, marker_line_width=0)
        fig1.update_layout(**PLOT_CFG, showlegend=False, height=340)
        fig1.update_yaxes(gridcolor='#1e3a5f')
        st.plotly_chart(fig1, use_container_width=True)
        pct_alta = (df['satisfacao_cliente']=='Alta').mean()*100
        pct_baixa = (df['satisfacao_cliente']=='Baixa').mean()*100
        st.markdown(f'<div class="ins">💡 <strong>{pct_alta:.0f}%</strong> dos produtos têm satisfação Alta. A classe Baixa representa apenas <strong>{pct_baixa:.1f}%</strong> — desbalanceamento severo que impacta a acurácia dos classificadores.</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Identificar quais categorias têm mais produtos e entender a representatividade de cada segmento (prior bayesiano).</div>', unsafe_allow_html=True)
        vol_c = df['categoria'].value_counts().reset_index()
        vol_c.columns = ['categoria','quantidade']
        fig2 = px.bar(vol_c, x='quantidade', y='categoria', orientation='h',
                      color='quantidade', color_continuous_scale=['#1e3a5f','#4d9de0'],
                      text='quantidade', template=TEMPLATE,
                      labels={'categoria':'','quantidade':'Produtos'},
                      title='Volume de produtos por categoria')
        fig2.update_traces(textposition='outside', textfont_size=11, marker_line_width=0)
        fig2.update_layout(**PLOT_CFG, coloraxis_showscale=False, height=340)
        fig2.update_xaxes(gridcolor='#1e3a5f')
        st.plotly_chart(fig2, use_container_width=True)
        top = vol_c.iloc[0]
        st.markdown(f'<div class="ins">💡 <strong>{top["categoria"]}</strong> lidera com <strong>{top["quantidade"]} produtos</strong>. As 3 maiores categorias somam 97% do dataset — isso eleva o prior bayesiano dessas categorias em qualquer análise por segmento.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Bloco 2 — Preços ─────────────────────────────────────────────────────
    st.markdown('<div class="sec">2 · Análise de Preços</div>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Comparar o ticket médio entre categorias e identificar segmentos de alto e baixo valor.</div>', unsafe_allow_html=True)
        preco_c = (df.groupby('categoria')['preco_original']
                     .mean().reset_index()
                     .sort_values('preco_original', ascending=True))
        preco_c['fmt'] = preco_c['preco_original'].apply(lambda x: f"R$ {x:,.0f}")
        fig3 = px.bar(preco_c, x='preco_original', y='categoria', orientation='h',
                      color='preco_original', color_continuous_scale=['#1a3a5c','#e74c3c'],
                      text='fmt', template=TEMPLATE,
                      labels={'preco_original':'Preço médio (R$)','categoria':''},
                      title='Preço médio por categoria (R$)')
        fig3.update_traces(textposition='outside', textfont_size=11, marker_line_width=0)
        fig3.update_layout(**PLOT_CFG, coloraxis_showscale=False, height=360)
        fig3.update_xaxes(gridcolor='#1e3a5f')
        st.plotly_chart(fig3, use_container_width=True)
        top_p = preco_c.iloc[-1]
        st.markdown(f'<div class="ins">💡 <strong>{top_p["categoria"]}</strong> tem preço médio de <strong>{top_p["fmt"]}</strong> — itens premium como Smart TVs e eletrodomésticos puxam esse valor.</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Visualizar a distribuição dos preços e confirmar assimetria positiva (cauda direita) gerada pelos outliers premium.</div>', unsafe_allow_html=True)
        fig4 = px.histogram(df, x='preco_original', nbins=40,
                            color_discrete_sequence=['#4d9de0'],
                            template=TEMPLATE,
                            labels={'preco_original':'Preço original (R$)','count':'Frequência'},
                            title='Distribuição dos preços (assimetria positiva = 4,55)')
        med = df['preco_original'].median()
        fig4.add_vline(x=med, line_dash="dash", line_color="#f39c12",
                       annotation_text=f"Mediana R${med:,.0f}",
                       annotation_font_color="#f39c12")
        fig4.add_vline(x=df['preco_original'].mean(), line_dash="dot", line_color="#e74c3c",
                       annotation_text=f"Média R${df['preco_original'].mean():,.0f}",
                       annotation_font_color="#e74c3c")
        fig4.update_layout(**PLOT_CFG, height=360)
        fig4.update_yaxes(gridcolor='#1e3a5f', title='Frequência')
        fig4.update_xaxes(gridcolor='#1e3a5f')
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown(f'<div class="ins">💡 Média (vermelho) muito acima da mediana (laranja) confirma assimetria positiva. A maioria dos produtos custa menos de R$ 200 — a média é distorcida por produtos de alto valor.</div>', unsafe_allow_html=True)

    # Boxplot preço × satisfação + teste t
    st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Verificar se o preço influencia o nível de satisfação (Teste t: Alta vs. Baixa).</div>', unsafe_allow_html=True)
    if len(df[df['satisfacao_cliente']=='Alta']) > 1 and len(df[df['satisfacao_cliente']=='Baixa']) > 1:
        t_stat, p_t = ttest_ind(
            df[df['satisfacao_cliente']=='Alta']['preco_original'],
            df[df['satisfacao_cliente']=='Baixa']['preco_original']
        )
        sig_txt = f"t = {t_stat:.2f} | p = {p_t:.4f} → {'diferença significativa ✅' if p_t < 0.05 else 'sem diferença significativa'}"
    else:
        sig_txt = "Dados insuficientes para o Teste t com os filtros atuais."
    fig5 = px.box(df, x='satisfacao_cliente', y='preco_original',
                  color='satisfacao_cliente', color_discrete_map=COR_MAP,
                  category_orders={'satisfacao_cliente':['Alta','Regular','Baixa']},
                  template=TEMPLATE, points="outliers",
                  labels={'satisfacao_cliente':'Satisfação','preco_original':'Preço original (R$)'},
                  title=f'Preço × Satisfação — Teste t: {sig_txt}')
    fig5.update_layout(**PLOT_CFG, showlegend=False, height=320)
    fig5.update_yaxes(gridcolor='#1e3a5f')
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown('<div class="ins">💡 Produtos com satisfação Alta têm preço médio maior (R$ 360) que os de satisfação Baixa (R$ 122) — diferença estatisticamente significativa (p = 0,033). Isso reflete a composição: classe Alta é dominada por Eletrônicos premium.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Bloco 3 — Avaliações & Descontos ─────────────────────────────────────
    st.markdown('<div class="sec">3 · Avaliações, Descontos e Correlações</div>', unsafe_allow_html=True)
    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Investigar a relação entre desconto e nota usando Correlação de Spearman (adequada para dados não-normais).</div>', unsafe_allow_html=True)
        fig6 = px.scatter(df, x='percentual_desconto', y='nota_avaliacao',
                          color='satisfacao_cliente', color_discrete_map=COR_MAP,
                          opacity=0.5, template=TEMPLATE, trendline="ols",
                          trendline_scope="overall", trendline_color_override="#ffffff",
                          labels={'percentual_desconto':'Desconto (%)','nota_avaliacao':'Nota','satisfacao_cliente':'Satisfação'},
                          title=f'Desconto × Nota  |  ρ Spearman = {rho_desc:.3f} (correlação fraca negativa)')
        fig6.update_layout(**PLOT_CFG, height=370)
        fig6.update_xaxes(gridcolor='#1e3a5f')
        fig6.update_yaxes(gridcolor='#1e3a5f')
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown(f'<div class="ins">💡 Correlação de Spearman ρ = {rho_desc:.3f} (p < 0,001): descontos maiores associam-se levemente a notas menores — produtos com promoções agressivas tendem a ter qualidade percebida ligeiramente inferior.</div>', unsafe_allow_html=True)

    with col_f:
        st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Comparar a nota média por categoria e verificar se as diferenças são estatisticamente significativas (ANOVA).</div>', unsafe_allow_html=True)
        nota_c = (df.groupby('categoria')['nota_avaliacao']
                    .mean().reset_index()
                    .sort_values('nota_avaliacao', ascending=True))
        nota_c['fmt'] = nota_c['nota_avaliacao'].apply(lambda x: f"{x:.2f}")
        fig7 = px.bar(nota_c, x='nota_avaliacao', y='categoria', orientation='h',
                      color='nota_avaliacao',
                      color_continuous_scale=[[0,'#e74c3c'],[0.5,'#f39c12'],[1,'#2ecc71']],
                      range_color=[3.5,4.5], text='fmt', template=TEMPLATE,
                      labels={'nota_avaliacao':'Nota média','categoria':''},
                      title='Nota média por categoria')
        fig7.update_traces(textposition='outside', textfont_size=11, marker_line_width=0)
        fig7.update_layout(**PLOT_CFG, coloraxis_showscale=False, height=370)
        fig7.update_xaxes(gridcolor='#1e3a5f', range=[3.4,4.7])
        st.plotly_chart(fig7, use_container_width=True)
        if df['categoria'].nunique() > 1:
            grupos = [g['nota_avaliacao'].values for _,g in df.groupby('categoria') if len(g) > 1]
            if len(grupos) > 1:
                f_stat, p_anova = f_oneway(*grupos)
                st.markdown(f'<div class="ins">💡 ANOVA: F = {f_stat:.2f} | p = {p_anova:.4f} — {"diferenças significativas entre categorias ✅" if p_anova < 0.05 else "diferenças não significativas"}</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sec">4 · Matriz de Correlação</div>', unsafe_allow_html=True)
    st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Medir a força e a direção da relação entre as variáveis. Valores próximos de 1 ou -1 indicam conexões fortes; valores próximos de 0 indicam que as variáveis não possuem relação linear clara.</div>', unsafe_allow_html=True)

    # Variáveis numéricas para o cálculo
    cols_num = ['preco_original', 'preco_com_desconto', 'percentual_desconto', 'nota_avaliacao', 'total_avaliacoes']
    lbl_num  = ['Preço original', 'Preço c/ desc.', '% Desconto', 'Nota', 'Nº avaliações']
    
    # Cálculo da matriz de correlação simples (completa e direta)
    corr_sp = df[cols_num].corr(method='spearman').round(2)

    # Criação do gráfico simplificado (Matriz Completa para leitura direta)
    fig8 = go.Figure(go.Heatmap(
        z=corr_sp.values, 
        x=lbl_num, 
        y=lbl_num,
        colorscale=[[0, '#c0392b'], [0.5, '#1e3a5f'], [1, '#2980b9']], # Suas cores mantidas
        zmin=-1, 
        zmax=1,
        text=corr_sp.values, 
        texttemplate="%{text}", 
        textfont={"size": 13, "color": "white"},
        hoverongaps=False
    ))
    
    fig8.update_layout(
        template=TEMPLATE, 
        **PLOT_CFG, 
        height=380,
        title='Matriz de Correlação de Spearman Simplificada'
    )
    st.plotly_chart(fig8, use_container_width=True)

    # ── GUIA DE LEITURA RÁPIDA (Explicação dos Números) ───────────────────────
    st.markdown("""
    <div style="background:#0d1b2a; border:1px solid #1e3a5f; border-radius:8px; padding:12px; margin-bottom:15px;">
        <span style="color:#7eb8f7; font-weight:bold; font-size:13px; text-transform:uppercase; letter-spacing:.05em;">Como ler os números deste gráfico:</span>
        <ul style="color:#e8f0fe; font-size:13px; margin-top:6px; margin-bottom:0px; padding-left:20px;">
            <li><strong>Perto de +1,00 (Azul):</strong> Correlação Positiva Forte. Quando uma variável sobe, a outra também sobe (ex: Preço Original e Preço com Desconto).</li>
            <li><strong>Perto de -1,00 (Vermelho):</strong> Correlação Negativa Forte. Quando uma variável sobe, a outra desce.</li>
            <li><strong>Perto de 0,00 (Azul Escuro/Fundo):</strong> Sem correlação. Uma variável não interfere no comportamento da outra (distante de 1 e -1).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Insights do Dashboard
    i1, i2, i3 = st.columns(3)
    with i1: 
        st.markdown('<div class="ins">💡 <strong>Preço × Nota:</strong> ρ ≈ 0,03 (Perto de 0) — O preço cobrado não interfere e não possui impacto na nota final do produto.</div>', unsafe_allow_html=True)
    with i2: 
        st.markdown('<div class="ins">💡 <strong>Desconto × Nota:</strong> ρ ≈ −0,15 (Levemente Negativo) — Produtos com descontos muito agressivos registram notas sutilmente menores.</div>', unsafe_allow_html=True)
    with i3: 
        st.markdown('<div class="ins">💡 <strong>Avaliações × Nota:</strong> ρ ≈ 0,18 (Levemente Positivo) — Itens populares com grande volume de vendas sustentam médias ligeiramente melhores.</div>', unsafe_allow_html=True)

    st.divider()


    # ── Bloco 5 — Popularidade & Satisfação Baixa ────────────────────────────
    st.markdown('<div class="sec">5 · Popularidade e Satisfação Baixa por Categoria</div>', unsafe_allow_html=True)
    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Identificar quais categorias geram maior engajamento (volume mediano de avaliações).</div>', unsafe_allow_html=True)
        pop_c = (df.groupby('categoria')['total_avaliacoes']
                   .median().reset_index()
                   .sort_values('total_avaliacoes', ascending=True))
        pop_c['fmt'] = pop_c['total_avaliacoes'].apply(lambda x: f"{x:,.0f}")
        fig9 = px.bar(pop_c, x='total_avaliacoes', y='categoria', orientation='h',
                      color='total_avaliacoes', color_continuous_scale=['#1a1a5c','#9b59b6'],
                      text='fmt', template=TEMPLATE,
                      labels={'total_avaliacoes':'Mediana de avaliações','categoria':''},
                      title='Popularidade por categoria (mediana de avaliações)')
        fig9.update_traces(textposition='outside', textfont_size=11, marker_line_width=0)
        fig9.update_layout(**PLOT_CFG, coloraxis_showscale=False, height=360)
        fig9.update_xaxes(gridcolor='#1e3a5f')
        st.plotly_chart(fig9, use_container_width=True)
        top_pop = pop_c.iloc[-1]
        st.markdown(f'<div class="ins">💡 <strong>{top_pop["categoria"]}</strong> tem a maior mediana de avaliações ({top_pop["fmt"]}), confirmando protagonismo em engajamento do consumidor.</div>', unsafe_allow_html=True)

    with col_h:
        st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Analisar onde se concentra a insatisfação (nota &lt; 3,5) — alvo da análise bayesiana de cenário de risco.</div>', unsafe_allow_html=True)
        baixa = df[df['satisfacao_cliente']=='Baixa']
        if len(baixa) == 0:
            st.info("Nenhum produto com satisfação Baixa no conjunto filtrado.")
        else:
            baixa_c = baixa['categoria'].value_counts().reset_index()
            baixa_c.columns = ['categoria','quantidade']
            fig10 = px.pie(baixa_c, names='categoria', values='quantidade', hole=0.42,
                           color_discrete_sequence=['#e74c3c','#c0392b','#e67e22','#d35400','#e84393'],
                           template=TEMPLATE, title='Satisfação Baixa por categoria')
            fig10.update_traces(textposition='inside', textinfo='percent+label', textfont_size=11)
            fig10.update_layout(**PLOT_CFG, showlegend=False, height=360)
            st.plotly_chart(fig10, use_container_width=True)
            top_b = baixa_c.iloc[0]
            st.markdown(f'<div class="ins">💡 <strong>{top_b["categoria"]}</strong> concentra mais insatisfação ({top_b["quantidade"]} produtos) — expectativas mais altas dos consumidores de tecnologia.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Bloco 6 — Outliers ───────────────────────────────────────────────────
    st.markdown('<div class="sec">6 · Análise de Outliers (Método IQR)</div>', unsafe_allow_html=True)
    st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Verificar se os outliers de preço são erros ou produtos legítimos de alto valor. Justificativa técnica para mantê-los no dataset.</div>', unsafe_allow_html=True)

    col_i, col_j = st.columns([1,1])
    Q1_v = df_raw['preco_original'].quantile(0.25)
    Q3_v = df_raw['preco_original'].quantile(0.75)
    IQR_v = Q3_v - Q1_v
    lim_sup = Q3_v + 1.5*IQR_v
    out_df  = df[df['is_outlier']]
    norm_df = df[~df['is_outlier']]

    with col_i:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=norm_df['preco_original'], name='Normais',
                                 marker_color='#4d9de0', boxmean=True,
                                 fillcolor='rgba(77,157,224,0.2)'))
        fig_box.add_trace(go.Box(y=out_df['preco_original'], name='Outliers',
                                 marker_color='#e74c3c', boxmean=True,
                                 fillcolor='rgba(231,76,60,0.2)'))
        fig_box.add_hline(y=lim_sup, line_dash='dash', line_color='#f39c12',
                          annotation_text=f"Limite IQR: R$ {lim_sup:,.0f}",
                          annotation_font_color='#f39c12')
        fig_box.update_layout(template=TEMPLATE, **PLOT_CFG, height=380,
                              title=f'Boxplot IQR — Q1=R${Q1_v:.0f} | Q3=R${Q3_v:.0f} | Limite=R${lim_sup:.0f}',
                              yaxis_title='Preço original (R$)')
        fig_box.update_yaxes(gridcolor='#1e3a5f')
        st.plotly_chart(fig_box, use_container_width=True)

    with col_j:
        st.markdown('<div class="obj" style="margin-top:.4rem">📌 Top 15 produtos outliers com maior preço</div>', unsafe_allow_html=True)
        if len(out_df) == 0:
            st.info("Nenhum outlier no conjunto filtrado.")
        else:
            top_out = (out_df[['nome_produto','categoria','preco_original','nota_avaliacao','satisfacao_cliente']]
                       .sort_values('preco_original', ascending=False).head(15).reset_index(drop=True))
            top_out.index += 1
            top_out['preco_original'] = top_out['preco_original'].apply(lambda x: f"R$ {x:,.0f}")
            top_out['nome_produto']   = top_out['nome_produto'].str[:42] + '…'
            top_out.columns = ['Produto','Categoria','Preço','Nota','Satisfação']
            st.dataframe(top_out, use_container_width=True, height=340)

    pct_out = df['is_outlier'].sum()/len(df)*100 if len(df) > 0 else 0
    st.markdown(f'<div class="ins">💡 <strong>{df["is_outlier"].sum()} outliers</strong> ({pct_out:.1f}%) mantidos com flag <code>is_outlier</code>. São produtos legítimos de alto valor — removê-los distorceria as análises de preço e os parâmetros gaussianos do Bayes.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Bloco 7 — Satisfação por Categoria ───────────────────────────────────
    st.markdown('<div class="sec">7 · Satisfação por Categoria (Stacked Bar)</div>', unsafe_allow_html=True)
    st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Visualizar, para cada categoria, a proporção de produtos em cada nível de satisfação — evidenciando quais segmentos têm melhor desempenho relativo.</div>', unsafe_allow_html=True)

    ct = (pd.crosstab(df['categoria'], df['satisfacao_cliente'], normalize='index') * 100).round(1).reset_index()
    cats_order = ct.set_index('categoria').get('Alta', pd.Series(dtype=float)).sort_values(ascending=False).index.tolist()

    fig_stack = go.Figure()
    for sat, cor in [('Alta','#4d9de0'),('Regular','#2ecc71'),('Baixa','#e74c3c')]:
        if sat in ct.columns:
            fig_stack.add_trace(go.Bar(
                name=sat, x=ct['categoria'], y=ct[sat],
                marker_color=cor, text=ct[sat].apply(lambda x: f"{x:.0f}%"),
                textposition='inside', textfont_size=11
            ))
    fig_stack.update_layout(
        barmode='stack', template=TEMPLATE, **PLOT_CFG, height=380,
        title='Proporção de satisfação por categoria (%)',
        yaxis_title='Proporção (%)', xaxis_title='',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    fig_stack.update_yaxes(gridcolor='#1e3a5f')
    st.plotly_chart(fig_stack, use_container_width=True)
    st.markdown('<div class="ins">💡 Produtos de Escritório e Brinquedos têm a maior proporção de satisfação Alta. Carros, Instrumentos e Saúde têm proporção 0% Alta — mas atenção: essas categorias têm apenas 1–2 produtos no dataset.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Bloco 8 — Estatísticas Descritivas ───────────────────────────────────
    st.markdown('<div class="sec">8 · Estatísticas Descritivas</div>', unsafe_allow_html=True)
    st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Resumo estatístico completo com medidas de posição, dispersão e forma. Assimetria e curtose confirmam outliers e justificam o uso de correlação de Spearman.</div>', unsafe_allow_html=True)

    cols_d = ['preco_original','preco_com_desconto','percentual_desconto','nota_avaliacao','total_avaliacoes']
    lbl_d  = ['Preço original (R$)','Preço c/ desc. (R$)','Desconto (%)','Nota','Nº avaliações']
    desc   = df[cols_d].describe().round(2)
    skew_v = df[cols_d].skew(numeric_only=True).round(2)
    kurt_v = df[cols_d].kurtosis(numeric_only=True).round(2)
    tabela = pd.DataFrame({
        'Variável':   lbl_d,
        'Média':      desc.loc['mean'].values,
        'Mediana':    desc.loc['50%'].values,
        'Desv. Pad.': desc.loc['std'].values,
        'Mín':        desc.loc['min'].values,
        'Máx':        desc.loc['max'].values,
        'Assimetria': skew_v.values,
        'Curtose':    kurt_v.values,
    })
    st.dataframe(tabela.set_index('Variável'), use_container_width=True)
    st.markdown('<div class="ins">💡 Assimetria elevada em Preço (4,55) e Nº Avaliações (5,67) confirma distribuições não-normais — justificando Spearman em vez de Pearson para correlações.</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="text-align:center;color:#4a7fa5;font-size:12px;padding-top:.5rem">📊 Seção 1 — EDA · Amazon Products Sales · Estatística e Probabilidade</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — CLASSIFICAÇÃO PROBABILÍSTICA
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec2">🤖 Seção 2 — Classificação Probabilística</div>', unsafe_allow_html=True)
    st.markdown("Insira os atributos de um produto e compare as predições dos três métodos: **Teorema de Bayes**, **Árvore de Decisão** e **KNN**.")

    st.divider()

    # ── Painel de input ───────────────────────────────────────────────────────
    st.markdown("### ✏️ Atributos do Produto")
    inp1, inp2, inp3, inp4 = st.columns(4)

    with inp1:
        preco_input = st.number_input(
            "Preço original (R$)",
            min_value=0.0, max_value=10000.0, value=150.0, step=10.0,
            help="Preço original do produto em Reais (convertido de INR)"
        )
    with inp2:
        desconto_input = st.number_input(
            "Percentual de desconto (%)",
            min_value=0.0, max_value=100.0, value=45.0, step=5.0,
            help="Percentual de desconto aplicado sobre o preço original"
        )
    with inp3:
        avals_input = st.number_input(
            "Número de avaliações",
            min_value=0, max_value=500000, value=8000, step=500,
            help="Total de avaliações recebidas pelo produto"
        )
    with inp4:
        cats_lista = sorted(df_raw['categoria'].unique())
        cat_input = st.selectbox(
            "Categoria do produto",
            options=cats_lista,
            help="Categoria principal do produto"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1,4])
    with col_btn:
        classificar = st.button("🔍 Classificar produto", type="primary", use_container_width=True)

    st.divider()

    # ── Resultados ────────────────────────────────────────────────────────────
    if classificar or True:  # mostra sempre, atualiza com o botão
        # Bayes manual
        probs_bayes = classificar_bayes(
            preco_input, desconto_input, avals_input,
            modelos['priori'], modelos['lk_params'], modelos['classes']
        )
        pred_bayes = max(probs_bayes, key=probs_bayes.get)

        # Árvore de Decisão
        cat_enc = modelos['le'].transform([cat_input])[0]
        X_novo_raw = pd.DataFrame([[preco_input, desconto_input, avals_input, cat_enc]],
                                  columns=['preco_original','percentual_desconto','total_avaliacoes','categoria_enc'])
        pred_dt = modelos['dt'].predict(X_novo_raw)[0]
        probs_dt = dict(zip(modelos['dt'].classes_, modelos['dt'].predict_proba(X_novo_raw)[0]))

        # KNN
        X_novo_sc = modelos['scaler'].transform(X_novo_raw)
        pred_knn = modelos['knn'].predict(X_novo_sc)[0]
        probs_knn = dict(zip(modelos['knn'].classes_, modelos['knn'].predict_proba(X_novo_sc)[0]))

        # ── Cards de resultado ────────────────────────────────────────────────
        st.markdown("### 🎯 Predição dos Três Métodos")
        r1, r2, r3 = st.columns(3)

        def resultado_card(col, titulo, icone, pred, cor_borda, extra=""):
            cor = {"Alta":"#4d9de0","Regular":"#2ecc71","Baixa":"#e74c3c"}.get(pred, "#888")
            with col:
                st.markdown(
                    f'<div class="resultado-card" style="border-color:{cor_borda};background:#0d1b2a;">'
                    f'<div style="font-size:13px;color:#7eb8f7;font-weight:600;margin-bottom:8px">{icone} {titulo}</div>'
                    f'<div style="font-size:32px;font-weight:700;color:{cor}">{pred}</div>'
                    f'<div style="font-size:11px;color:#4a7fa5;margin-top:6px">{extra}</div>'
                    f'</div>', unsafe_allow_html=True
                )

        resultado_card(r1, "Teorema de Bayes", "🧮", pred_bayes, "#4d9de0",
                       f"P = {probs_bayes[pred_bayes]*100:.1f}%")
        resultado_card(r2, "Árvore de Decisão", "🌳", pred_dt, "#2ecc71",
                       f"Confiança: {probs_dt.get(pred_dt,0)*100:.1f}%")
        resultado_card(r3, "KNN (k=7)", "👥", pred_knn, "#9b59b6",
                       f"Confiança: {probs_knn.get(pred_knn,0)*100:.1f}%")

        # Consenso
        votos = [pred_bayes, pred_dt, pred_knn]
        consenso = max(set(votos), key=votos.count)
        n_consenso = votos.count(consenso)
        cor_cons = {"Alta":"#4d9de0","Regular":"#2ecc71","Baixa":"#e74c3c"}.get(consenso,"#888")
        st.markdown(
            f'<div style="text-align:center;margin-top:.8rem;padding:.8rem;background:#1e3a5f;'
            f'border-radius:10px;font-size:16px;color:#fff">'
            f'🗳️ Consenso: <strong style="color:{cor_cons};font-size:20px">{consenso}</strong> '
            f'— {n_consenso}/3 métodos concordam</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # ── Comparação visual de probabilidades ───────────────────────────────
        st.markdown("### 📊 Comparação Visual das Probabilidades por Classe")
        pb1, pb2, pb3 = st.columns(3)

        def prob_bars(col, titulo, probs_dict):
            with col:
                st.markdown(f'<div class="tab-header">{titulo}</div>', unsafe_allow_html=True)
                for classe, cor in [("Alta","#4d9de0"),("Regular","#2ecc71"),("Baixa","#e74c3c")]:
                    p = probs_dict.get(classe, 0)
                    largura = max(p*100, 2)
                    st.markdown(
                        f'<div style="display:flex;align-items:center;margin:4px 0">'
                        f'<div style="width:70px;font-size:12px;color:#e8f0fe">{classe}</div>'
                        f'<div style="flex:1;background:#1e3a5f;border-radius:6px;overflow:hidden;height:24px">'
                        f'<div style="width:{largura:.1f}%;background:{cor};height:100%;'
                        f'display:flex;align-items:center;padding-left:6px;'
                        f'font-size:11px;font-weight:600;color:#fff;min-width:36px">'
                        f'{p*100:.1f}%</div></div></div>',
                        unsafe_allow_html=True
                    )

        prob_bars(pb1, "🧮 Bayes Manual", probs_bayes)
        prob_bars(pb2, "🌳 Árvore de Decisão", probs_dt)
        prob_bars(pb3, "👥 KNN (k=7)", probs_knn)

        st.divider()

        # ── Raciocínio Bayesiano detalhado ────────────────────────────────────
        st.markdown("### 🧮 Detalhamento do Cálculo Bayesiano")
        st.markdown('<div class="obj">Demonstração passo a passo de P(C|X) = P(C) × P(X|C) / P(X) para os atributos informados.</div>', unsafe_allow_html=True)

        bayes_rows = []
        post_raw = {}
        for c in sorted(modelos['classes']):
            prior = modelos['priori'][c]
            lk_p  = gaussiana(preco_input,    *modelos['lk_params'][c]['preco_original'])
            lk_d  = gaussiana(desconto_input,  *modelos['lk_params'][c]['percentual_desconto'])
            lk_a  = gaussiana(avals_input,     *modelos['lk_params'][c]['total_avaliacoes'])
            joint = prior * lk_p * lk_d * lk_a
            post_raw[c] = joint
            bayes_rows.append({
                'Classe': c,
                'P(C) — priori': f"{prior:.4f}",
                'P(Preço|C)':    f"{lk_p:.2e}",
                'P(Desc.|C)':    f"{lk_d:.4f}",
                'P(Avals.|C)':   f"{lk_a:.2e}",
                'Produto bruto': f"{joint:.2e}",
                'P(C|X) — final': f"{probs_bayes[c]*100:.2f}%"
            })

        df_bayes = pd.DataFrame(bayes_rows).set_index('Classe')
        st.dataframe(df_bayes, use_container_width=True)

        st.markdown(
            f'<div class="ins">💡 <strong>Como interpretar:</strong> '
            f'P(C) é a frequência da classe no dataset (priori). '
            f'P(X|C) é a densidade gaussiana de cada feature dado que a classe é C — calculada com μ e σ da classe. '
            f'O produto bruto é proporcional à posteriori. '
            f'P(C|X) final é o produto bruto normalizado para que os três valores somem 100%.</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # ── Gráfico radar de probabilidades ──────────────────────────────────
        st.markdown("### 🕸️ Comparação em Radar — Probabilidades por Classe e Método")
        classes_radar = ["Alta","Regular","Baixa"]
        fig_radar = go.Figure()
        for nome, probs, cor in [
            ("Bayes Manual", probs_bayes, "#4d9de0"),
            ("Árvore de Decisão", probs_dt, "#2ecc71"),
            ("KNN (k=7)", probs_knn, "#9b59b6"),
        ]:
            vals = [probs.get(c,0)*100 for c in classes_radar]
            vals.append(vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=classes_radar + [classes_radar[0]],
                fill='toself', name=nome, line_color=cor,
                fillcolor=f'rgba({int(cor[1:3],16)},{int(cor[3:5],16)},{int(cor[5:7],16)},0.15)',
                opacity=0.85
            ))
        fig_radar.update_layout(
            template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e8f0fe', height=400,
            polar=dict(
                bgcolor='rgba(13,27,42,0.8)',
                radialaxis=dict(visible=True, range=[0,100], gridcolor='#1e3a5f', tickcolor='#7eb8f7'),
                angularaxis=dict(gridcolor='#1e3a5f')
            ),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.15),
            title='Probabilidades (%) por classe e método — mesmo produto de teste'
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()

        # ── Desempenho dos modelos no conjunto de teste ───────────────────────
        st.markdown("### 📈 Desempenho dos Modelos no Conjunto de Teste (30% do dataset)")
        st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Comparar acurácia e F1-score dos três métodos para fundamentar qual modelo é mais confiável para este problema.</div>', unsafe_allow_html=True)

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            # Acurácia
            modelos_nomes = ['Árvore de Decisão\n(max_depth=5)', 'KNN\n(k=7)', 'Bayes Manual\n(Gaussiano)']
            acuracias = [modelos['acc_dt']*100, modelos['acc_knn']*100, 0.0]

            # Calcular acurácia Bayes no test set (cache)
            @st.cache_data
            def acc_bayes_test(hash_key):
                y_pred_bayes = []
                for _, row in modelos['X_test'].iterrows():
                    probs = classificar_bayes(
                        row['preco_original'], row['percentual_desconto'], row['total_avaliacoes'],
                        modelos['priori'], modelos['lk_params'], modelos['classes']
                    )
                    y_pred_bayes.append(max(probs, key=probs.get))
                return accuracy_score(modelos['y_test'], y_pred_bayes), y_pred_bayes

            acc_b, y_pred_bayes_test = acc_bayes_test("v1")
            acuracias[2] = acc_b*100

            fig_acc = go.Figure(go.Bar(
                x=modelos_nomes, y=acuracias,
                marker_color=['#2196F3','#4CAF50','#FF9800'],
                text=[f"{a:.2f}%" for a in acuracias],
                textposition='outside', textfont_size=14
            ))
            fig_acc.update_layout(
                template=TEMPLATE, **PLOT_CFG, height=360,
                title='Acurácia Global por Modelo (%)',
                yaxis=dict(range=[0,80], gridcolor='#1e3a5f'),
                showlegend=False
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_m2:
            # F1-score por classe
            classes_ord = ['Alta','Regular','Baixa']
            f1_dt    = f1_score(modelos['y_test'], modelos['y_pred_dt'],    labels=classes_ord, average=None, zero_division=0)
            f1_knn   = f1_score(modelos['y_test'], modelos['y_pred_knn'],   labels=classes_ord, average=None, zero_division=0)
            f1_bayes = f1_score(modelos['y_test'], y_pred_bayes_test,       labels=classes_ord, average=None, zero_division=0)

            fig_f1 = go.Figure()
            for nome, cor in [('Árvore','#2196F3'),
                  ('KNN','#4CAF50'),
                  ('Bayes','#FF9800')]:
                vals_f1 = {'Árvore': f1_dt, 'KNN': f1_knn, 'Bayes': f1_bayes}[nome]
                fig_f1.add_trace(go.Bar(
                    name=nome, x=classes_ord, y=vals_f1,
                    marker_color={'Árvore':'#2196F3','KNN':'#4CAF50','Bayes':'#FF9800'}[nome]
                ))
            fig_f1.update_layout(
                barmode='group', template=TEMPLATE, **PLOT_CFG, height=360,
                title='F1-Score por Classe e Modelo',
                yaxis=dict(range=[0,1], gridcolor='#1e3a5f', title='F1-Score'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        # Tabela resumo
        st.markdown("#### Resumo Comparativo")
        resumo = pd.DataFrame({
            'Modelo': ['Árvore de Decisão (max_depth=5)', 'KNN (k=7)', 'Bayes Manual (Gaussiano)'],
            'Acurácia': [f"{modelos['acc_dt']*100:.2f}%", f"{modelos['acc_knn']*100:.2f}%", f"{acc_b*100:.2f}%"],
            'F1 — Alta':    [f"{f1_dt[0]:.3f}", f"{f1_knn[0]:.3f}", f"{f1_bayes[0]:.3f}"],
            'F1 — Regular': [f"{f1_dt[1]:.3f}", f"{f1_knn[1]:.3f}", f"{f1_bayes[1]:.3f}"],
            'F1 — Baixa':   [f"{f1_dt[2]:.3f}", f"{f1_knn[2]:.3f}", f"{f1_bayes[2]:.3f}"],
            'Vantagem principal': [
                'Maior acurácia — aprende cortes diretos na nota',
                'F1 equilibrado entre classes majoritárias',
                'Único com F1 > 0 para classe Baixa — probabilidades calibradas'
            ]
        }).set_index('Modelo')
        st.dataframe(resumo, use_container_width=True)

        st.markdown(
            '<div class="warn">⚠️ <strong>Sobre a classe Baixa:</strong> '
            'representa apenas 2,8% dos dados. Árvore e KNN têm F1 = 0 para essa classe — '
            'o desbalanceamento severo impede recall positivo. '
            'O Bayes Manual é o único capaz de expressar incerteza calibrada para todas as classes.</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # ── Matrizes de confusão ──────────────────────────────────────────────
        st.markdown("### 🔲 Matrizes de Confusão")
        st.markdown('<div class="obj">🎯 <strong>Objetivo:</strong> Identificar exatamente quais classes cada modelo confunde, além da acurácia global.</div>', unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        classes_ord = ['Alta','Regular','Baixa']

        for col_mc, nome, y_pred, cor_escala in [
            (mc1, "Árvore de Decisão", modelos['y_pred_dt'],    [[0,'#0d1b2a'],[1,'#2196F3']]),
            (mc2, "KNN (k=7)",          modelos['y_pred_knn'],   [[0,'#0d1b2a'],[1,'#4CAF50']]),
            (mc3, "Bayes Manual",        y_pred_bayes_test,       [[0,'#0d1b2a'],[1,'#FF9800']]),
        ]:
            y_true_aligned = [c for c in modelos['y_test'] if c in classes_ord]
            y_test_list = list(modelos['y_test'])
            y_pred_list = list(y_pred)
            cm = confusion_matrix(y_test_list, y_pred_list, labels=classes_ord)

            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=classes_ord, y=classes_ord,
                colorscale=cor_escala,
                text=cm, texttemplate="%{text}",
                textfont={"size":16,"color":"white"},
                showscale=False
            ))
            fig_cm.update_layout(
                template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font_color='#e8f0fe',
                height=280, margin=dict(t=48,b=8,l=8,r=8),
                title=f'{nome}',
                xaxis_title='Previsto', yaxis_title='Real',
                yaxis=dict(autorange='reversed')
            )
            with col_mc:
                st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown(
            '<div class="ins">💡 As diagonais principais mostram acertos. '
            'A Árvore confunde Alta e Regular mas raramente erra entre extremos. '
            'O Bayes erra mais no geral, mas consegue identificar alguns casos da classe Baixa (diagonal [2,2] > 0).</div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown('<div style="text-align:center;color:#4a7fa5;font-size:12px;padding-top:.5rem">🤖 Seção 2 — Classificação Probabilística · Amazon Products Sales · Estatística e Probabilidade</div>', unsafe_allow_html=True)