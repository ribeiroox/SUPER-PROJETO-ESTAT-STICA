Amazon Products — Estatística e Probabilidade
Projeto final da disciplina de Estatística e Probabilidade — CESUPA, Belém/PA, 2025.
Análise exploratória, classificação probabilística com Teorema de Bayes e algoritmos de machine learning aplicados ao Amazon Products Sales Dataset (Amazon Índia).

👥 Equipe

- MARIA ISADORA SANTOS 
- EVELLYN RIBEIRO DA SILVA

📁 Estrutura do Repositório
📦 amazon-products-estatistica
├── Amazon_completo.ipynb            # Notebook principal (limpeza, EDA, Bayes, ML)
├── amazon.csv                       # Dataset original (Amazon Índia — Kaggle)
├── dashboard.py                     # Dashboard interativo (Streamlit)
├── Relatorio_Tecnico_Completo.docx  # Relatório técnico do projeto
└── README.md

📊 Sobre o Dataset

Fonte: Amazon Products Sales Dataset — Kaggle
Domínio: E-commerce (Amazon Índia)
Instâncias: 1.465 produtos
Atributos: 16 colunas (preços, descontos, notas, avaliações, categorias)
Variável-alvo criada: satisfacao_cliente (Alta / Regular / Baixa), derivada da nota de avaliação por engenharia de atributos


🔬 O que foi feito
1. Limpeza e Tratamento dos Dados

Tradução e padronização das colunas (inglês → português)
Conversão de tipos: remoção de símbolos monetários (₹, $), vírgulas e % para float64
Conversão cambial INR → BRL (fator 0,0522)
Tratamento de valores ausentes e caracteres corrompidos
Simplificação da hierarquia de categorias
Detecção de outliers via IQR (mantidos — produtos premium legítimos)
Criação da variável-alvo satisfacao_cliente

2. Análise Exploratória (EDA)

Distribuição da satisfação dos clientes
Preço médio e desconto médio por categoria
Heatmap de correlação entre variáveis numéricas
Relação entre desconto e nota de avaliação
Produtos mais populares por volume de avaliações

3. Teorema de Bayes (implementação manual)

Cálculo das probabilidades a priori P(C) direto do dataset
Verossimilhança modelada com função gaussiana implementada do zero
Cálculo das probabilidades a posteriori P(C | X) normalizadas
3 cenários de probabilidade condicional analisados sobre o dataset
Classificador interativo: usuário informa atributos e recebe a probabilidade de cada classe

4. Algoritmos de Classificação
ModeloAcuráciaÁrvore de Decisão (max_depth=5)~58%KNN — k=7 (com StandardScaler)~56%Bayes Manual (gaussiano)~54%
5. Dashboard Interativo (Streamlit)

Seção 1: visualizações da EDA com insights analíticos
Seção 2: classificador interativo — usuário preenche atributos e recebe predição dos 3 métodos
Seção 3: comparação de métricas dos modelos


▶️ Como executar o notebook
Opção 1 — Google Colab (recomendado, sem instalar nada):

Acesse colab.research.google.com
Faça upload do Amazon_completo.ipynb e do amazon.csv
Execute as células em ordem

Opção 2 — Local (VSCode):
bashpip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook Amazon_completo.ipynb

⚠️ O arquivo amazon.csv precisa estar na mesma pasta do notebook.


🖥️ Como usar o Dashboard
Pré-requisito: instalar o Streamlit
bashpip install streamlit
Rodar o dashboard:
bashstreamlit run dashboard.py
O dashboard abrirá automaticamente no navegador em http://localhost:8501.

⚠️ O arquivo amazon.csv precisa estar na mesma pasta do dashboard.py.


🤖 Declaração de Uso de IA Generativa
Este projeto utilizou ferramentas de IA generativa (Claude — Anthropic) como apoio em:

Depuração de código Python e correção de erros de execução
Estruturação do relatório técnico
Geração do dashboard interativo

O uso teve como objetivo de aprendizado compreender as etapas de implementação dos algoritmos e visualizar os resultados de forma mais clara. Todas as decisões analíticas, justificativas técnicas e interpretações dos resultados foram elaboradas pela equipe.


