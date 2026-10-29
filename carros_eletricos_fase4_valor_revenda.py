# Databricks notebook source
# MAGIC %md
# MAGIC # Carros Elétricos — Fase 4 / Projeto 1: Previsão do Valor de Revenda e Depreciação
# MAGIC
# MAGIC Modelo de regressão que prevê o **preço de revenda de um EV usado** a partir da marca, ano (idade), quilometragem e capacidade da bateria — treinado em **53k anúncios reais de EVs** do mercado norte-americano (cars.com via Hugging Face), enriquecidos com as especificações de bateria da `silver_ev_specs`.
# MAGIC
# MAGIC **Diferencial de portfólio (Step 8):** análise de sensibilidade que compara quanto a **degradação da bateria** (perda de capacidade útil) afeta o preço final em comparação com a **quilometragem total**.
# MAGIC
# MAGIC | Step | Descrição | Output |
# MAGIC |---|---|---|
# MAGIC | 1 | Setup — imports e experimento MLflow | experimento MLflow |
# MAGIC | 2 | Ingestão anúncios EV usados (cars.com) | `bronze_ev_used_listings` (Delta) |
# MAGIC | 3 | Silver — limpeza + join bateria (specs) | `silver_ev_used_prices` (Delta) |
# MAGIC | 4 | EDA — preços, idades, km, marcas | describe + tabelas |
# MAGIC | 5 | Features, leakage e split por modelo | treino / teste |
# MAGIC | 6 | Treino Ridge / Lasso / RF / XGBoost com MLflow | runs + métricas |
# MAGIC | 7 | Avaliação — resíduos, R² por modelo | diagnóstico |
# MAGIC | 8 | **Sensibilidade: bateria vs quilometragem** | curvas + taxa de troca |
# MAGIC | 9 | Refit final + conclusões | modelo MLflow final |

# COMMAND ----------

# ==============================================================================
# STEP 1: SETUP — IMPORTS E EXPERIMENTO MLFLOW
# ==============================================================================
# Objectivo: Preparar o ambiente (bibliotecas, caminhos e experimento MLflow)
# que todos os steps seguintes vão usar.
#
# Input:  —
# Output: experimento MLflow ev_project_revenda + caminho do parquet bruto

import os, re
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

import mlflow
from mlflow.models import infer_signature

try:
    from xgboost import XGBRegressor
    TEM_XGBOOST = True
except Exception:
    TEM_XGBOOST = False
    print("⚠️ xgboost indisponível → será usado GradientBoosting (sklearn)")

VOLUME_RAW = "/Volumes/workspace/ev_project/raw_files/cars_com"
PARQUET_RAW = f"{VOLUME_RAW}/usa_used_cars.parquet"
ANO_REF = 2026  # ano de referência para a idade do veículo

MLFLOW_EXPERIMENTO = "/Users/gonaldomichael@gmail.com/ev_project_revenda"
mlflow.set_experiment(MLFLOW_EXPERIMENTO)
print(f"✅ Experimento MLflow: {MLFLOW_EXPERIMENTO}")
print(f"XGBoost disponível: {TEM_XGBOOST}")

# COMMAND ----------

# ==============================================================================
# STEP 2: INGESTÃO — ANÚNCIOS DE EVs USADOS (CARS.COM) → BRONZE
# ==============================================================================
# Objectivo: Carregar os anúncios reais do mercado norte-americano (1,6M
# veículos) e isolar os EVs usados (fuel = Electric) na camada bronze.
#
# Input:  parquet bruto no volume raw_files/cars_com (139 MB, descarregado da
#         HF `KTHESAT/usa-used-car-market-cars-com` se ainda não existir)
# Output: workspace.ev_project.bronze_ev_used_listings (Delta)

if not os.path.exists(PARQUET_RAW):
    os.makedirs(VOLUME_RAW, exist_ok=True)
    url = ("https://huggingface.co/datasets/KTHESAT/usa-used-car-market-cars-com/"
           "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(PARQUET_RAW, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Descarregado: {os.path.getsize(PARQUET_RAW)/1e6:.0f} MB")

total = pd.read_parquet(PARQUET_RAW, columns=["fuel"])
ev = pd.read_parquet(PARQUET_RAW, filters=[("fuel", "=", "Electric")]).copy()
print(f"Anúncios no dataset: {len(total):,} | EVs (fuel=Electric): {len(ev):,}")

ev["ingested_at"] = pd.Timestamp.utcnow()
(spark.createDataFrame(ev)
   .write.format("delta").mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable("workspace.ev_project.bronze_ev_used_listings"))
print("✅ Tabela: workspace.ev_project.bronze_ev_used_listings")
display(spark.table("workspace.ev_project.bronze_ev_used_listings")
         .groupBy("make", "model").count()
         .orderBy("count", ascending=False).limit(12))

# COMMAND ----------

# ==============================================================================
# STEP 3: SILVER — LIMPEZA + ENRIQUECIMENTO COM A BATERIA (SPECS)
# ==============================================================================
# Objectivo: Limpar os anúncios (preços inválidos, kms absurdos, duplicados)
# e enriquecer cada anúncio com a capacidade da bateria (kWh) vinda da
# `silver_ev_specs` (ev-database.org) via matching hierárquico de nome.
#
# Input:  bronze_ev_used_listings + silver_ev_specs
# Output: workspace.ev_project.silver_ev_used_prices (Delta)
#
# Operações:
#   - dedup por VIN; preço $1k–$300k; quilometragem ≤ 250k milhas
#   - idade_anos = 2026 − ano (≥ 0); quilometragem em milhas e km
#   - matching de bateria em 3 níveis: nome exacto → nome-base → 1º token

def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()

STOP = {"long", "range", "performance", "standard", "plus", "pro", "max",
        "awd", "rwd", "edition", "pack", "sport", "premium", "luxury",
        "launch", "gran", "coupe", "sedan", "suv"}

def base_nome(md):
    """'model 3 long range' → 'model 3' (nome-base do modelo, sem versão)."""
    base = []
    for t in md.split():
        if t in STOP and base:
            break
        base.append(t)
        if len(base) >= 2:
            break
    return " ".join(base)

specs = (spark.table("workspace.ev_project.silver_ev_specs")
         .select("brand", "model", "battery_kwh").toPandas())
specs["mk"] = specs["brand"].apply(norm)
specs["md"] = specs["model"].apply(norm)
specs["base"] = specs["md"].apply(base_nome)

dict_exato = (specs.groupby(["mk", "md"])["battery_kwh"].median().to_dict())
dict_base = (specs.groupby(["mk", "base"])["battery_kwh"].median().to_dict())
specs["tok"] = specs["base"].str.split().str[0]
dict_tok = (specs.groupby(["mk", "tok"])["battery_kwh"].median().to_dict())

def apanha_bateria(mk, md):
    """Matching hierárquico: exacto → base → 1º token. Devolve (kWh, nível)."""
    if (mk, md) in dict_exato:
        return dict_exato[(mk, md)], "exacto"
    b = base_nome(md)
    if (mk, b) in dict_base:
        return dict_base[(mk, b)], "base"
    t = md.split()[0] if md.split() else ""
    if t and (mk, t) in dict_tok:
        return dict_tok[(mk, t)], "token"
    return np.nan, "sem_match"

raw = spark.table("workspace.ev_project.bronze_ev_used_listings").toPandas()
antes = len(raw)
raw = raw.drop_duplicates(subset="vin", keep="first")
raw = raw[(raw["price_usd"] > 1000) & (raw["price_usd"] < 300000)]
raw = raw[(raw["mileage_miles"] >= 0) & (raw["mileage_miles"] <= 250000)]
raw = raw[(raw["year"] >= 2011) & (raw["year"] <= 2027)]
print(f"Limpeza: {antes:,} → {len(raw):,} anúncios "
      f"(dedup VIN, preço/kms/ano válidos)")

raw["mk"] = raw["make"].apply(norm)
raw["md"] = raw["model"].apply(norm)
bat = raw.apply(lambda r: apanha_bateria(r["mk"], r["md"]), axis=1, result_type="expand")
raw[["battery_kwh", "match_nivel"]] = bat

silver = (raw.drop(columns=["mk", "md"])
          .assign(idade_anos=lambda d: (ANO_REF - d["year"]).clip(lower=0),
                  quilometragem_km=lambda d: (d["mileage_miles"] * 1.609).round(0))
          .rename(columns={"mileage_miles": "quilometragem_miles"}))

cob = silver["battery_kwh"].notna().mean() * 100
print(f"Cobertura de bateria: {cob:.1f}% "
      f"({silver['battery_kwh'].notna().sum():,} com kWh da specs)")
print(silver["match_nivel"].value_counts().to_string())

(spark.createDataFrame(silver)
   .write.format("delta").mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable("workspace.ev_project.silver_ev_used_prices"))
print("✅ Tabela: workspace.ev_project.silver_ev_used_prices")

# COMMAND ----------

# ==============================================================================
# STEP 4: EDA — PREÇOS, IDADES, QUILOMETRAGEM E MARCAS
# ==============================================================================
# Objectivo: Compreender o mercado dos EVs usados antes de modelar:
# distribuição de preços, depreciação por idade, efeito da quilometragem
# e diferenças entre marcas.
#
# Input:  workspace.ev_project.silver_ev_used_prices
# Output: describe + tabela de depreciação por idade

dados = spark.table("workspace.ev_project.silver_ev_used_prices").toPandas()
print(f"{len(dados):,} anúncios | {dados['make'].nunique()} marcas | "
      f"{dados.groupby(['make', 'model']).ngroups} modelos")
print(f"Preço: ${dados['price_usd'].min():,.0f} – ${dados['price_usd'].max():,.0f} "
      f"| mediana ${dados['price_usd'].median():,.0f}")
display(dados[["price_usd", "idade_anos", "quilometragem_miles",
                "battery_kwh"]].describe().T)

dep = (dados.groupby("idade_anos")["price_usd"]
       .median().round(0).rename("preco_mediano_usd"))
display(dep.head(8).to_frame())

top_marcas = (dados.groupby("make").agg(n=("price_usd", "size"),
                                         preco_mediano=("price_usd", "median"),
                                         idade_media=("idade_anos", "mean"))
              .sort_values("n", ascending=False).head(12))
display(top_marcas)

# COMMAND ----------

# ==============================================================================
# STEP 5: FEATURES, RASTREIO DE LEAKAGE E SPLIT POR MODELO
# ==============================================================================
# Objectivo: Definir as features (marca, idade, quilometragem, bateria),
# validar que nenhuma é derivada do preço e separar treino/teste POR MODELO
# de carro (o teste mede a generalização para modelos nunca vistos).
#
# Input:  dataframe dados (Step 4)
# Output: treino / teste (modelos disjuntos)
#
# Nota de leakage: msrp_usd (preço novo de tabela) fica FORA das features —
# é 0 na maioria dos anúncios e seria quase-proxy do preço. A capacidade da
# bateria é uma característica do modelo, legitimamente conhecida.

# treinar apenas onde há bateria real da specs (para o sinal de kWh ser limpo)
ml = dados[dados["battery_kwh"].notna()].copy()
print(f"Anúncios com bateria (specs): {len(ml):,} de {len(dados):,}")

FEATURES_NUM = ["idade_anos", "quilometragem_miles", "battery_kwh", "cpo"]
FEATURES_CAT = ["make"]
TARGET = "price_usd"

corr = ml[FEATURES_NUM + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
display(corr.to_frame("correlacao_com_preco"))

ml["grupo_modelo"] = ml["make"] + "|" + ml["model"]
split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, te_idx = next(split.split(ml, groups=ml["grupo_modelo"]))
treino, teste = ml.iloc[tr_idx], ml.iloc[te_idx]
print(f"Treino: {treino['grupo_modelo'].nunique()} modelos, {len(treino):,} anúncios")
print(f"Teste:  {teste['grupo_modelo'].nunique()} modelos, {len(teste):,} anúncios")
assert set(treino["grupo_modelo"]) & set(teste["grupo_modelo"]) == set()
print("\n✅ Split por modelo: o teste mede a generalização para modelos nunca vistos")

# COMMAND ----------

# ==============================================================================
# STEP 6: TREINO — RIDGE / LASSO / RANDOM FOREST / XGBOOST (MLFLOW)
# ==============================================================================
# Objectivo: Treinar 4 famílias de modelos com hiperparâmetros fixos e
# compará-las no holdout, com tracking completo em MLflow.
#
# Input:  treino / teste (Step 5)
# Output: runs MLflow + tabela comparativa (RMSE $, MAE $, MAPE %, R²)

X_tr, y_tr = treino[FEATURES_NUM + FEATURES_CAT], treino[TARGET]
X_te, y_te = teste[FEATURES_NUM + FEATURES_CAT], teste[TARGET]

def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)

pre_lin = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler())]), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)])
pre_arv = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)])

MODELOS = {
    "ridge": Pipeline([("pre", pre_lin), ("m", Ridge(alpha=10.0))]),
    "lasso": Pipeline([("pre", pre_lin), ("m", Lasso(alpha=100.0, max_iter=5000))]),
    "random_forest": Pipeline([("pre", pre_arv),
                                ("m", RandomForestRegressor(n_estimators=300,
                                                             min_samples_leaf=3,
                                                             random_state=42,
                                                             n_jobs=-1))]),
}
if TEM_XGBOOST:
    MODELOS["xgboost"] = Pipeline([("pre", pre_arv),
                                   ("m", XGBRegressor(n_estimators=400,
                                                      learning_rate=0.05,
                                                      max_depth=5,
                                                      random_state=42,
                                                      n_jobs=-1))])
else:
    MODELOS["gradient_boosting"] = Pipeline([("pre", pre_arv),
                                            ("m", GradientBoostingRegressor(
                                                n_estimators=300,
                                                learning_rate=0.05,
                                                max_depth=3,
                                                random_state=42))])

resultados, preds = [], {}
for nome, pipe in MODELOS.items():
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    preds[nome] = pred
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae = mean_absolute_error(y_te, pred)
    r2 = r2_score(y_te, pred)
    mp = mape(y_te.values, pred)
    assinatura = infer_signature(X_tr.head(100), pipe.predict(X_tr.head(100)))
    with mlflow.start_run(run_name=f"revenda_{nome}"):
        mlflow.log_param("modelo", nome)
        mlflow.log_param("split", "group_shuffle_por_modelo")
        mlflow.log_param("n_train", len(treino))
        mlflow.log_param("n_test", len(teste))
        mlflow.log_metric("rmse_usd", rmse)
        mlflow.log_metric("mae_usd", mae)
        mlflow.log_metric("mape_pct", mp)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(pipe, name="modelo",
                                 signature=assinatura, input_example=X_tr.head(3))
    resultados.append({"modelo": nome, "rmse_usd": rmse, "mae_usd": mae,
                       "mape_pct": mp, "r2": r2})

display(pd.DataFrame(resultados).sort_values("rmse_usd"))

# COMMAND ----------

# ==============================================================================
# STEP 7: AVALIAÇÃO — RESÍDUOS, R² POR MODELO E IMPORTÂNCIA
# ==============================================================================
# Objectivo: Diagnosticar o melhor modelo: previsão vs real, resíduos (média
# ~0, distribuição aprox. normal), R² por modelo de carro e importância das
# features (agregando os one-hot da marca).
#
# Input:  resultados do Step 6 + holdout
# Output: gráfico de diagnóstico + R² por modelo

melhor_nome = min(resultados, key=lambda r: r["rmse_usd"])["modelo"]
melhor_pipe = MODELOS[melhor_nome]
pred = preds[melhor_nome]
resid = y_te.values - pred
print(f"Melhor modelo: {melhor_nome}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(y_te, pred, s=8, alpha=0.4)
lims = [0, y_te.max()]
ax.plot(lims, lims, "r--", lw=1)
ax.set(xlabel="preço real ($)", ylabel="preço previsto ($)",
       title=f"{melhor_nome}: previsto vs real (holdout)")
plt.tight_layout()
plt.show()

nt = stats.normaltest(resid)
print(f"Resíduos: média=${resid.mean():,.0f} | desvio=${resid.std():,.0f} | "
      f"normaltest p={nt.pvalue:.3f}")

teste2 = teste.copy()
teste2["preco_previsto"] = pred
r2_mod = pd.DataFrame([
    {"modelo_carro": g["grupo_modelo"].iloc[0], "n": len(g),
     "r2": r2_score(g[TARGET], g["preco_previsto"])}
    for _, g in teste2.groupby("grupo_modelo") if len(g) >= 20
]).sort_values("r2")
display(r2_mod)

# importância: somar os one-hot da marca numa única linha "marca (one-hot)"
if hasattr(melhor_pipe[-1], "feature_importances_"):
    nomes = list(melhor_pipe[0].get_feature_names_out())
    imp = pd.Series(melhor_pipe[-1].feature_importances_, index=nomes)
    agrupada = {}
    for n, v in imp.items():
        chave = "marca (one-hot)" if n.startswith("cat__") else n
        agrupada[chave] = agrupada.get(chave, 0) + v
    imp_agg = pd.Series(agrupada).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    imp_agg.plot(kind="barh", ax=ax, color="#55A868")
    ax.set(title=f"Importância das features — {melhor_nome}",
           xlabel="importância relativa (ganho)")
    plt.tight_layout()
    plt.show()
else:
    print("Modelo linear: coeficientes disponíveis no MLflow (one-hot por marca)")

# COMMAND ----------

# ==============================================================================
# STEP 8: SENSIBILIDADE — DEGRADAÇÃO DA BATERIA vs QUILOMETRAGEM
# ==============================================================================
# Objectivo (diferencial do portfólio): quantificar quanto a degradação da
# bateria (perda de capacidade útil) afecta o preço de revenda, em comparação
# com a quilometragem total.
#
# Input:  melhor modelo (Step 7) + anúncios com bateria
# Output: curvas de sensibilidade + taxa de troca ($/1% SoH vs $/10k milhas)
#
# METODOLOGIA E ASSUMPÇÃO (explícita): o dataset de anúncios não regista o SoH
# real de cada carro — a bateria disponível é a NOMINAL do modelo. O modelo
# aprende como o preço depende da capacidade; SIMULAMOS a degradação
# reduzindo a capacidade efectiva (kWh × SoH). É um cenário de análise, não
# uma medição de mercado.

# carro de referência: o modelo mais comum do dataset com bateria
ref_modelo = (ml[ml["battery_kwh"].notna()].groupby("grupo_modelo")
              .size().sort_values(ascending=False).index[0])
ref = ml[ml["grupo_modelo"] == ref_modelo]
linha_ref = {
    "idade_anos": float(ref["idade_anos"].median()),
    "quilometragem_miles": float(ref["quilometragem_miles"].median()),
    "battery_kwh": float(ref["battery_kwh"].median()),
    "cpo": int(ref["cpo"].mode().iloc[0]),
    "make": ref["make"].mode().iloc[0],
}
print(f"Carro de referência: {ref_modelo}")
print(f"  idade={linha_ref['idade_anos']:.0f} anos | "
      f"km={linha_ref['quilometragem_miles']:,.0f} milhas "
      f"({linha_ref['quilometragem_miles']*1.609:,.0f} km) | "
      f"bateria={linha_ref['battery_kwh']:.0f} kWh")

def prev(ref_l, **over):
    linha = dict(ref_l)
    linha.update(over)
    df = pd.DataFrame([linha])
    return float(melhor_pipe.predict(df[FEATURES_NUM + FEATURES_CAT])[0])

# Curva A — quilometragem 0 → 200k milhas (bateria a 100%)
graus_km = np.arange(0, 200001, 5000)
preco_km = [prev(linha_ref, quilometragem_miles=float(k)) for k in graus_km]

# Curva B — SoH 100% → 60% (quilometragem fixa na mediana)
graus_soh = np.arange(100, 59, -1)
preco_soh = [prev(linha_ref, battery_kwh=linha_ref["battery_kwh"] * s / 100)
             for s in graus_soh]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(graus_km / 1000, preco_km, color="#C44E52", lw=2)
ax1.set(title="Preço × quilometragem (bateria 100%)",
        xlabel="milhares de milhas", ylabel="preço previsto ($)")
ax2.plot(graus_soh, preco_soh, color="#4C72B0", lw=2, marker="o", ms=2)
ax2.invert_xaxis()
ax2.set(title="Preço × degradação da bateria (km fixos)",
        xlabel="SoH da bateria (%)", ylabel="preço previsto ($)")
plt.tight_layout()
plt.show()

# Taxas de troca — efeitos marginais nos pontos de referência
p0 = prev(linha_ref)
p_mais_10k = prev(linha_ref, quilometragem_miles=linha_ref["quilometragem_miles"] + 10000)
custo_10k_milhas = p0 - p_mais_10k
p_soh_90 = prev(linha_ref, battery_kwh=linha_ref["battery_kwh"] * 0.90)
custo_10_soh = p0 - p_soh_90
p_soh_80 = prev(linha_ref, battery_kwh=linha_ref["battery_kwh"] * 0.80)
custo_20_soh = p0 - p_soh_80

equivalencia = custo_10_soh / (custo_10k_milhas / 10000) if custo_10k_milhas > 0 else np.nan

print(f"Preço base previsto: ${p0:,.0f}")
print(f"+10k milhas:   −${custo_10k_milhas:,.0f} "
      f"(${custo_10k_milhas/10:,.0f}/milha)")
print(f"SoH 100→90%: −${custo_10_soh:,.0f} (${custo_10_soh/10:,.0f}/ponto de SoH)")
print(f"SoH 100→80%: −${custo_20_soh:,.0f}")
print(f"\n➡️ Perder 10% de bateria ≈ percorrer mais {equivalencia:,.0f} milhas "
      f"({equivalencia*1.609:,.0f} km) no conta-quilómetros")

custo_10k_km = custo_10k_milhas / 1.609  # 10k km = 6.215 milhas

tabela_sens = pd.DataFrame({
    "factor": ["+10k milhas", "+10k km", "SoH −10 p.p.", "SoH −20 p.p."],
    "queda_preco_usd": [custo_10k_milhas, custo_10k_km,
                        custo_10_soh, custo_20_soh],
    "pct_do_preco": [custo_10k_milhas / p0 * 100, custo_10k_km / p0 * 100,
                      custo_10_soh / p0 * 100, custo_20_soh / p0 * 100],
})
display(tabela_sens.round(1))

# COMMAND ----------

# ==============================================================================
# STEP 9: REFIT FINAL EM TODOS OS DADOS + CONCLUSÕES
# ==============================================================================
# Objectivo: Refitar o melhor pipeline em todos os anúncios com bateria e
# logar o modelo final em MLflow, pronto para registo em UC.
#
# Input:  melhor modelo (Step 7) + dataset completo com bateria
# Output: run MLflow "final_refit" com o modelo final logado

X_tudo = ml[FEATURES_NUM + FEATURES_CAT]
y_tudo = ml[TARGET]
melhor_pipe.fit(X_tudo, y_tudo)
assinatura = infer_signature(X_tudo.head(100), melhor_pipe.predict(X_tudo.head(100)))
with mlflow.start_run(run_name="final_refit"):
    mlflow.log_param("modelo", melhor_nome)
    mlflow.log_param("n_final_fit", len(ml))
    mlflow.log_param("features_num", len(FEATURES_NUM))
    mlflow.log_param("features_cat", len(FEATURES_CAT))
    mlflow.sklearn.log_model(melhor_pipe, name="modelo_final",
                             signature=assinatura, input_example=X_tudo.head(3))

print(f"✅ Modelo final ({melhor_nome}) refitado em {len(ml):,} anúncios "
      f"e logado no experimento ev_project_revenda")
print("\nResumo Projeto 1 — Valor de Revenda:")
print(f"  - Dados: {len(dados):,} anúncios EV usados (cars.com), "
      f"{len(ml):,} com bateria da specs")
melhor_res = min(resultados, key=lambda r: r["rmse_usd"])
print(f"  - Melhor modelo: {melhor_nome} | "
      f"RMSE=${melhor_res['rmse_usd']:,.0f} | "
      f"MAPE={melhor_res['mape_pct']:.1f}%")
print("  - Sensibilidade: comparar degradação da bateria vs quilometragem "
      "(Step 8) — análise baseada na dependência aprendida do preço na "
      "capacidade efectiva da bateria (assumpção documentada)")
