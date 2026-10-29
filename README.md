# Carros Elétricos — Fase 4 / Projeto 1: Previsão do Valor de Revenda e Depreciação de EVs

Modelo de machine learning que prevê o **preço de revenda de um EV usado** a partir da marca, idade, quilometragem e capacidade da bateria, treinado em **anúncios reais do mercado norte-americano**, com uma **análise de sensibilidade** que quantifica o impacto da degradação da bateria (SoH) face à quilometragem no preço final.

## Dados

* **Fonte:** `KTHESAT/usa-used-car-market-cars-com` (Hugging Face) — 1,6M anúncios de carros usados de cars.com; filtrados **53.138 EVs** (fuel = Electric), anos 2008–2027.
* **Enriquecimento:** capacidade da bateria (kWh) por modelo, via matching hierárquico (nome exacto → nome-base → 1º token) com especificações de EVs (ev-database.org) — cobertura de **67,4%** (35.015 anúncios com bateria).
* **Limpeza:** deduplicação por VIN, preço $1k–$300k, quilometragem ≤ 250k milhas → 51.949 anúncios válidos.

## Metodologia

1. **Bronze/Silver** (Delta Lake): ingestão idempotente do parquet, limpeza, idade do veículo, quilometragem em milhas/km, join da bateria.
2. **Features:** marca (one-hot), idade, quilometragem, capacidade da bateria, CPO. O MSRP fica de fora (risco de leakage).
3. **Split por modelo de carro** (GroupShuffleSplit): o holdout contém **21 modelos nunca vistos no treino** — mede a generalização real.
4. **Modelos comparados** (com tracking MLflow): Ridge, Lasso, Random Forest e Gradient Boosting (fallback de XGBoost).
5. **Sensibilidade (Step 8):** degradação da bateria simulada como capacidade efectiva (kWh × SoH), porque os anúncios não registem o SoH real — assumpção documentada no código.

## Resultados

| Modelo | RMSE | MAPE | R² |
|---|---|---|---|
| **Ridge (melhor)** | **$33.054** | **29,3%** | **0,54** |
| Lasso | $33.952 | 29,2% | 0,51 |
| Gradient Boosting | $41.436 | 28,1% | 0,28 |
| Random Forest | $47.205 | 32,3% | 0,06 |

Na generalização para **modelos nunca vistos**, os modelos lineares vencem as árvores — estas dependem demasiado da identidade dos modelos observados no treino.

### Sensibilidade: bateria vs quilometragem (Tesla Model 3, 4 anos, 56k milhas, 75 kWh, preço previsto $31.501)

| Factor | Queda de preço | % do preço |
|---|---|---|
| +10k milhas | −$1.643 | 5,2% |
| +10k km | −$1.021 | 3,2% |
| SoH −10 p.p. | −$6.409 | 20,3% |
| SoH −20 p.p. | −$12.819 | 40,7% |

**Conclusão:** perder 10% da capacidade da bateria equivale a percorrer mais ~39.000 milhas (63.000 km) — o mercado valoriza a capacidade da bateria ~4× mais do que a quilometragem (≈ $854/kWh implícitos).

## Como executar

O código completo está em `carros_eletricos_fase4_valor_revenda.py` (formato SOURCE do Databricks): importar directamente num workspace Databricks (ou Azure/AWS) e executar as células por ordem. Requer:

* Databricks com Unity Catalog e MLflow;
* tabelas `silver_ev_specs`, `bronze_ev_used_listings` e `silver_ev_used_prices` (criadas pelo próprio notebook);
* bibliotecas: scikit-learn, pandas, numpy, scipy, matplotlib, mlflow (pré-instaladas no Databricks ML Runtime); xgboost opcional (fallback automático para GradientBoosting).

## Estrutura do notebook (9 steps)

1. Setup — imports e experimento MLflow
2. Ingestão bronze — anúncios EV usados (cars.com)
3. Silver — limpeza + enriquecimento com a bateria (matching hierárquico)
4. EDA — preços, idades, quilometragem, marcas
5. Features, verificação de leakage e split por modelo
6. Treino — Ridge / Lasso / Random Forest / Gradient Boosting (MLflow)
7. Avaliação — resíduos, R² por modelo, diagnóstico
8. **Sensibilidade — degradação da bateria vs quilometragem** (diferencial)
9. Refit final em todos os dados + conclusões

## Limitações (documentadas no código)

* O SoH real de cada anúncio não é observado — a degradação é simulada escalando a capacidade efectiva;
* ~33% dos anúncios ficam sem bateria (modelos só dos EUA ausentes na base de specs) e são excluídos do treino;
* Holdout por modelo induz subprevisão sistemática em modelos de luxo nunca vistos (ex.: Mercedes G-Class, Porsche Taycan).
