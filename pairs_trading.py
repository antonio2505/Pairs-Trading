# -*- coding: utf-8 -*-
"""Pairs Trading.ipynb

# PAIRS TRADING STRATEGY

PAIRS TRADING est une stratégie de market-neutral trading  permettant aux traders de profiter pratiquement toutes les conditions du marché : tendance haussière, tendance baissière ou mouvement latéral. Cette stratégie est classée comme une stratégie d'arbitrage statistique et de négociation de convergence(Hedging) [wikipedia](https://en.wikipedia.org/wiki/Pairs_trade)

- Qu'est ce que market-neutral trading strategy ?

market-neutral trading strategy est un type de stratégie d'investissement par un investisseur ou un gestionnaire d'investissement qui cherche à profiter à la fois de la hausse et de la baisse des prix sur un ou plusieurs marchés tout en essayant d'éviter complètement  prise la de risque de marché.

Le Trading de Paires est classée comme une stratégie d'arbitrage statistique.

- Qu'est ce que la stratégie d'arbitrage statistique ?

L'arbitrage est une stratégie d'investissement dans laquelle un investisseur achète et vend simultanément un actif sur différents marchés pour profiter d'une différence de prix et générer un profit.

## 1- PARTIE I: IMPLEMENTATION DE LA STRATEGY

### Installation des libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
idx = pd.IndexSlice
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from sklearn.model_selection import train_test_split
import pandas as pd
import yahoo_fin.stock_info as si
import yfinance as yf

"""1. Selection des Stocks et  Identification de Pairs de stocks cointegrer.
2. Effectuer le test de stationariter pour la selection des pairs.
3. Generer le Signal de Trading avec Z-Score test.
4. Calcule de de Gain et perte de notre portefeuil

## 1. Selection des Stocks et Identification de Pairs de stocks cointegrer.
"""

stocks = si.tickers_sp500(include_company_data=True)
stocks = stocks[stocks['GICS Sector']=="Information Technology"]
len(stocks) #74 stocks

#date de debut a maintenant
fromdate = datetime(2010, 1, 1)
today = datetime.now()

#Prix des stocks de 2010 a maintenant
price = pd.DataFrame()
for ticker in stocks.Symbol:
    price[ticker] = yf.download(ticker,start=fromdate, end=today)["Adj Close"]
    price.dropna(axis = 1, inplace=True) #axis =1 supprime les stocks moins de 10 ans
len(price) #3072 ligne de donnees       # 60 stocks valable

price.head()

#train test split
train_close, test_close = train_test_split(price, test_size=0.5, shuffle=False)

"""
Person Correlation,
"""
fig, ax = plt.subplots(figsize=(30,20))
sns.heatmap(train_close.pct_change().corr(method ='pearson'), ax=ax, cmap='coolwarm', annot=True, fmt=".2f") #spearman
ax.set_title('Assets Correlation Matrix')

"""### Person Correlation,

Le coefficient de corrélation de Pearson varie entre +1 et -1 et est une mesure linéaire de la relation entre deux variables. La valeur +1 indique une forte corrélation positive, zéro indique l'absence de relation et -1 indique une forte relation négative. Nous pouvons voir dans la carte thermique ci-dessous qu'il existe plusieurs paires avec une forte corrélation positive.
"""

# function to find cointegrated pairs
def find_cointegrated_pairs(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            result = coint(data[keys[i]], data[keys[j]])
            pvalue_matrix[i, j] = result[1]
            if result[1] < 0.05:
                pairs.append((keys[i], keys[j]))
    return pvalue_matrix, pairs

"""## Calcule du p-value

"""

# calculate p-values and plot as a heatmap
pvalues, pairs = find_cointegrated_pairs(train_close)
print(pairs)
fig, ax = plt.subplots(figsize=(20,12))
sns.heatmap(pvalues, xticklabels = train_close.columns,
                yticklabels = train_close.columns, cmap = 'RdYlGn_r', annot = True, fmt=".2f",
                mask = (pvalues >= 0.99))
ax.set_title('Assets Cointregration Matrix p-values Between Pairs')
plt.tight_layout()
#('CTXS', 'HPQ')

"""Analysons également le résultat du test de cointégration. Nous pouvons voir dans ci-dessus qu'il existe de nombreuses paires avec une pvalue inférieure à 0,05. Cela signifie que pour ces paires, nous pouvons rejeter l'hypothèse nulle et qu'elles peuvent être cointégrées

### Effectuer un test stationnaire pour la paire sélectionnée

Maintenant, nous avons de nombreux candidats de paires pour la stratégie où la pvalue est inférieure à 0,05. La sélection de la bonne paire est de la plus haute importance car la stratégie ne fonctionnera pas bien si les prix évoluent exactement ensemble. Ils doivent être divergents et mean-reverting pour que notre stratégie soit rentable.

Allons-y avec la paire ('CTXS', 'HPQ'), CTXS: Citrix Systems, Inc et HPQ: HP Inc,et testons davantage la stationnarité de la propagation à l'aide du test Augmented Dickey-Fuller. Il est important que la propagation soit stationnaire. Une série chronologique est considérée comme stationnaire si des paramètres tels que la moyenne et la variance ne changent pas dans le temps. Nous allons d'abord calculer le ratio de couverture(hedge ratio) entre ces deux stockers en utilisant la régression OLS. Ensuite, en utilisant le ratio de couverture(hedge ratio), nous calculerons le spread et exécuterons le test Augmented Dickey-Fuller.
"""

asset1 = 'CTXS'
asset2 = 'HPQ'

# create a train dataframe of 2 assets
train = pd.DataFrame()
train['asset1'] = train_close[asset1]
train['asset2'] = train_close[asset2]


# visualizer prices
ax = train[['asset1','asset2']].plot(figsize=(12, 6), title = 'Prix Journalier de {} et {}'.format(asset1,asset2))
ax.set_ylabel("Closing Price")
ax.grid(True);

# effections notre OLS regression
model=sm.OLS(train.asset1, train.asset2).fit()

# visualization du resultat du test
plt.rc('figure', figsize=(12, 8))
plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 16}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.subplots_adjust(left=0.2, right=0.8, top=0.7, bottom=0.1)

print('Hedge Ratio = ', model.params[0])



# calculate la dispersion
spread = train.asset1 - model.params[0] * train.asset2

# Plot the spread
ax = spread.plot(figsize=(12, 6), title = "Pair's Spread")
ax.set_ylabel("Spread")
ax.grid(True);

# Augmented Dickey-Fuller test
adf = adfuller(spread, maxlag = 1)
print('Critical Value = ', adf[0])

# probablity critical values
print(adf[4])

"""### Interpretation de la propagation(spread)
Une valeur  R-square et une pvalue proche de zéro de la régression OLS suggèrent une très forte corrélation entre ces deux actions. La propagation(spread) semble stationnaire et la valeur Dickey-Fuller est de 0.005, ce qui est inférieur à la valeur au niveau de signification de 5 %. Par conséquent, nous sommes en mesure de rejeter l'hypothèse nulle selon laquelle la propagation a une racine unitaire et pouvons conclure qu'il s'agit de stationnarité.

## Générer des signaux de trading à l'aide du z-score

Nous avons utilisé l'ensemble de données d'entraînement jusqu'à présent pour finaliser la paire d'actions pour notre stratégie. À partir de maintenant, nous utiliserons l'ensemble de données de test pour nous assurer que la génération de signaux de trading et le backtesting utilisent un ensemble de données d'échantillon. Nous utiliserons le score z du rapport entre les deux cours boursiers pour générer des signaux de trading et définir les seuils supérieur et inférieur. Cela nous indiquera à quel point un prix est éloigné de la valeur moyenne de la population. S'il est positif et que la valeur est supérieure aux seuils supérieurs, le cours de l'action est supérieur à la valeur moyenne du cours. Par conséquent, son prix devrait baisser, nous voulons donc vendre cette action et acheter l'autre.

1. Definir la fonction **zscore**
"""

# calculate z-score
def zscore(series):
 return (series - series.mean()) / np.std(series)

"""### 2. Creer une dataframe pour les Signaux des 2 stocks avec leur prix en utilisant les donnee du test dataset"""

# creer une dataframe pour les signaux de trading
signals = pd.DataFrame()
signals['asset1'] = test_close[asset1]
signals['asset2'] = test_close[asset2]
ratios = signals.asset1 / signals.asset2

"""### 3. Calculez le score z pour le rapport et définissez les seuils supérieur et inférieur avec plus et moins un écart type."""

# z-score and upper and lower seuils
signals['z'] = zscore(ratios)
signals['z upper limit'] = np.mean(signals['z']) + np.std(signals['z'])
signals['z lower limit'] = np.mean(signals['z']) - np.std(signals['z'])

"""### 4. Créez une colonne de signal avec la logique suivante

- Si le score z est supérieur au seuil supérieur, nous aurons -1 ( vendre),
- si le score z est inférieur au seuil inférieur, +1 (Acheter) et la valeur par défaut est nul pour aucun signal.
"""

# signaux - vendre si z-score est plus grand que le niveau a la hausse, ou acheter dans le cas contraire
signals['signals1'] = 0
signals['signals1'] = np.select([signals['z'] > signals['z upper limit'], signals['z'] < signals['z lower limit']], [-1, 1], default=0)

"""### 5.

Prenez la différence de premier ordre de la colonne de signal pour obtenir la position du stock. Si c'est +1 alors nous sommes en Achat, -1 alors vente et 0 si pas de position.

Le deuxième signal sera juste à l'opposé du premier, ce qui signifie que nous sommes Achat sur une action et simultanément courts sur l'autre. De même, prenez la différence de premier ordre pour le deuxième signal et calculez la deuxième colonne de position.
"""

# we take the first order difference to obtain portfolio position in that stock
signals['positions1'] = signals['signals1'].diff()
signals['signals2'] = -signals['signals1']
signals['positions2'] = signals['signals2'].diff()

signals.head()

"""### 6. Visualization des signaux"""

# visualize trading signals and position
fig=plt.figure(figsize=(14,6))
bx = fig.add_subplot(111)
bx2 = bx.twinx()

#plot two different assets
l1, = bx.plot(signals['asset1'], c='#4abdac')
l2, = bx2.plot(signals['asset2'], c='#907163')

u1, = bx.plot(signals['asset1'][signals['positions1'] == 1], lw=0, marker='^', markersize=8, c='g',alpha=0.7)

d1, = bx.plot(signals['asset1'][signals['positions1'] == -1], lw=0,marker='v',markersize=8, c='r',alpha=0.7)

u2, = bx2.plot(signals['asset2'][signals['positions2'] == 1], lw=0,marker=2,markersize=9, c='g',alpha=0.9, markeredgewidth=3)

d2, = bx2.plot(signals['asset2'][signals['positions2'] == -1], lw=0,marker=3,markersize=9, c='r',alpha=0.9,markeredgewidth=3)

bx.set_ylabel(asset1,)
bx2.set_ylabel(asset2, rotation=270)
bx.yaxis.labelpad=15
bx2.yaxis.labelpad=15
bx.set_xlabel('Date')
bx.xaxis.labelpad=15

plt.legend([l1,l2,u1,d1,u2,d2], [asset1, asset2,'Acheter {}'.format(asset1),
           'Vendre {}'.format(asset1),
           'Acheter {}'.format(asset2),
           'Vendre {}'.format(asset2)], loc ='best')

plt.title('Pair Trading - Signaux de Trading et Positions')
plt.xlabel('Date')
plt.grid(True)
plt.tight_layout()

## 2- PARTIE II: BACTESTING


# capital initial de $100 000
initial_capital = 100000

# calculerons le nombre maximum d'actions de chaque action
portefeuille1 = initial_capital// max(signals['asset1'])
portefeuille2 = initial_capital// max(signals['asset2'])
print("asset1", portefeuille1)
print("asset1", portefeuille2)

# pnl pour l'actif 1
portefeuille = pd.DataFrame()
portefeuille['asset1'] = signals['asset1']
portefeuille['holdings1'] = signals['positions1'].cumsum() * signals['asset1'] * portefeuille1
portefeuille['cash1'] = initial_capital - (signals['positions1'] * signals['asset1'] * portefeuille1).cumsum()
portefeuille['total asset1'] = portefeuille['holdings1'] + portefeuille['cash1']
portefeuille['return1'] = portefeuille['total asset1'].pct_change()
portefeuille['positions1'] = signals['positions1']

# pnl pour l'actif 1
portefeuille['asset2'] = signals['asset2']
portefeuille['holdings2'] = signals['positions2'].cumsum() * signals['asset2'] * portefeuille2
portefeuille['cash2'] = initial_capital - (signals['positions2'] * signals['asset2'] * portefeuille2).cumsum()
portefeuille['total asset2'] = portefeuille['holdings2'] + portefeuille['cash2']
portefeuille['return2'] = portefeuille['total asset2'].pct_change()
portefeuille['positions2'] = signals['positions2']

"""- total pnl and z-score"""

# total pnl and z-score
portefeuille['z'] = signals['z']
portefeuille['total asset'] = portefeuille['total asset1'] + portefeuille['total asset2']
portefeuille['z upper limit'] = signals['z upper limit']
portefeuille['z lower limit'] = signals['z lower limit']
portefeuille = portefeuille.dropna()

"""- tracer la variation de la valeur de l'actif du portefeuille et du pnl avec le score z"""

fig = plt.figure(figsize=(14,6),)
ax = fig.add_subplot(111)
ax2 = ax.twinx()
l1, = ax.plot(portefeuille['total asset'], c='g')
l2, = ax2.plot(portefeuille['z'], c='black', alpha=0.3)
b = ax2.fill_between(portefeuille.index,portefeuille['z upper limit'],\
                portefeuille['z lower limit'], \
                alpha=0.2,color='#ffb48f')
ax.set_ylabel('Asset Value')
ax2.set_ylabel('Z Statistics',rotation=270)
ax.yaxis.labelpad=15
ax2.yaxis.labelpad=15
ax.set_xlabel('Date')
ax.xaxis.labelpad=15
plt.title('Performance du portefeuille avec profits et pertes')
plt.legend([l2,b,l1],['Z Statistics',
                      'Z Statistics +-1 Sigma',
                      'Total Portfolio Value'],loc='upper left');



final_portfolio = portefeuille['total asset'].iloc[-1]
delta = (portefeuille.index[-1] - portefeuille.index[0]).days
print('Number of days = ', delta)
YEAR_DAYS = 365
returns = (final_portfolio/initial_capital) ** (YEAR_DAYS/delta) - 1
print('CAGR = {:.3f}%' .format(returns * 100))
