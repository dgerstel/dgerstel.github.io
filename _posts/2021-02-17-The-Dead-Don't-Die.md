---
layout: post 
mathjax: true 
title: The Dead Don't Die 
subtitle: Feature engineer zombies out of your data!
classes: wide 
excerpt: "Here I take a critical look at feature engineering in a heart failure prediction paper. Then I
perform an exploratory analysis. In future posts I'll tackle classification and deploy a web app in Flask."
image: /images/DataCleaningAndExploration_files/zombi.jpg
---

# Introduction

While looking for an interesting dataset to hone my data science skills, I stumbled upon one about heart failure prediction.
It drew my attention as it's more real-life and with impact than, say, predicting who will survive the Titanic disaster
(I really hope no one is planning to rebuild that ship and actually put people on it again ;-).) In the stark contrast,
predicting who will survive after a heart failure incident might be of future use.

The dataset is


-  analysed in this [paper](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5): _Davide Chicco, Giuseppe
Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction
alone. BMC Medical Informatics and Decision Making 20, 16 (2020)_

- available from
[Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

The **motivation** behind this post is that I believe the paper **authors made a mistake in feature engineering** thus
selecting patients that might have died as "survivors", hence the potentially _dead_ patients _don't die_ in the analysis.
# Data Cleaning and Exploration

The original dataset looks as follows.
{% comment %}
```python
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

# Set up figure directory; prepare for a Flask web app
FIG_DIR = "../flask/static"
```

```python
df = pd.read_csv('../data/heart_failure_clinical_records_dataset.csv')
```

```python
df.head()
```
{% endcomment %}

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>anaemia</th>
      <th>creatinine_phosphokinase</th>
      <th>diabetes</th>
      <th>ejection_fraction</th>
      <th>high_blood_pressure</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
      <th>sex</th>
      <th>smoking</th>
      <th>time</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>0</td>
      <td>582</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>265000.00</td>
      <td>1.9</td>
      <td>130</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>0</td>
      <td>7861</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>263358.03</td>
      <td>1.1</td>
      <td>136</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.0</td>
      <td>0</td>
      <td>146</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>162000.00</td>
      <td>1.3</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>1</td>
      <td>111</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>210000.00</td>
      <td>1.9</td>
      <td>137</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.0</td>
      <td>1</td>
      <td>160</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>327000.00</td>
      <td>2.7</td>
      <td>116</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

{% comment %}
```python
df.columns = ' '.join(df.columns).replace('creatinine_ph',
                                          'creatine_ph').split()
df.columns
```

    Index(['age', 'anaemia', 'creatine_phosphokinase', 'diabetes',
           'ejection_fraction', 'high_blood_pressure', 'platelets',
           'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
           'DEATH_EVENT'],
          dtype='object')
{% endcomment %}


Let's have a look at the descriptive statistics of the data.
{% comment %}
```python
df.columns = ' '.join(df.columns).replace('sex', 'Male').split()
df.columns
```

    Index(['age', 'anaemia', 'creatine_phosphokinase', 'diabetes',
           'ejection_fraction', 'high_blood_pressure', 'platelets',
           'serum_creatinine', 'serum_sodium', 'Male', 'smoking', 'time',
           'DEATH_EVENT'],
          dtype='object')
{% endcomment %}

{% comment %}
```python
LABEL = "DEATH_EVENT"  # store target label
POS_LABEL, NEG_LABEL = "Deceased", "Survived"
```

```python
df.shape
```

    (299, 13)

```python
df.describe()
```
{% endcomment %}

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>anaemia</th>
      <th>creatinine_phosphokinase</th>
      <th>diabetes</th>
      <th>ejection_fraction</th>
      <th>high_blood_pressure</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
      <th>sex</th>
      <th>smoking</th>
      <th>time</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.00000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.00000</td>
      <td>299.000000</td>
      <td>299.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>60.833893</td>
      <td>0.431438</td>
      <td>581.839465</td>
      <td>0.418060</td>
      <td>38.083612</td>
      <td>0.351171</td>
      <td>263358.029264</td>
      <td>1.39388</td>
      <td>136.625418</td>
      <td>0.648829</td>
      <td>0.32107</td>
      <td>130.260870</td>
      <td>0.32107</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.894809</td>
      <td>0.496107</td>
      <td>970.287881</td>
      <td>0.494067</td>
      <td>11.834841</td>
      <td>0.478136</td>
      <td>97804.236869</td>
      <td>1.03451</td>
      <td>4.412477</td>
      <td>0.478136</td>
      <td>0.46767</td>
      <td>77.614208</td>
      <td>0.46767</td>
    </tr>
    <tr>
      <th>min</th>
      <td>40.000000</td>
      <td>0.000000</td>
      <td>23.000000</td>
      <td>0.000000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>25100.000000</td>
      <td>0.50000</td>
      <td>113.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>4.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>51.000000</td>
      <td>0.000000</td>
      <td>116.500000</td>
      <td>0.000000</td>
      <td>30.000000</td>
      <td>0.000000</td>
      <td>212500.000000</td>
      <td>0.90000</td>
      <td>134.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>73.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>60.000000</td>
      <td>0.000000</td>
      <td>250.000000</td>
      <td>0.000000</td>
      <td>38.000000</td>
      <td>0.000000</td>
      <td>262000.000000</td>
      <td>1.10000</td>
      <td>137.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>115.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>1.000000</td>
      <td>582.000000</td>
      <td>1.000000</td>
      <td>45.000000</td>
      <td>1.000000</td>
      <td>303500.000000</td>
      <td>1.40000</td>
      <td>140.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>203.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>95.000000</td>
      <td>1.000000</td>
      <td>7861.000000</td>
      <td>1.000000</td>
      <td>80.000000</td>
      <td>1.000000</td>
      <td>850000.000000</td>
      <td>9.40000</td>
      <td>148.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>285.000000</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>


A few observations so far:
- There is a typo with a superfluous *ni* in `creati(ni)ne_phosphokinase` (`serum_creatinine` is
correct).
  
- The column `sex` should be interpreted (as mentioned in the paper) as `Male` for value 1 and `Female` for value 0. Let's
rename it to `Male` for clarity.
  
- For convenience, let's rescale platelets by dividing by 1000.
{% comment %}
```python
df['platelets'] /= 1000
```

```python
df[LABEL].sum() / df.shape[0]
```

    0.3210702341137124
{% endcomment %}

- 32% of patients passed away. The dataset is imbalanced, which we'll need to keep in mind

- The column names should have units and improved formatting. We take the former from the paper. 
  
- The columns are either binary or numerical. There's no categorical columns (`sex' having only two values here may be interpreted as binary)

Now the dataset looks more readible:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age [years]</th>
      <th>Anaemia</th>
      <th>Creatine Phosphokinase [mcg/L]</th>
      <th>Diabetes</th>
      <th>Ejection Fraction [%]</th>
      <th>High Blood Pressure</th>
      <th>Platelets [1000 platelets/mL]</th>
      <th>Serum Creatinine [mg/dL]</th>
      <th>Serum Sodium [mEq/L]</th>
      <th>Male</th>
      <th>Smoking</th>
      <th>Time [days]</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>0</td>
      <td>582</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>265.00000</td>
      <td>1.9</td>
      <td>130</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>0</td>
      <td>7861</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>263.35803</td>
      <td>1.1</td>
      <td>136</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.0</td>
      <td>0</td>
      <td>146</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>162.00000</td>
      <td>1.3</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>1</td>
      <td>111</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>210.00000</td>
      <td>1.9</td>
      <td>137</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.0</td>
      <td>1</td>
      <td>160</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>327.00000</td>
      <td>2.7</td>
      <td>116</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

We shall make different plots for binary and non-binary variables.

{% comment %}
```python
from pprint import PrettyPrinter as PP

units = {
    'age': 'years',
    'creatine_phosphokinase': 'mcg/L',
    'ejection_fraction': '%',
    'platelets': '1000 platelets/mL',  # multipled by 1000 wrt paper
    'serum_creatinine': 'mg/dL',
    'serum_sodium': 'mEq/L',
    'time': 'days'
}


def improve_col_name(name):
    title = name.replace('_', ' ').title()  # Title-like text
    unit = units.get(name, None)  # unit or empty string
    if unit:
        return title + f' [{unit}]'  # eg 'mass [kg]'
    return title


# Improve column names
df.columns = [improve_col_name(c) for c in df.columns if c != LABEL] + [LABEL]
print("New columns:", df.columns)


def is_binary(key):
    return df[key].unique().size == 2


# Store feature metadata (useful for web forms).
# One might add more properties if needed.
feature_metadata = {c: {'is_binary': is_binary(c)} for c in df.columns}

# Print feature metadata
pp = PP(indent=4)
print("\nFeature metadata:")
pp.pprint(feature_metadata)
```

    New columns: Index(['Age [years]', 'Anaemia', 'Creatine Phosphokinase [mcg/L]', 'Diabetes',
           'Ejection Fraction [%]', 'High Blood Pressure',
           'Platelets [1000 platelets/mL]', 'Serum Creatinine [mg/dL]',
           'Serum Sodium [mEq/L]', 'Male', 'Smoking', 'Time [days]',
           'DEATH_EVENT'],
          dtype='object')
    
    Feature metadata:
    {   'Age [years]': {'is_binary': False},
        'Anaemia': {'is_binary': True},
        'Creatine Phosphokinase [mcg/L]': {'is_binary': False},
        'DEATH_EVENT': {'is_binary': True},
        'Diabetes': {'is_binary': True},
        'Ejection Fraction [%]': {'is_binary': False},
        'High Blood Pressure': {'is_binary': True},
        'Male': {'is_binary': True},
        'Platelets [1000 platelets/mL]': {'is_binary': False},
        'Serum Creatinine [mg/dL]': {'is_binary': False},
        'Serum Sodium [mEq/L]': {'is_binary': False},
        'Smoking': {'is_binary': True},
        'Time [days]': {'is_binary': False}}
{% endcomment %}
Let us plot the binary features with the sex of a patient colour-coded.

{% comment %}
```python
def plot_features(binary,
                  kind,
                  ncols=3,
                  figsize=(15, 8),
                  save=True,
                  barhue=None):
    """Plot binary or non-binary features."""
    # Select binary or non-binary columns
    cols = [c for c in df.columns[:-1] if is_binary(c) == binary]
    if kind == 'bar':
        if barhue not in cols:
            raise ValueError("`barhue` needs a column name")
        cols.remove(barhue)  # `barhue` column will be dealt with separately
    nrows = math.ceil(len(cols) / ncols)
    f = plt.figure(figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            ind = i * ncols + j
            # Don't draw empty plots if finished iterating over cols
            if ind > len(cols) - 1:
                break
            ax = f.add_subplot(nrows, ncols, ind + 1)
            cat1 = df[df[LABEL] == 1]
            cat2 = df[df[LABEL] == 0]
            if kind == 'bar':
                sns.barplot(x=cols[ind],
                            y=LABEL,
                            hue=barhue,
                            data=df,
                            ax=ax,
                            capsize=.15)
            elif kind == 'box':
                sns.boxplot(y=cols[ind],
                            x=LABEL,
                            data=df,
                            ax=ax,
                            orient='v',
                            showfliers=False)
            elif kind == 'violin':
                sns.violinplot(y=cols[ind],
                               x=LABEL,
                               data=df,
                               ax=ax,
                               orient='v',
                               showfliers=False)
            elif kind == 'hist':
                cat1[cols[ind]].hist(alpha=0.5,
                                     ax=ax,
                                     color='r',
                                     density=True,
                                     label=POS_LABEL)
                cat2[cols[ind]].hist(alpha=0.5,
                                     ax=ax,
                                     color='b',
                                     density=True,
                                     label=NEG_LABEL)
                ax.set_title(cols[ind])
                ax.legend(loc='best')
            else:
                raise ValueError("Wrong kind of plot requested")
            f.subplots_adjust(hspace=0.5, wspace=0.25, top=1.2)
    if save:
        plt.savefig(os.path.join(FIG_DIR, "features_distr_" + kind + ".png"),
                    bbox_inches='tight')


plot_features(binary=True, kind='bar', barhue='Male', ncols=2)
```
{% endcomment %}

![png](/images/DataCleaningAndExploration_files/DataCleaningAndExploration_20_0.png)

Perhaps surprisingly, the categorical features: anaemia, diabetes, hypertension and smoking, do not seem to exhibit
statistically significant impact on the patient survival prospects. Neither does the sex. Curiously, the big uncertainty
for the smoking women (`sex = 0`) must be reflective of very few patients in this category.

{% comment %}
```python
df.query('Male == 0 & Smoking == 1')[LABEL]
```

    41     1
    54     1
    76     0
    105    1
    Name: DEATH_EVENT, dtype: int64
{% endcomment %}

Indeed there were only 4 such women and 3 have passed away, yielding 75% value of the bar height and big uncertainty.

Now, we'll peek at the non-binary features. We're going to use histograms with either patient category superimposed as
well as box plots, where the horizontal bars mark the ranges and quartiles.

{% comment %}
```python
plot_features(binary=False, kind='hist', ncols=3)
```
{% endcomment %}

![png](/images/DataCleaningAndExploration_files/DataCleaningAndExploration_24_0.png)

{% comment %}
```python
plot_features(binary=False, kind='box', ncols=3)
```
{% endcomment %}

![png](/images/DataCleaningAndExploration_files/DataCleaningAndExploration_25_0.png)

Here are some observations from the above graphs.
- **Age**: patients above the age of 70 are obviously at a higher risk
- **Ejection Fraction** is a strong predictor, especially below around 30 units
- **Serum Creatinine** is also a very strong feature, specifically about around 2 units
- **Serum Sodium** might be a helpful feature, but a little less so than the two previous ones
- **Follow-up Duration (Time)**: We can tell that the patient chance to die in the follow-up period roughly follows an
  exponential distribution, for those that will not survive, as one might expect. The meaning of the follow-up period
  for the survivors is different: it seems it merely reflects how long these patients were monitored. 
  
Let's have a closer look at the follow-up duration. Below we plot it for the deceased patients on top and for the "surviving" ones on the bottom.
  

{% comment %}
```python
def plot_time(n_days=60):
    """Plot distribution of patient follow-up time for surviving
    and deceased patients."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
    KERNEL_WIDTH = 0.2  # KDE smoothing parameter

    # Configure what plots display
    from collections import namedtuple
    config = namedtuple('config',
                        ['text1', 'text2', 'title', 'colour', 'hlight'])
    survived_config =
        config(f'Situation unknown\nafter {n_days} days\n("zombies")',
               f'Survived > {n_days} days',
               r'Follow-up duration of $\bf{survived}$ patients in days',
               'b', (n_days, 350))
    dead_config =
        config(f'Passed away\nwithin {n_days} days',
               f'Survived > {n_days} days',
               r'$\bf{Passed}$ $\bf{away}$ on a given follow up day',
               'r', (0, n_days))
    configs = (survived_config, dead_config)

    max_time = 1.05 * df['Time [days]'].max()
    # Make the plots
    for i, ax in enumerate(axes):
        df[df[LABEL] == i]['Time [days]'].plot.kde(bw_method=KERNEL_WIDTH,
                                                   ax=ax,
                                                   color=configs[i].colour)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        ax.set_title(configs[i].title)
        ax.set_xlabel('Day')

        ax.set_xlim(0, max_time)
        ax.vlines(x=[n_days],
                  ymin=0,
                  ymax=ax.get_ylim()[1],
                  color='g',
                  linestyles='dashed')

        ax.annotate(text=configs[i].text1,
                    xy=(5, 0.45 * ax.get_ylim()[1]),
                    xytext=(5, 0.45 * ax.get_ylim()[1]),
                    ha='left',
                    va='bottom')
        ax.annotate(text='',
                    xy=(0, 0.4 * ax.get_ylim()[1]),
                    xytext=(n_days, 0.4 * ax.get_ylim()[1]),
                    arrowprops=dict(arrowstyle='<->', color=configs[i].colour))

        ax.annotate(text=configs[i].text2,
                    xy=(n_days, 0.85 * ax.get_ylim()[1]),
                    xytext=(150, 0.85 * ax.get_ylim()[1]),
                    va='bottom')
        ax.annotate(text='',
                    xy=(n_days, 0.8 * ax.get_ylim()[1]),
                    xytext=(max_time, 0.8 * ax.get_ylim()[1]),
                    arrowprops=dict(arrowstyle='<->', color=configs[i].colour))

        ax.axvspan(*configs[i].hlight, facecolor='orange', alpha=0.5)


plot_time()
```
{% endcomment %}
![png](/images/DataCleaningAndExploration_files/DataCleaningAndExploration_27_0.png)

Indeed, the 'Time' feature has different meaning for either category of patients:

- for **survived** patients: duration of the follow-up time
- for **deceased** patients: day of their passing

Therefore, the former ones might have passed away anytime after the follow-up termination. For instance, they might have
passed away anytime between their follow-up duration and the full 350-day period. Hence, the patients that are
potentially **dead, don't die** in the data; they will be hereafter referred to as 'zombies' ;-).

As a break while reading this post I recommend the wonderful Sturgill
Simpson's ["The Dead Don't Die" song](https://www.youtube.com/watch?v=xiukuoSjDj0) ;-)

Selecting the 'survivors' of follow-up duration greater than $$m$$ days ($$Time > m$$ days) means they have *certainly*
survived the first $$m$$ days. On the other hand, to be consistent, we must choose the 'non-survivors' who have passed
away *within* the first $$m$$ days ($$Time < m$$ days). The corresponding regions are highlighted in the graphs.

This way we will be able to analyse how likely a given patient is to survive $$m$$ days, e.g. 2 months, after the heart
failure.

The patients whose follow-up duration is smaller than $$m$$ need to be rejected, reducing our statistics.

Curiously, for the **survivors** (within $$m$$ days) target label, we might also include the ones who passed away after
$$m$$ days. However, it seems preferable not to do this: I would rather tell a patient they are going to survive at
least $$m$$ days if they *might survive even longer* than that rather than if they are *bound to* pass away anywhere
from the day $$m+1$$ onwards. Both approaches are fine, though, as long as we remain clear.

At this point it is astonishing that the paper authors **ignored** the $$Time$$ variable in one of their studies, trying
to justify it as follows:
> In the previous part of the analysis, we excluded follow-up time from the dataset because we preferred to focus on the clinical features and to try to discover something meaningful about them.

This approach is evidently **biased** as they might have had contaminated train and test samples: people who *might*
have died were considered *survivors*.

Let us see how many patients are we left with for various values of $$m$$.

{% comment %}
```python
def plot_survived_died_period(n_min=10, n_max=350, step=10):
    days = []
    survived_arr = []
    deceased_arr = []
    f1 = []
    sum_all = []
    for n in range(n_min, n_max, step):
        days.append(n)
        survived = ((df[LABEL] == 0) & (df['Time [days]'] > n)).sum()
        survived_fr = survived / (df[LABEL] == 0).sum()
        died = ((df[LABEL] == 1) & (df['Time [days]'] < n)).sum()
        died_fr = died / (df[LABEL] == 1).sum()
        survived_arr.append(survived)
        deceased_arr.append(died)
        f1.append(2 * (survived * died) / (survived + died))
        sum_all.append(survived + died)
    max_sum_arg = np.argmax(sum_all)
    max_f1_arg = np.argmax(f1)
    print("Max sum arg:", days[max_sum_arg], "days")
    print("Max F1 arg:", days[max_f1_arg], "days")
    plt.plot(days, survived_arr, label='survived', color='b')
    plt.plot(days, deceased_arr, label='deceased', color='r')
    plt.plot(days, f1, label='harmonic average', color='g', linestyle='dotted')
    plt.plot(days,
             sum_all,
             label='survived+deceased',
             color='gray',
             linestyle='dashed')
    plt.vlines(x=days[max_sum_arg],
               ymin=0,
               ymax=sum_all[max_sum_arg],
               linestyles='dashed',
               colors='gray')
    plt.vlines(x=days[max_f1_arg],
               ymin=0,
               ymax=f1[max_f1_arg],
               linestyles='dotted',
               colors='gray')
    plt.xlabel("Day")
    plt.ylabel("No. of patients")
    plt.title(
        "Number of surviving (deceased) patients at least (within)"
        + " given number of days after heart failure"
    )
    plt.legend(loc='best')
    plt.show()


plot_survived_died_period()
```

    Max sum arg: 70 days
    Max F1 arg: 100 days
{% endcomment %}
![png](/images/DataCleaningAndExploration_files/DataCleaningAndExploration_30_1.png)

Maximising harmonic average ensures the two variables are high and close by (i.e. balanced). The harmonic average
approximately has a plateau in the region around 75-175 days. We might try to run the ML algorithms in that range.

The maximum harmonic average in on the 100-th day and the maximum sum of survived and deceased patients is achieved if
we split at the 70-th day. If we want to have both, high statistics and good balance between the categories, we should
consider splitting around the 75-100 days, yielding around 210-250 patients. Of course, all other values are technically
possible, but they should yield less precise ML performance (which claim we might test).

We'll select, therefore, the highlighted below regions of either patient group.

{% comment %}
```python
plot_time(100)
```
{% endcomment %}

![png](/images/DataCleaningAndExploration_files/DataCleaningAndExploration_32_0.png)


{% comment %}
```python
import joblib

joblib.dump(df, "df.pkl")
joblib.dump(feature_metadata, "feature_metadata.pkl")
```

    ['feature_metadata.pkl']

As a sanity check, let's see if we can properly load the saved objects.

```python
df = joblib.load("df.pkl")
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age [years]</th>
      <th>Anaemia</th>
      <th>Creatine Phosphokinase [mcg/L]</th>
      <th>Diabetes</th>
      <th>Ejection Fraction [%]</th>
      <th>High Blood Pressure</th>
      <th>Platelets [1000 platelets/mL]</th>
      <th>Serum Creatinine [mg/dL]</th>
      <th>Serum Sodium [mEq/L]</th>
      <th>Male</th>
      <th>Smoking</th>
      <th>Time [days]</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>0</td>
      <td>582</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>265.00000</td>
      <td>1.9</td>
      <td>130</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>0</td>
      <td>7861</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>263.35803</td>
      <td>1.1</td>
      <td>136</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.0</td>
      <td>0</td>
      <td>146</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>162.00000</td>
      <td>1.3</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>1</td>
      <td>111</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>210.00000</td>
      <td>1.9</td>
      <td>137</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.0</td>
      <td>1</td>
      <td>160</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>327.00000</td>
      <td>2.7</td>
      <td>116</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
fm = joblib.load("feature_metadata.pkl")
fm
```

    {'Age [years]': {'is_binary': False},
     'Anaemia': {'is_binary': True},
     'Creatine Phosphokinase [mcg/L]': {'is_binary': False},
     'Diabetes': {'is_binary': True},
     'Ejection Fraction [%]': {'is_binary': False},
     'High Blood Pressure': {'is_binary': True},
     'Platelets [1000 platelets/mL]': {'is_binary': False},
     'Serum Creatinine [mg/dL]': {'is_binary': False},
     'Serum Sodium [mEq/L]': {'is_binary': False},
     'Male': {'is_binary': True},
     'Smoking': {'is_binary': True},
     'Time [days]': {'is_binary': False},
     'DEATH_EVENT': {'is_binary': True}}
{% endcomment %}

# Conclusions -- what I've learnt so far

- Feeding features directly to an ML algorithm is a *wrong* practise
- Meaning of target labels might be more subtle than it seems (here 'DEATH_EVENT' relates to another feature, follow-up
  time);
- Here, we must *extract* the target labels using another feature **before** making predictions
- Ignoring some features we do not understand is dangerous, especially if they somehow relate to the labels
- Be patient and use a "pause and think" (or "pause and plot") approach before any classification / regression
- Regarding data collection, the follow-up period should have been the same for each patient (if possible), e.g. 1 year.
  It is understandable, though, that it might have not been possible.

**In the next post** we'll implement a couple of classifiers to predict patient survival within 100 days after a heart
failure.
