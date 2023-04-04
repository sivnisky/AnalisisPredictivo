# Trabajo Práctico N° 1 - Análisis Predictivo - Sofia Ivnisky - 1° cuatrimestre 2023


# Importar librerías
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import savgol_filter

# GPD x País para los gráficos con mapas
def load_gpd_world():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Códigos de país faltantes
    world.loc[world['name'] == 'France', 'iso_a3'] = 'FRA'
    world.loc[world['name'] == 'Norway', 'iso_a3'] = 'NOR'
    world.loc[world['name'] == 'Somaliland', 'iso_a3'] = 'SOM'
    world.loc[world['name'] == 'Kosovo', 'iso_a3'] = 'RKS'
    # Para que no haya 2 valores = Somalía
    world = world.dissolve(by='iso_a3').reset_index()
    return world

# Leer la base de datos
data = pd.read_csv('/Users/sofiaivnisky/Downloads/Mental health Depression disorder Data.csv')
data = data.drop('index', axis=1)

# EDA
data.head()
data.info()
data.describe()

# Separar las primeras 3 tablas
# Tabla 1
mental_health_disorder_share = data.iloc[:6468].copy()
print("nombres: ", mental_health_disorder_share.columns)
mental_health_disorder_share = mental_health_disorder_share.rename(columns = {'Entity': 'country', 'Code': 'country_code', 'Year': 'year','Schizophrenia (%)': 'Esquizofrenia','Bipolar disorder (%)': 'Trastorno Bipolar','Eating disorders (%)': 'Trastornos Alimenticios', 'Anxiety disorders (%)': 'Ansiedad','Drug use disorders (%)': 'Adicción a las drogas', 'Depression (%)': 'Depresion','Alcohol use disorders (%)': 'Alcoholismo'})
mental_health_disorder_share.iloc[:, 2:] = mental_health_disorder_share.iloc[:, 2:].apply(pd.to_numeric)


# Tabla 2
disorder_per_sex = data.iloc[6469:54276, :6].copy()
disorder_per_sex.columns = ['country', 'country_code', 'year','males_share', 'females_share','population']
disorder_per_sex.iloc[:, 3:] = disorder_per_sex.iloc[:, 3:].apply(pd.to_numeric)

# Tabla 3
suicidios = data.iloc[54277:102084]
suicidios.columns = ['index', 'country', 'country_code', 'year', 'suicide_rate', 'depression_rate', 'population', 'a', 'b', 'c']
suicidios = suicidios.drop(axis=1, columns=['index','a','b','c'])

# Ver qué países tiene código null en la tabla 1
print("Tabla 1: ",mental_health_disorder_share['country'].loc[mental_health_disorder_share['country_code'].isnull()].unique())

# Ver qué países tiene código null en la tabla 2
print("Tabla 2: ",disorder_per_sex['country'].loc[disorder_per_sex['country_code'].isnull()].unique())

# Ver qué países tiene código null en la tabla 3
print("Tabla 3: ",suicidios['country'].loc[suicidios['country_code'].isnull()].unique())

# Eliminar valores null
mental_health_disorder_share.dropna(axis=0, inplace = True)
disorder_per_sex.dropna(axis=0, inplace = True)
suicidios.dropna(axis=0, inplace = True)

# Ver tipos de datos para cada tabla
print("Tabla 1: ",mental_health_disorder_share.dtypes)
print("Tabla 2: ",disorder_per_sex.dtypes)
print("Tabla 3: ",suicidios.dtypes)

# Outliers
cols = ['Esquizofrenia', 'Trastorno Bipolar', 'Trastornos Alimenticios', 'Ansiedad', 'Adicción a las drogas', 'Depresion', 'Alcoholismo']
data = mental_health_disorder_share[cols]
fig, axs = plt.subplots(nrows=1, ncols=len(cols), figsize=(20, 5))
colors = ["#FF5733", "#FBB12B", "#93DD10", "#34E0EA", "#7489EF", "#FA77B3", "#11A4D1" ]
fig.patch.set_facecolor('#F2F2F2')
for i, col in enumerate(cols):
    bp = axs[i].boxplot(data[col], patch_artist=True)
    bp['boxes'][0].set_facecolor(colors[i])
    axs[i].set_title(col)
    axs[i].set_ylabel('Porcentaje')
fig.suptitle('Distribución de trastornos mentales', fontsize=16)
plt.tight_layout()
plt.show()


# Tabla de pobalción global x año
# Toma 'country_code', 'year' y 'population' de disorder_per_sex
population_per_year = (disorder_per_sex.dropna().loc[disorder_per_sex.country_code != 'OWID_WRL',['country_code', 'year', 'population']].copy())
# Convierte en valores numéricos a 'year' y 'population'
population_per_year['year'] = pd.to_numeric(population_per_year['year'])
population_per_year['population'] = pd.to_numeric(population_per_year['population'])
# Utiliza la función de Geopandas anterior
world = load_gpd_world()
world = world.rename(columns={'iso_a3':'country_code'})
# Modifica el data frame mental_health_disorder_share uniendo 'country_code', 'continent' y 'geometry' de word y 'share' de mental_health_disorder_share.
mental_health_disorder_share = world[['country_code', 'continent', 'geometry']].merge(mental_health_disorder_share)

# Se visualiza la información
mental_health_disorder_share.info()
disorder_per_sex.info()

# ¿Cómo se distribuyen los datos? ¿El avance en los años es lineal?
# Media
mean = mental_health_disorder_share[mental_health_disorder_share.columns[5:12]].mean(axis=0)
print("Medias: ", mean)
# Desviación estándar
std = mental_health_disorder_share[mental_health_disorder_share.columns[5:12]].std(axis=0)
print("Desviaciones estándar: ", std)

# Análisis gráfico
# Gráfico de líneas
# Normalizar
norm = (mental_health_disorder_share[mental_health_disorder_share.columns[5:12]] - mean)/std
norm = pd.concat([mental_health_disorder_share['year'],norm], axis = 1)
norm.groupby('year').mean().plot(figsize=(13,6), title='Trastornos por año (promedio de todos los países)', ylabel = 'Valores normalizados', xlabel = 'Años');
plt.show()

# Porcentaje de personas con cada trastorno
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.set_facecolor('#f2f2f2')
# Depresión
avgdep = mental_health_disorder_share.groupby('year').mean(numeric_only=True)
axs[0, 0].plot(avgdep['Depresion'], color = "#FF5733", linewidth = 2)
axs[0, 0].set(title='Pacientes con depresión', xlabel='Años', ylabel='Depresión %')

# Trastorno Bipolar
avgbip = mental_health_disorder_share.groupby('year').mean(numeric_only=True)
axs[0, 1].plot(avgbip['Trastorno Bipolar'], color = "#FBB12B", linewidth = 2)
axs[0, 1].set(title='Pacientes con Trastorno Bipolar', xlabel='Años', ylabel='Trastorno Bipolar %')

# Problemas de drogas
avgdro = mental_health_disorder_share.groupby('year').mean(numeric_only=True)
axs[0, 2].plot(avgdro['Adicción a las drogas'], color = "#93DD10", linewidth = 2)
axs[0, 2].set(title='Pacientes con adicción a las drogas', xlabel='Años', ylabel='Adicción a las drogas %')

# Alcoholismo
avgalc = mental_health_disorder_share.groupby('year').mean(numeric_only=True)
axs[0, 3].plot(avgalc['Alcoholismo'], color = "#34E0EA", linewidth = 2)
axs[0, 3].set(title='Pacientes con Alcoholismo', xlabel='Años', ylabel='Alcoholismo %')

# Esquizofrenia
avgesq = mental_health_disorder_share.groupby('year').mean(numeric_only=True)
axs[1, 0].plot(avgesq['Esquizofrenia'], color = "#7489EF", linewidth = 2)
axs[1, 0].set(title='Pacientes con Esquizofrenia', xlabel='Años', ylabel='Esquizofrenia %')

# Ansiedad
avgans = mental_health_disorder_share.groupby('year').mean(numeric_only=True)
axs[1, 1].plot(avgans['Ansiedad'], color = "#FA77B3", linewidth = 2)
axs[1, 1].set(title='Pacientes con Ansiedad', xlabel='Años', ylabel='Ansiedad %')

# Problemas alimenticios
avgali = mental_health_disorder_share.groupby('year').mean(numeric_only=True)
axs[1, 2].plot(avgali['Trastornos Alimenticios'], color = "#924DE4", linewidth = 2)
axs[1, 2].set(title='Pacientes Trastornos Alimenticios', xlabel='Años', ylabel='Trastornos Alimenticios %')

plt.tight_layout()
plt.show()

# Por continente
plot_df = (mental_health_disorder_share
               .loc[mental_health_disorder_share.year == 2017]
               .groupby('continent')
               .mean()
               .reset_index()
               .drop(columns=['year'])
               .melt(id_vars='continent'))

fig = px.bar(plot_df, x="variable", y="value",
             color="continent", barmode="group",
             title="Desórdenes de salud mental por continente (2017)",
             height = 700,
             width = 1400,
             color_discrete_sequence=["#FF5733", "#FBB12B", "#93DD10", "#34E0EA", "#7489EF", "#FA77B3"])

fig.update_layout(
    yaxis_title="Porcentaje de participación",
    xaxis_title="Desorden mental",
    font_family="Tahoma",
    font_size=18,
    legend_title=""
    )
fig.show()

# Caso particular: Argentina
# Gráfico de líneas
mental_health_disorder_share_country = mental_health_disorder_share.loc[mental_health_disorder_share['country'] == 'Argentina']
mental_health_disorder_share_country_year = mental_health_disorder_share_country.groupby('year').mean()
fig, ax = plt.subplots()
ax.plot(mental_health_disorder_share_country_year.index, mental_health_disorder_share_country_year.values, linewidth=2)
mental_health_disorder_share_country_year.plot(title='Desórdenes de salud mental en Argentina',
                                               xlabel='Año',
                                               ylabel='Porcentaje de participación')
ax.set_facecolor('#F5F5F5')
plt.show()

# Gráfico de barras
mental_health_disorder_share_country = mental_health_disorder_share.loc[(mental_health_disorder_share['country'] == 'Argentina')]
mental_health_disorder_share_country = mental_health_disorder_share_country[['Esquizofrenia','Trastorno Bipolar','Trastornos Alimenticios','Ansiedad','Adicción a las drogas','Depresion','Alcoholismo']]
mental_health_disorder_share_country = mental_health_disorder_share_country.melt(var_name='variable', value_name='value')
fig = px.bar(mental_health_disorder_share_country, x="variable", y="value",
             title="Desórdenes de salud mental en Argentina",
             height=700, width=1400,
             color = "variable",
             color_discrete_sequence= ["#FF5733", "#FBB12B", "#93DD10", "#34E0EA", "#7489EF", "#FA77B3"],
             barmode = "stack")
fig.update_layout(
    yaxis_title="Porcentaje de participación",
    xaxis_title="Desorden mental",
    font_family="Tahoma",
    font_size=18,
    legend_title="",)
fig.show()

# Correlación
# Pearson
sns.set_theme(style="white")
disorder_correlation = mental_health_disorder_share.loc[mental_health_disorder_share.year == 2017, ['Esquizofrenia','Trastorno Bipolar','Trastornos Alimenticios','Ansiedad','Adicción a las drogas','Depresion','Alcoholismo']].corr()
print("Correlación: ",round(disorder_correlation, 2))
# Mapa de calor
mask = np.triu(np.ones_like(disorder_correlation, dtype=bool))
highest_correlation = disorder_correlation[disorder_correlation != 1].max().max()
f, ax = plt.subplots(figsize=(10, 8))
heatmap = sns.heatmap(disorder_correlation, mask=mask, cmap="RdPu", vmax=highest_correlation, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True, fmt=".2f")
plt.show()

# Distribución global del alcoholismo
plot_df = mental_health_disorder_share.loc[mental_health_disorder_share.year == 2017]
cmap='RdPu'
fig, ax = plt.subplots(figsize=(10, 5))
measure_column = 'Alcoholismo'
plot_df.plot(edgecolor='gray',
                            linewidth=1,
                            column=measure_column,
                            cmap=cmap,
                            ax=ax)

sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=plot_df[measure_column].min(), vmax=plot_df[measure_column].max()), cmap=cmap)
cbaxes = fig.add_axes([0.1, 0.25, 0.01, 0.5])
cbar = fig.colorbar(sm, cax=cbaxes)
fig.suptitle("Porcentaje de la población con problemas de alcoholismo", fontsize=20)
ax.axis('off')
plt.show()

# Distribución global de la adicción a las drogas
plot_df = mental_health_disorder_share.loc[mental_health_disorder_share.year == 2017]
cmap='Blues'
fig, ax = plt.subplots(figsize=(10, 5))
measure_column = 'Adicción a las drogas'
plot_df.plot(edgecolor='gray',
                            linewidth=1,
                            column=measure_column,
                            cmap=cmap,
                            ax=ax)

sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=plot_df[measure_column].min(), vmax=plot_df[measure_column].max()), cmap=cmap)
cbaxes = fig.add_axes([0.1, 0.25, 0.01, 0.5])
cbar = fig.colorbar(sm, cax=cbaxes)
fig.suptitle("Porcentaje de la población con problemas de drogas", fontsize=20)
ax.axis('off')
plt.show()

# Distribución global de la depresión
plot_df = mental_health_disorder_share.loc[mental_health_disorder_share.year == 2017]
cmap='Reds'
fig, ax = plt.subplots(figsize=(10, 5))
measure_column = 'Depresion'
plot_df.plot(edgecolor='gray',
                            linewidth=1,
                            column=measure_column,
                            cmap=cmap,
                            ax=ax)

sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=plot_df[measure_column].min(), vmax=plot_df[measure_column].max()), cmap=cmap)
cbaxes = fig.add_axes([0.1, 0.25, 0.01, 0.5])
cbar = fig.colorbar(sm, cax=cbaxes)
fig.suptitle("Porcentaje de la población con depresión", fontsize=20)
ax.axis('off')
plt.show()

# Distribución global del trastorno bipolar
plot_df = mental_health_disorder_share.loc[mental_health_disorder_share.year == 2017]
cmap='Greens'
fig, ax = plt.subplots(figsize=(10, 5))
measure_column = 'Trastorno Bipolar'
plot_df.plot(edgecolor='gray',
                            linewidth=1,
                            column=measure_column,
                            cmap=cmap,
                            ax=ax)

sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=plot_df[measure_column].min(), vmax=plot_df[measure_column].max()), cmap=cmap)
cbaxes = fig.add_axes([0.1, 0.25, 0.01, 0.5])
cbar = fig.colorbar(sm, cax=cbaxes)
fig.suptitle("Porcentaje de la población con trastorno bipolar", fontsize=20)
ax.axis('off')
plt.show()

# Distribución global de la ansiedad
plot_df = mental_health_disorder_share.loc[mental_health_disorder_share.year == 2017]
cmap='Purples'
fig, ax = plt.subplots(figsize=(10, 5))
measure_column = 'Ansiedad'
plot_df.plot(edgecolor='gray',
                            linewidth=1,
                            column=measure_column,
                            cmap=cmap,
                            ax=ax)

sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=plot_df[measure_column].min(), vmax=plot_df[measure_column].max()), cmap=cmap)
cbaxes = fig.add_axes([0.1, 0.25, 0.01, 0.5])
cbar = fig.colorbar(sm, cax=cbaxes)
fig.suptitle("Porcentaje de la población con ansiedad", fontsize=20)
ax.axis('off')
plt.show()

# Distribución global de los trastornos alimenticios
plot_df = mental_health_disorder_share.loc[mental_health_disorder_share.year == 2017]
cmap='Blues'
fig, ax = plt.subplots(figsize=(10, 5))
measure_column = 'Trastornos Alimenticios'
plot_df.plot(edgecolor='gray',
                            linewidth=1,
                            column=measure_column,
                            cmap=cmap,
                            ax=ax)

sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=plot_df[measure_column].min(), vmax=plot_df[measure_column].max()), cmap=cmap)
cbaxes = fig.add_axes([0.1, 0.25, 0.01, 0.5])
cbar = fig.colorbar(sm, cax=cbaxes)
fig.suptitle("Porcentaje de la población con trastornos alimenticios", fontsize=20)
ax.axis('off')
plt.show()

# Distribución global de la esquizofrenia
plot_df = mental_health_disorder_share.loc[mental_health_disorder_share.year == 2017]
cmap='RdPu'
fig, ax = plt.subplots(figsize=(10, 5))
measure_column = 'Esquizofrenia'
plot_df.plot(edgecolor='gray',
                            linewidth=1,
                            column=measure_column,
                            cmap=cmap,
                            ax=ax)

sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=plot_df[measure_column].min(), vmax=plot_df[measure_column].max()), cmap=cmap)
cbaxes = fig.add_axes([0.1, 0.25, 0.01, 0.5])
cbar = fig.colorbar(sm, cax=cbaxes)
fig.suptitle("Porcentaje de la población con esquizofrenia", fontsize=20)
ax.axis('off')
plt.show()

# Tendencias globales en trastornos de salud mental
# Calcular desórdenes por año
disorders_yearly = mental_health_disorder_share.merge(population_per_year, on =['country_code','year'])
disorders_yearly.iloc[:, 5:-1] = disorders_yearly.iloc[:, 5:-1].apply(lambda x: (x/100) * disorders_yearly.iloc[:,-1])
disorders_yearly = (disorders_yearly
     .groupby('year')
     .sum()
     .reset_index()
)
disorders_yearly.iloc[:, 1:-1] = disorders_yearly.iloc[:, 1:-1].apply(lambda x : (x / disorders_yearly.iloc[:,-1]) * 100)
disorders_yearly = (disorders_yearly
                        .drop(columns='population')
                        .melt(id_vars='year')
                        .rename(columns={'variable':'disease', 'value':'share'})
                )

fig = px.line(disorders_yearly, x='year', y='share', color='disease', markers=True,
             title="Tendencias globales en trastornos de salud mental",
             height=400)

fig.update_layout(
    yaxis_title="Porcentaje %",
    xaxis_title="",
    font_family="Tahoma",
    font_size=18,
    legend_title="",
)
fig.show()

# Cambios regionales
cambios_regionales = mental_health_disorder_share.copy()
cambios_regionales['total_disorders_share'] = cambios_regionales.iloc[:, 5:].sum(axis=1)
cambios_regionales = cambios_regionales[['country', 'country_code', 'year','total_disorders_share']]
cambios_regionales = cambios_regionales.sort_values(['country','year'])
cambios_regionales = cambios_regionales.groupby(['country', 'country_code']).agg({'total_disorders_share':['first','last']}).reset_index()
cambios_regionales.columns =['country','country_code','first','last']
cambios_regionales['change'] = cambios_regionales.iloc[:, -1] - cambios_regionales.iloc[:, -2]
cambios_regionales = world[['country_code', 'continent', 'geometry']].merge(cambios_regionales)
cmap = 'Greens'
fig, ax = plt.subplots(figsize=(10, 5))
cambios_regionales.plot(edgecolor='gray',
                               linewidth=1,
                               column='change',
                               cmap=cmap,
                               ax=ax)
sm = plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, vmin=cambios_regionales['change'].min(),
                                             vmax=cambios_regionales['change'].max()), cmap=cmap)
cbaxes = fig.add_axes([0.1, 0.25, 0.01, 0.5])
cbar = fig.colorbar(sm, cax=cbaxes)
fig.suptitle("Diferencias entre los % de trastornos mentales (1990 vs. 2017)", fontsize=20)
ax.axis('off')
plt.show()

# Análisis de depresión
mean_x_pais = mental_health_disorder_share.groupby('country')[['Esquizofrenia','Trastorno Bipolar','Trastornos Alimenticios','Ansiedad','Adicción a las drogas','Depresion','Alcoholismo']
].mean().reset_index()
mean_depresion_pais = mean_x_pais['Depresion'].mean()

plt.figure(figsize = (15,10))
sns.barplot(data = mean_x_pais,
            x = 'Depresion',
            y = 'country',
            order = mean_x_pais.sort_values('Depresion', ascending = False).country.head(20),
            palette = "rocket")
plt.axvline(mean_depresion_pais, color = 'pink', linestyle ='--')
plt.title('Top países con mas % de depresión',
          fontsize = 18,
          loc = 'left',
          fontweight='bold')

plt.ylabel('',)
plt.xlabel('');
plt.show()

# Análisis de trastornos alimenticios
mean_x_pais = mental_health_disorder_share.groupby('country')[['Esquizofrenia','Trastorno Bipolar','Trastornos Alimenticios','Ansiedad','Adicción a las drogas','Depresion','Alcoholismo']
].mean().reset_index()
mean_alim_pais = mean_x_pais['Trastornos Alimenticios'].mean()

plt.figure(figsize = (15,10))
sns.barplot(data = mean_x_pais,
            x = 'Trastornos Alimenticios',
            y = 'country',
            order = mean_x_pais.sort_values('Trastornos Alimenticios', ascending = False).country.head(20),
            palette = "Blues_r")
plt.axvline(mean_alim_pais, color = 'pink', linestyle ='--')
plt.title('Top países con mas % de Trastornos Alimenticios',
          fontsize = 18,
          loc = 'left',
          fontweight='bold')

plt.ylabel('',)
plt.xlabel('');
plt.show()

# Análisis de Esquizofrenia
mean_x_pais = mental_health_disorder_share.groupby('country')[['Esquizofrenia','Trastorno Bipolar','Trastornos Alimenticios','Ansiedad','Adicción a las drogas','Depresion','Alcoholismo']
].mean().reset_index()
mean_esq_pais = mean_x_pais['Esquizofrenia'].mean()

plt.figure(figsize = (15,10))
sns.barplot(data = mean_x_pais,
            x = 'Esquizofrenia',
            y = 'country',
            order = mean_x_pais.sort_values('Esquizofrenia', ascending = False).country.head(20),
            palette = "Purples_r")
plt.axvline(mean_esq_pais, color = 'pink', linestyle ='--')
plt.title('Top países con mas % de Esquizofrenia',
          fontsize = 18,
          loc = 'left',
          fontweight='bold')

plt.ylabel('',)
plt.xlabel('');
plt.show()

# Análisis de trastorno bipolar
mean_x_pais = mental_health_disorder_share.groupby('country')[['Esquizofrenia','Trastorno Bipolar','Trastornos Alimenticios','Ansiedad','Adicción a las drogas','Depresion','Alcoholismo']
].mean().reset_index()
mean_bipolar_pais = mean_x_pais['Trastorno Bipolar'].mean()

plt.figure(figsize = (15,10))
sns.barplot(data = mean_x_pais,
            x = 'Trastorno Bipolar',
            y = 'country',
            order = mean_x_pais.sort_values('Trastorno Bipolar', ascending = False).country.head(20),
            palette = "Reds_r")
plt.axvline(mean_bipolar_pais, color = 'pink', linestyle ='--')
plt.title('Top países con mas % de Trastorno Bipolar',
          fontsize = 18,
          loc = 'left',
          fontweight='bold')

plt.ylabel('',)
plt.xlabel('');
plt.show()

# Análisis de ansiedad
mean_x_pais = mental_health_disorder_share.groupby('country')[['Esquizofrenia','Trastorno Bipolar','Trastornos Alimenticios','Ansiedad','Adicción a las drogas','Depresion','Alcoholismo']
].mean().reset_index()
mean_ansiedad_pais = mean_x_pais['Ansiedad'].mean()

plt.figure(figsize = (15,10))
sns.barplot(data = mean_x_pais,
            x = 'Ansiedad',
            y = 'country',
            order = mean_x_pais.sort_values('Ansiedad', ascending = False).country.head(20),
            palette = "Oranges_r")
plt.axvline(mean_ansiedad_pais, color = 'pink', linestyle ='--')
plt.title('Top países con mas % de Ansiedad',
          fontsize = 18,
          loc = 'left',
          fontweight='bold')

plt.ylabel('',)
plt.xlabel('');
plt.show()

# Diferencias entre hombres y mujeres
# Top 10 países con mas diferencia
top_10 = disorder_per_sex.groupby('country').mean()[['males_share','females_share']].sort_values('males_share', ascending = False)[:10]
color = ["#FFA132", "#FF32A8"]
top_10.plot.bar(ylabel= ('%'), figsize = (8,5), color = color);
plt.show()

# Diferencia a lo largo del tiempo
gap = disorder_per_sex
gap['gap'] = np.abs(disorder_per_sex['females_share'] - disorder_per_sex['males_share'])
gap.groupby('year').mean()[['gap']].plot(title = 'Diferencia mujeres/hombres en el tiempo', xlabel = 'Años', ylabel = 'Diferencia %', color = "#FE2380");
plt.show()