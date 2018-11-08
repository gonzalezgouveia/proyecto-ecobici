
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid") # estilo de salida de las gráficas


# In[3]:


# Leyendo los datos
estaciones = pd.read_csv('ecobiciestaciones.csv')
viajes = pd.read_csv('2018-10.csv')


# In[6]:


n = viajes.shape[0] # shape[0] cantidad de observaciones. shape[1] cantidad de variables
print('total de viajes octubre 2018: ',n)


# In[7]:


from datetime import datetime


# In[15]:


# concatenar Hora_Retiro y Fecha_Retiro
viajes['fecha_hora_retiro'] = viajes.Fecha_Retiro+' '+viajes.Hora_Retiro

# convertir fecha y hora a datetime
viajes.fecha_hora_retiro.head()

def convierte_fecha(str_fecha_hora):
    return datetime.strptime(str_fecha_hora, '%d/%m/%Y %H:%M:%S')

viajes['fecha_hora'] = viajes.fecha_hora_retiro.map(convierte_fecha)

# reindexing the dataframe
viajes.index = viajes.fecha_hora

# limpiando valores de otros años
viajes = viajes.loc['2018-10']


# In[16]:


viajes


# In[18]:


viajes_resample_day = viajes.Bici.resample('H').count()
print(viajes_resample_day.head())
viajes_resample_day.plot()
plt.show()


# In[29]:


df_resample = pd.concat([viajes_resample_day], axis=1)
df_resample['dayofweek'] = df_resample.index.dayofweek # 0 is monday
df_resample.head()


# In[34]:


df_mon_to_fri = df_resample[df_resample.dayofweek.isin([0,1,2,3,4])].Bici
df_mon_to_fri


# # Modelacion de serie temporal

# In[35]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[39]:


df_mon_to_fri.to_csv('df_mon_to_fri.csv')


# In[188]:


df_mon_to_fri.plot()
plt.savefig('mon_to_fri.png')
plt.show()


# In[191]:


df_mon_to_fri.loc['2018-10-01':'2018-10-05'].plot()
plt.savefig('semana_oct.png')
plt.show()


# In[220]:


# x = df_mon_to_fri.reset_index().Bici
x = df_mon_to_fri
sarima_model = SARIMAX(x, order=(2,0,1), seasonal_order=(2,1,0,24))
results = sarima_model.fit()
print(results.summary())


# In[221]:


results.plot_diagnostics()
plt.show()


# In[116]:


# para sacar predicciones y bandas de confianza del metodo get_forecast() de results
pred_jue_vie = results.get_forecast(steps=24*2)
pred_jue_vie.predicted_mean
pred_jue_vie.conf_int().plot()
plt.show()


# # hacer prediccion para semana 29-oct al 2-nov

# In[222]:


# tomar de datos originales dias 29-oct, 30-oct, y 31-oct
df_29_31 = df_mon_to_fri.loc['2018-10-29':'2018-10-31']
df_29_31.plot()

# agregar bandas de confianza
pred_1_2_conf = results.get_forecast(steps=24*2).conf_int()
pred_1_2_conf.index = pd.date_range(start='11/1/2018', end='11/3/2018', freq='H')[:-1]
pred_1_2_conf.head()
x = pd.date_range(start='11/1/2018', end='11/3/2018', freq='H')[:-1]
y1 = pred_1_2_conf['lower Bici']
y2 = pred_1_2_conf['upper Bici']
plt.fill_between(x, y1, y2, alpha=0.6)


# predecir para 1-nov y 2-nov
pred_1_2 = results.get_forecast(steps=24*2).predicted_mean

# agregar fechas y horas a la prediccion
pred_1_2.index = pd.date_range(start='11/1/2018', end='11/3/2018', freq='H')[:-1]
pred_1_2.plot()


# formato de la grafica final
plt.title('Pronóstico de viajes')
plt.ylabel('Cantidad de viajes')
plt.xlabel('Semana lun-29-oct al vie-02-nov')

plt.legend(('Datos originales octubre', 'Pronóstico noviembre'),
           loc='lower left')
plt.savefig('pronostico.png')
plt.show()


# In[148]:


pd.date_range(start='11/1/2018', end='11/3/2018', freq='H')[:-1]

