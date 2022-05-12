import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time 
from scipy.stats import uniform
import datetime
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier


class ModelStateReset(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.reset_states()
reset=ModelStateReset()


def LSTM_Model(epochs=1, LSTM_units=1, num_samples=1, look_back=1, num_features=None, dropout_rate=0,recurrent_dropout=0, verbose=0):
    model=Sequential()
    model.add(LSTM(units=LSTM_units, 
                   batch_input_shape=(num_samples, look_back, num_features), 
                   stateful=True, 
                   recurrent_dropout=recurrent_dropout)) 
    model.add(Dropout(dropout_rate))       
    model.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.he_normal(seed=1)))
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

start_sp=datetime.datetime(1980, 1, 1) 
end_sp=datetime.datetime(2019, 2, 28)

yf.pdr_override() 
sp500=pdr.get_data_yahoo('^GSPC', start_sp,end_sp)
sp500.shape

X_train_1, X_test_1, y_train_1, y_test_1=train_test_split(sp500.iloc[:,24:47], sp500.iloc[:,70], test_size=0.1 ,shuffle=False, stratify=None)
X_train_1_lstm=X_train_1.values.reshape(X_train_1.shape[0], 1, X_train_1.shape[1])
X_test_1_lstm=X_test_1.values.reshape(X_test_1.shape[0], 1, X_test_1.shape[1])
X_train_2, X_test_2, y_train_2, y_test_2=train_test_split(sp500.iloc[:,1:24], sp500.iloc[:,70], test_size=0.1 ,shuffle=False, stratify=None)
X_train_2_lstm=X_train_2.values.reshape(X_train_2.shape[0], 1, X_train_2.shape[1])
X_test_2_lstm=X_test_2.values.reshape(X_test_2.shape[0], 1, X_test_2.shape[1])
X_train_3, X_test_3, y_train_3, y_test_3=train_test_split(sp500.iloc[:,47:70], sp500.iloc[:,70], test_size=0.1 ,shuffle=False, stratify=None)
X_train_3_lstm=X_train_3.values.reshape(X_train_3.shape[0], 1, X_train_3.shape[1])
X_test_3_lstm=X_test_3.values.reshape(X_test_3.shape[0], 1, X_test_3.shape[1])
X_train_4, X_test_4, y_train_4, y_test_4=train_test_split(sp500.iloc[:,1:47], sp500.iloc[:,70], test_size=0.1 ,shuffle=False, stratify=None)
X_train_4_lstm=X_train_4.values.reshape(X_train_4.shape[0], 1, X_train_4.shape[1])
X_test_4_lstm=X_test_4.values.reshape(X_test_4.shape[0], 1, X_test_4.shape[1])
X_train_5, X_test_5, y_train_5, y_test_5=train_test_split(sp500.iloc[:,24:70], sp500.iloc[:,70], test_size=0.1 ,shuffle=False, stratify=None)
X_train_5_lstm=X_train_5.values.reshape(X_train_5.shape[0], 1, X_train_5.shape[1])
X_test_5_lstm=X_test_5.values.reshape(X_test_5.shape[0], 1, X_test_5.shape[1])
X_train_6, X_test_6, y_train_6, y_test_6=train_test_split(pd.concat([sp500.iloc[:,1:24], sp500.iloc[:,47:70]], axis=1), sp500.iloc[:,70], test_size=0.1 ,shuffle=False, stratify=None)
X_train_6_lstm=X_train_6.values.reshape(X_train_6.shape[0], 1, X_train_6.shape[1])
X_test_6_lstm=X_test_6.values.reshape(X_test_6.shape[0], 1, X_test_6.shape[1])
X_train_7, X_test_7, y_train_7, y_test_7=train_test_split(sp500.iloc[:,1:70], sp500.iloc[:,70], test_size=0.1 ,shuffle=False, stratify=None)
X_train_7_lstm=X_train_7.values.reshape(X_train_7.shape[0], 1, X_train_7.shape[1])
X_test_7_lstm=X_test_7.values.reshape(X_test_7.shape[0], 1, X_test_7.shape[1])

sp500['Return_Label']=pd.Series(sp500['Log_Ret_1d']).shift(-21).rolling(window=21).sum()
sp500['Label']=np.where(sp500['Return_Label'] > 0, 1, 0)
sp500=sp500.dropna("index")
sp500=sp500.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', "Return_Label"], axis=1)


# Plot the logarithmic returns
sp500.iloc[:,1:24].plot(subplots=True, color='blue', figsize=(20, 20));

# Plot the Volatilities
sp500.iloc[:,24:47].plot(subplots=True, color='blue',figsize=(20, 20));

focus_cols=sp500.iloc[:,24:47].columns 
corr=sp500.iloc[:,24:70].corr().filter(focus_cols).drop(focus_cols)
mask=np.zeros_like(corr); mask[np.triu_indices_from(mask)]=True 
heat_fig, (ax)=plt.subplots(1, 1, figsize=(9,6))
heat=sns.heatmap(corr, 
                   ax=ax, 
                   mask=mask, 
                   vmax=.5, 
                   square=True, 
                   linewidths=.2, 
                   cmap="Blues_r")
heat_fig.subplots_adjust(top=.93)
heat_fig.suptitle('Volatility vs. Volume', fontsize=14, fontweight='bold')
plt.savefig('heat1.eps', dpi=200, format='eps');

print("Training Bias = "+ str(np.mean(y_train_7==1))+"%")
print("Testing Bias = " + str(np.mean(y_test_7==1))+"%")

A3=uniform(loc=0.00001, scale=0.0001) 
ratio_R3=uniform(loc=0, scale=1) 
hyperparameters_r_3_b={'logistic__alpha':A3, 'logistic__l1_ratio':ratio_R3, 'logistic__penalty':penalty_b,'logistic__max_iter':iterations_3_b}
grid_search3b=RandomizedSearchCV(pipeline_b, hyperparameters_r_3_b, n_iter=20, random_state=1, cv=tscv, verbose=0, n_jobs=-1, scoring=scoring_b, refit=metric_b, return_train_score=False)
model_trading_volume=grid_search3b.fit(X_train_3, y_train_3)
print('Loss function:', model_trading_volume.best_estimator_.get_params()['logistic__loss'])
print(metric_b +' of the best model: ', model_trading_volume.best_score_);print("\n")
print("Best hyperparameters:")
print('Number of iterations:', model_trading_volume.best_estimator_.get_params()['logistic__max_iter'])
print('Penalty:', model_trading_volume.best_estimator_.get_params()['logistic__penalty'])
print('Alpha:', model_trading_volume.best_estimator_.get_params()['logistic__alpha'])
print('Ratio:', model_trading_volume.best_estimator_.get_params()['logistic__l1_ratio'])
print("Total number of features:", len(model_trading_volume.best_estimator_.steps[1][1].coef_[0][:]))
print("Number of selected features:", np.count_nonzero(model_trading_volume.best_estimator_.steps[1][1].coef_[0][:]))
plt.title('Gridsearch')
pvt_3_b=pd.pivot_table(pd.DataFrame(model_trading_volume.cv_results_), values='mean_test_accuracy', index='param_logistic__l1_ratio', columns='param_logistic__alpha')
ax_3_b=sns.heatmap(pvt_3_b, cmap="Blues")
plt.show()


## Pot the confusion matrix
y_pred_3_b=model_trading_volume.predict(X_test_3)
fig, ax=plt.subplots()
sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_test_3, y_pred_3_b)), annot=True, cmap="Blues" ,fmt='g')
plt.title('Confusion matrix'); plt.ylabel('Actual label'); plt.xlabel('Predicted label')
ax.xaxis.set_ticklabels(['Down', 'Up']); ax.yaxis.set_ticklabels(['Down', 'Up'])
print("Model  Accuracy:",metrics.accuracy_score(y_test_3, y_pred_3_b))
print("ModelPrecision:",metrics.precision_score(y_test_3, y_pred_3_b))



## LSTM MODEL
start=time.time()
epochs=1
LSTM_units_1_lstm=195
num_features_1_lstm=X_train_1.shape[1]
dropout_rate=0.1
recurrent_dropout=0.1 # 0.21
verbose=0
batch_size=[1] 
hyperparameter_1_lstm={'batch_size':batch_size}

clf_1_lstm=KerasClassifier(build_fn=create_shallow_LSTM, epochs=epochs, LSTM_units=LSTM_units_1_lstm, num_samples=num_samples, look_back=look_back, num_features=num_features_1_lstm, dropout_rate=dropout_rate,recurrent_dropout=recurrent_dropout,verbose=verbose)
search_1_lstm=GridSearchCV(estimator=clf_1_lstm, param_grid=hyperparameter_1_lstm,  n_jobs=-1,  cv=tscv, scoring=scoring_lstm, return_train_score=False)
# Fit the model
tuned_model_1_lstm=search_1_lstm.fit(X_train_1_lstm, y_train_1)
print(scoring_lstm +' score: ', tuned_model_1_lstm.best_score_)
print("Best hyperparameters:")
print('Number of epochs:', tuned_model_1_lstm.best_estimator_.get_params()['epochs'])
print('Batch Size:', tuned_model_1_lstm.best_estimator_.get_params()['batch_size'])
print('Rate:', tuned_model_1_lstm.best_estimator_.get_params()['dropout_rate'])


#Confusion Matrix
y_pred_1_lstm=tuned_model_1_lstm.predict(X_test_1_lstm)
fig, ax=plt.subplots()
sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_test_1, y_pred_1_lstm)), annot=True, cmap="Blues" ,fmt='g')
plt.title('Matrix'); plt.ylabel('Labe'); plt.xlabel('Prediction')
print("Model Accuracy:",metrics.accuracy_score(y_test_1, y_pred_1_lstm))
print("Model Precision:",metrics.precision_score(y_test_1, y_pred_1_lstm))

#ROC Curve
y_proba_1_lstm=tuned_model_1_lstm.predict_proba(X_test_1_lstm)[:, 1]
fpr, tpr, _=metrics.roc_curve(y_test_1,  y_proba_1_lstm)
auc=metrics.roc_auc_score(y_test_1, y_proba_1_lstm)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()

#Model Volatality
grid_search_volatility_volume=GridSearchCV(estimator=pipeline_b, param_grid=hyperparameters_g_2_b, cv=tscv, verbose=0, n_jobs=-1)
model_volatility=grid_search_volatility_volume.fit(X_train_2, y_train_2)
print('Loss function:', model_volatility.best_estimator_.get_params()['logistic__loss'])
print("Hyperparameters:")
print('Iterations:', model_volatility.best_estimator_.get_params()['logistic__max_iter'])
print('Penalty:', model_volatility.best_estimator_.get_params()['logistic__penalty'])
print('Alpha Rate:', model_volatility.best_estimator_.get_params()['logistic__alpha'])
print('Ratio:', model_volatility.best_estimator_.get_params()['logistic__l1_ratio'])


#Model Trading Volume
A3=uniform(loc=0.00001, scale=0.0001) 
ratio_R3=uniform(loc=0, scale=1) 
hyperparameters_r_3_b={'logistic__alpha':A3, 'logistic__l1_ratio':ratio_R3, 'logistic__penalty':penalty_b,'logistic__max_iter':iterations_3
grid_search3b=RandomizedSearchCV(pipeline_b, hyperparameters_r_3_b, n_iter=20, random_state=1, cv=tscv, verbose=0, n_jobs=-1, scoring=scoring_b, refit
model_trading_volume=grid_search3b.fit(X_train_3, y_train_3)
print('Loss function:', model_trading_volume.best_estimator_.get_params()['logistic__loss'])
print(metric_b +' Best Model: ', model_trading_volume.best_score_);print("\n")
print("Hyperparameters:")
print('Iterations:', model_trading_volume.best_estimator_.get_params()['logistic__max_iter'])
print('Penalty:', model_trading_volume.best_estimator_.get_params()['logistic__penalty'])
print('Alpha Rate:', model_trading_volume.best_estimator_.get_params()['logistic__alpha'])
print('Ratio:', model_trading_volume.best_estimator_.get_params()['logistic__l1_ratio'])

#Model Volatility and Return
grid_search_vR=GridSearchCV(estimator=pipeline_b, param_grid=hyperparameters_g_4_b, cv=tscv, verbose=0, n_jobs=-1, scoring=scoring_b, refit=metric_b, )
model_volatility_return=grid_search_vR.fit(X_train_4, y_train_4)
print('Loss function:', model_volatility_return.best_estimator_.get_params()['logistic__loss'])
print("Hyperparameters:")
print('Iterations:', model_volatility_return.best_estimator_.get_params()['logistic__max_iter'])
print('Penalty:', model_volatility_return.best_estimator_.get_params()['logistic__penalty'])
print('Alpha Rate:', model_volatility_return.best_estimator_.get_params()['logistic__alpha'])
print('l1_ratio:', model_volatility_return.best_estimator_.get_params()['logistic__l1_ratio'])

#Model Volatility Volume
grid_search_volatility_volume=GridSearchCV(estimator=pipeline_b, param_grid=hyperparameters_g_4_b, cv=tscv, verbose=0, n_jobs=-1, scoring=scoring_b, refit=metric_b, )
model_volatility_volume=grid_search_volatility_volume.fit(X_train_5, y_train_5)
print('Loss function:', model_volatility_volume.best_estimator_.get_params()['logistic__loss'])
print("Hyperparameters:")
print('Iterations:', model_volatility_volume.best_estimator_.get_params()['logistic__max_iter'])
print('Penalty:', model_volatility_volume.best_estimator_.get_params()['logistic__penalty'])
print('Alpha Rate:', model_volatility_volume.best_estimator_.get_params()['logistic__alpha'])
print('l1_ratio:', model_volatility_volume.best_estimator_.get_params()['logistic__l1_ratio'])

#Model Volume Return
grid_search_return_volume=GridSearchCV(estimator=pipeline_b, param_grid=hyperparameters_g_4_b, cv=tscv, verbose=0, n_jobs=-1, scoring=scoring_b, refit=metric_b, )
model_return_volume=grid_search_return_volume.fit(X_train_6, y_train_6)
print('Loss function:', model_return_volume.best_estimator_.get_params()['logistic__loss'])
print("Hyperparameters:")
print('Iterations:', model_return_volume.best_estimator_.get_params()['logistic__max_iter'])
print('Penalty:', model_return_volume.best_estimator_.get_params()['logistic__penalty'])
print('Alpha Rate:', model_return_volume.best_estimator_.get_params()['logistic__alpha'])
print('l1_ratio:', model_return_volume.best_estimator_.get_params()['logistic__l1_ratio'])

#Model Volatility Return & Volume
grid_search_volatility_return_volume=GridSearchCV(estimator=pipeline_b, param_grid=hyperparameters_g_4_b, cv=tscv, verbose=0, n_jobs=-1, scoring=scoring_b, refit=metric_b, )
model_volatility_return_volume=grid_search_volatility_return_volume.fit(X_train_7, y_train_7)
print('Loss function:', model_volatility_return_volume.best_estimator_.get_params()['logistic__loss'])
print("Hyperparameters:")
print('Iterations:', model_volatility_return_volume.best_estimator_.get_params()['logistic__max_iter'])
print('Penalty:', model_volatility_return_volume.best_estimator_.get_params()['logistic__penalty'])
print('Alpha Rate:', model_volatility_return_volume.best_estimator_.get_params()['logistic__alpha'])
print('l1_ratio:', model_volatility_return_volume.best_estimator_.get_params()['logistic__l1_ratio'])


models = []
models.append(model_volatility_return_volume)
model.append(model_return_volume)
model.append(model_volatility_volume)
model.append(model_volatility_return)
model.append(model_volatility)
model.append(model_return)

#LMST for Volatility 
lsmt_volatility=KerasClassifier(build_fn=create_shallow_LSTM, 
                          epochs=1, 
                          LSTM_units=188, 
                          num_samples=num_samples, 
                          look_back=look_back, 
                          num_features=X_train_1.shape[1] 
                          dropout_rate=dropout_rate,
                          recurrent_dropout=0.1,
                          verbose=0)

grid_search_volatility=GridSearchCV(estimator=lsmt_volatility, 
                          param_grid=1,  
                          n_jobs=-1,  
                          cv=tscv, 
                          scoring=scoring_lstm,
                          refit=True)

lsmt_model_volatility=grid_search_volatility.fit(X_train_1_lstm, y_train_1)
print(Accuracy +' of the best model: ', lsmt_model_volatility.best_score_)


#LMST for  Return
lsmt_return=KerasClassifier(build_fn=create_shallow_LSTM, 
                          epochs=1, 
                          LSTM_units=188, 
                          num_samples=num_samples, 
                          look_back=look_back, 
                          num_features=X_train_2.shape[1] 
                          dropout_rate=dropout_rate,
                          recurrent_dropout=0.1,
                          verbose=0)

grid_search_return=GridSearchCV(estimator=clf_1_lstm, 
                          param_grid=1,  
                          n_jobs=-1,  
                          cv=tscv, 
                          scoring=scoring_lstm,
                          refit=True)

lsmt_model_return=grid_search_return.fit(X_train_2_lstm, y_train_2)
print(Accuracy +' :', lsmt_model_return.best_score_)


#LMST for Volume
lsmt_volume=KerasClassifier(build_fn=create_shallow_LSTM, 
                          epochs=1, 
                          LSTM_units=188, 
                          num_samples=num_samples, 
                          look_back=look_back, 
                          num_features=X_train_3.shape[1] 
                          dropout_rate=dropout_rate,
                          recurrent_dropout=0.1,
                          verbose=0)

grid_search_volume=GridSearchCV(estimator=lsmt_volume, 
                          param_grid=1,  
                          n_jobs=-1,  
                          cv=tscv, 
                          scoring=scoring_lstm,
                          refit=True)

lsmt_model_volume=grid_search_volume.fit(X_train_3_lstm, y_train_3)
print(Accuracy +' :', lsmt_model_volume.best_score_)


#LMST for Volume + Return
lsmt_volume_return=KerasClassifier(build_fn=create_shallow_LSTM, 
                          epochs=1, 
                          LSTM_units=188, 
                          num_samples=num_samples, 
                          look_back=look_back, 
                          num_features=X_train_4.shape[1] 
                          dropout_rate=dropout_rate,
                          recurrent_dropout=0.1,
                          verbose=0)

grid_search_volume_return=GridSearchCV(estimator=lsmt_volume_return, 
                          param_grid=1,  
                          n_jobs=-1,  
                          cv=tscv, 
                          scoring=scoring_lstm,
                          refit=True)

lsmt_model_volume_return=grid_search_volume_return.fit(X_train_4_lstm, y_train_4)
print(Accuracy +' :', lsmt_model_volume_return.best_score_)

#LMST for Volatility + Return
lsmt_volatility_return=KerasClassifier(build_fn=create_shallow_LSTM, 
                          epochs=1, 
                          LSTM_units=188, 
                          num_samples=num_samples, 
                          look_back=look_back, 
                          num_features=X_train_5.shape[1] 
                          dropout_rate=dropout_rate,
                          recurrent_dropout=0.1,
                          verbose=0)

grid_search_volatility_return=GridSearchCV(estimator=lsmt_volatility_return, 
                          param_grid=1,  
                          n_jobs=-1,  
                          cv=tscv, 
                          scoring=scoring_lstm,
                          refit=True)

lsmt_model_volatility_return=grid_search_volatility_return.fit(X_train_5_lstm, y_train_5)
print(Accuracy +' :', lsmt_model_volatility_return.best_score_)

#LMST for Volatility + Volume
lsmt_volatility_volume=KerasClassifier(build_fn=create_shallow_LSTM, 
                          epochs=1, 
                          LSTM_units=188, 
                          num_samples=num_samples, 
                          look_back=look_back, 
                          num_features=X_train_6.shape[1] 
                          dropout_rate=dropout_rate,
                          recurrent_dropout=0.1,
                          verbose=0)

grid_search_volatility_volume=GridSearchCV(estimator=lsmt_volatility_volume, 
                          param_grid=1,  
                          n_jobs=-1,  
                          cv=tscv, 
                          scoring=scoring_lstm,
                          refit=True)

lsmt_model_volatility_volume=grid_search_volatility_volume.fit(X_train_6_lstm, y_train_6)
print(Accuracy +' :', lsmt_model_volatility_volume.best_score_)

#LMST for Volatility + Volume + Return
lsmt_volatility_volume_return=KerasClassifier(build_fn=create_shallow_LSTM, 
                          epochs=1, 
                          LSTM_units=188, 
                          num_samples=num_samples, 
                          look_back=look_back, 
                          num_features=X_train_6.shape[1] 
                          dropout_rate=dropout_rate,
                          recurrent_dropout=0.1,
                          verbose=0)

grid_search_volatility_volume_return=GridSearchCV(estimator=lsmt_volatility_volume_return, 
                          param_grid=1,  
                          n_jobs=-1,  
                          cv=tscv, 
                          scoring=scoring_lstm,
                          refit=True)

lsmt_model_volatility_volume_return=grid_search_volatility_volume_return.fit(X_train_7_lstm, y_train_7)
print(Accuracy +' :', lsmt_model_volatility_volume_return.best_score_)