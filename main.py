import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
from catboost import CatBoostClassifier, Pool
import optuna
from optuna_integration import CatBoostPruningCallback

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(columns = ['campaign_success'])
y = train['campaign_success']

cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

for i in cat_features:
    modes = train[i].mode().iloc[0]
    X[i] = X[i].fillna(modes)
    train[i] = train[i].fillna(modes)
    test[i] = train[i].fillna(modes)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=54)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)
test_pool = Pool(test, cat_features=cat_features)

cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(cat_features)

def objective(trial):
    param_optuna = {
        'iterations': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 20),
        'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 128, 254),
        'loss_function': 'Logloss',
        'random_seed': 54,
        'verbose': 1,
        'task_type': 'CPU',
    }
    pruning_callback = CatBoostPruningCallback(trial, 'TotalF1:average=Macro')
    model = CatBoostClassifier(**param_optuna, eval_metric='TotalF1:average=Macro')
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,
        callbacks=[pruning_callback],
        verbose=1,
    )
    preds = model.predict(X_val)
    preds = np.squeeze(preds)
    score = f1_score(y_val, preds, average='macro')
    return score

study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=100)
)
study.optimize(objective, n_trials=30, timeout=1200)

print(f"Лучшие параметры: {study.best_params}")
print(f"Лучший Macro F1: {study.best_value}")

model = CatBoostClassifier(
    **study.best_params,
    iterations=2000,
    random_seed=54,
    early_stopping_rounds=50,
    loss_function='Logloss',
    verbose=100,
    task_type='CPU'
)

model.fit(
    train_pool,
    eval_set=val_pool
)

preds = model.predict(test_pool)
submissions = pd.DataFrame({
    'campaign_success': preds,
})

submissions.to_csv('submissions.cvs', index=False)