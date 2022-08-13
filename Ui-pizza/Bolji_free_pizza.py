import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc


dfs = {}
for name in ['train', 'test']:
    df = pd.read_json('%s.json' % name)
    df['_data'] = name
    dfs[name] = df

# funkcijom append spajamo podatke iz train i test datoteka u isti df
# ogranicavamo podatke na zajednicke stupce, ukljucujuci indikator za predvidanje
df = dfs['train'].append(dfs['test'])
df = df.reset_index(drop=True)

cols = list(dfs['test'].columns) + ['requester_received_pizza']
df = df[cols]

# preimenujemo stupce radi preglednosti
df.rename(columns={
        'request_title': 'title', 
        'request_text_edit_aware': 'body',
        'requester_received_pizza': 'got_pizza',
}, inplace=True)

# pretvorimo varijablu 'got_pizza' u integer, izbacujemo neiskoristene stupce te na kraju spajamo 'title' i 'body' stupce
df['got_pizza'] = df['got_pizza'].apply(lambda x: -1 if pd.isnull(x) else int(x))

cols_to_keep = ['_data', 'request_id', 'title', 'body', 'got_pizza']
df = df[cols_to_keep]
df.iloc[0]

df['txt_raw'] = df['title'] + ' ' + df['body']

# provjeravamo radi li ispravno
for col in ['title', 'body', 'txt_raw']:
    print  (df.iloc[0][col])
    print ('--')
    
# srediti cemo tekst tako sto cemo pretvoriti sve u mala slova i te izbaciti sve sto nisu slova te 'stopwords'
import re
def clean_txt(raw, remove_stop=False):
    letters_only = re.sub("[^a-zA-Z]", " ", raw) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    
    return " ".join(words)

df['txt_clean'] = df['txt_raw'].apply(clean_txt)

# provjeravamo radi li ispravno
for col in ['txt_raw', 'txt_clean']:
    print (df.iloc[0][col])
    print ('--')

# pripremamo skup podataka za treniranje za daljnju obradu tako da konstruiramo numericko polje koje predstavlja znakove iz teksta radi lakse daljnje implementacije

def get_xy(vectorizer=None, txt_col='txt_clean'):
    if vectorizer is None:
        vectorizer = CountVectorizer()
        
    dg = df[df['_data'] == 'train']

    X = vectorizer.fit_transform(dg[txt_col]).toarray()
    y = dg['got_pizza'].astype(int).to_numpy()

    return X, y
    
# nasumicno podijelimo skup podataka za treniranje i isprobamo defaultni NB model

X, y = get_xy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

model = MultinomialNB().fit(X_train, y_train)

print ("Tocnost na skupu podataka za treniranje: %f" % model.score(X_train, y_train))
print ("Tocnost na skupu podataka za testiranje: %f" % model.score(X_test, y_test))

y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print ("AUC: %f" % auc(fpr, tpr))

# biramo vrijednosti koje maksimiziraju roc i auc
alphas = [1, 5, 10, 25]
min_dfs = [0.001, 0.01, 0.02, 0.05]

# prolazimo kroz petlju varijabli kako bismo pronasli optimalne vrijednosti
best_alpha, best_min_df = None, None
max_auc = -np.inf

for alpha in alphas:
    for min_df in min_dfs:
        
        vectorizer = CountVectorizer(min_df = min_df)

        X, y = get_xy(vectorizer)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

        model = MultinomialNB(alpha=alpha).fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc_val = auc(fpr, tpr)

        if auc_val > max_auc:
            max_auc = auc_val
            best_alpha, best_min_df = alpha, min_df 

                
print ("alpha: %f" % best_alpha)
print ("min_df: %f" % best_min_df)
print ("best auc: %f" % max_auc)

# provjeravamo da je model bolji 
vectorizer = CountVectorizer(min_df = best_min_df)

X, y = get_xy(vectorizer)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

model = MultinomialNB(alpha=best_alpha).fit(X_train, y_train)

print ("Tocnost na skupu podataka za treniranje: %f" % model.score(X_train, y_train))
print ("Tocnost na skupu podataka za testiranje: %f" % model.score(X_test, y_test))

# treniramo cijeli skup podataka za treniranje sa najboljim parametrima i spremamo predvidanja

vectorizer = CountVectorizer(min_df = best_min_df)

X, y = get_xy(vectorizer)

model = MultinomialNB(alpha=best_alpha).fit(X, y)

df_test = df[df['_data'] == 'test'].copy()
X_test = vectorizer.transform(df_test['txt_clean'])
y_pred = model.predict_proba(X_test)[:, 1]

df_test['requester_received_pizza'] = y_pred
final_df = df_test[['request_id', 'requester_received_pizza']]

print (final_df.head(5))

# kreiranje csv datoteke
final_df.to_csv('sampleSubmission.csv', index=False)

# za kraj, pogledajmo koje riječi najbolje garantiraju uspješnost zahtjeva

words = np.array(vectorizer.get_feature_names())

x = np.eye(X.shape[1])
probs = model.predict_proba(x)[:, 1]

word_df = pd.DataFrame()
word_df['word'] = words
word_df['P(pizza | word)'] = probs
word_df.sort_values('P(pizza | word)', ascending=False, inplace=True)

print ('good words')
print (word_df.head(10))
print ('\n---\n')
print ('bad words')
print (word_df.tail(10))