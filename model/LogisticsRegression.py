import pickle
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def load_data(path, verbose=True):
    with open(path, 'rb') as file:
        saved_data = pickle.load(file)
        file.close()
    if verbose:
        print("Loaded data from file %s." % path)
    return saved_data

def word_embeddings(path_glove):
    embeddings_index = {}
    f = open(path_glove + '/glove.6B.300d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
           coefs = np.asarray(values[1:], dtype='float32')
           embeddings_index[word] = coefs
        except ValueError:
           pass
    f.close()
    return embeddings_index

def sen2vec(sentence):
    words = str(sentence).lower()
    words = words.split()
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return np.average(M,axis=0)


path_data = ".../data"
path_glove = ".../embedding"
swda_data = load_data(path_data+'/swda.pkl')

embeddings_index = word_embeddings(path_glove)

X = [sen2vec(x) for x in swda_data['text']]
y = swda_data['label']
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


C = 10000
lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', C= C, max_iter=4000)
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))