
## Fun with Word Embeddings
*Christopher Δ 05//2025*
1. load a word2vec embedding model
2. apply the model to a relatively short list of common English words
3. begin at a specified word and engage in a 'random walk' limited to unvisited, similar words
4. store these in an ordered list
5. calculate the first four principal components of the visited words, reducing the embedding dimensionality from 300 to 4
6. generate an animated plot, where x, y, size, and color represent the respective principal components


```python
#!/usr/bin/env python
# coding: utf-8


# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import gensim.downloader as api
from gensim.models import KeyedVectors

MODEL_NAME = "glove-wiki-gigaword-300"   # https://huggingface.co/fse/glove-wiki-gigaword-300
                                         # https://nlp.stanford.edu/projects/glove/
WORD_LIST_PATH = "/usr/share/dict/words" # https://packages.debian.org/sid/wamerican
SUBSET_FILEN = "model_subset.wv"

try:
    model = KeyedVectors.load_word2vec_format(SUBSET_FILEN)
    word_list = [w.strip().lower() for w in open(WORD_LIST_PATH, "r")]
    words_in_common = [w for w in word_list if w in model]
except:
    model = api.load(MODEL_NAME)
    word_list = [w.strip().lower() for w in open(WORD_LIST_PATH, "r")]
    words_in_common = [w for w in word_list if w in model]  # eliminate words that are not modeled
    model = model.vectors_for_all(words_in_common) # subset model
    model.save_word2vec_format(SUBSET_FILEN)



# In[2]:


def random_talk(token, N=5, limit=128, model=model, words_in_common=words_in_common):
    """
    Takes a word, finds the N most similar words (according to cosine),
    chooses one at random.
    If we've been there before, checks others until we find one. Yields word.
    """
    never_here = set(words_in_common)
    been_here = set()
    for i in range(limit):
        if i % 1000 == 0 and i > 1:
            model = model.vectors_for_all(never_here)
        yield(token)
        been_here.add(token)
        never_here.remove(token)
        these = model.most_similar(token, topn=N)
        rando = np.random.randint(N)
        last_token = token
        token = these[rando][0]
        j = 1
        n = N
        while token in been_here:
            token = these[(rando + j) % N][0]
            j += 1
            if i > N:
                n += 1
                these = model.most_similar(last_token, topn=n)
                token = these[n-1][0]

def convert_to_pca_model(w2v_model, n_components=2):
    vectors = np.array([w2v_model[word] for word in w2v_model.index_to_key])
    pca = PCA(n_components=n_components)
    pca.fit(vectors)
    return pca



# In[3]:


words = [w for w in random_talk("pianoforte", 3, 512)]



# In[4]:


pca = convert_to_pca_model(model, n_components=4)
pca_dict = {key: pca.transform(model[key].reshape(1,-1)).reshape(-1) for key in model.index_to_key} # 😎
component_list = [[key] + list(pca_dict[key]) for key in words]
df = pd.DataFrame(component_list, columns = ["token", "pc1", "pc2", "pc3", "pc4"])
positron = -1.01 * min(df.pc3)
df.pc3 = pd.Series([int(x) for x in (df.pc3 + positron)**1.7 * 100])



# In[5]:


print(df)



# In[6]:


ext = 0.25
range_x = [min(df.pc1) - ext, max(df.pc1) + ext]
range_y = [min(df.pc2) - ext, max(df.pc2) + ext]
range_c = [min(df.pc4)      , max(df.pc4)      ]

fig = px.scatter(df, x = "pc1", y = "pc2", size = "pc3", color = "pc4",
                 text="token", size_max=120, height=800, width=800,
                 range_x = range_x, range_y = range_y, range_color = range_c,
                 title="Bouncing around Word Space", subtitle = "size ~ pc3\n",
                 animation_frame = df.index)

fig.update_layout(font=dict(size=20))
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 450
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 420

fig.show()
fig.write_html("word_space_animation.html")


```
