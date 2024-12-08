from transformers import AutoModel
import numpy as np
import pandas as pd
import faiss

cos_sim = lambda a,b: (a @ b.T) / (np.norm(a)*np.norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

csv = pd.read_csv('./csv/quotes.csv')

df = csv.copy()
df['id'] = range(len(df))

quotes_array = df['quote'].head(1000)
arr = quotes_array.tolist()

embeddings = model.encode(arr)

embeddings_list = [embedding.tolist() if hasattr(embedding, 'tolist') else embedding for embedding in embeddings]
author = df['author'].head(100)

new_df = pd.DataFrame({
    'quote': quotes_array,
    'author': author,                  
    'embeddings': embeddings_list,                 
})

xq = model.encode('hate')


d = embeddings.shape[1]
index_l2 = faiss.IndexFlatL2(d)
index_l2.is_trained

embeddings_array = np.array(new_df.embeddings.tolist())

index_l2.add(embeddings_array)
index_l2.ntotal

_, document_indices = index_l2.search(np.expand_dims(xq, axis=0), k=10)

results = new_df.iloc[document_indices[0]]

pd.set_option('display.max_colwidth', 1000)
print(results)




