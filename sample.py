from setup import * 
from questions import * 


sentence = "Evaluate the trade-offs between using relational databases and NoSQL databases for storing and querying large-scale datasets in distributed systems."
prediction = preprocess_and_predict(sentence,tfidf,reference_embeddings)
print(f"Prediction: {prediction}")