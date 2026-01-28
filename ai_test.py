from sentence_transformers import SentenceTransformer, util

# 1. Load the "Brain"
# This is a pre-trained model. It has already read millions of sentences.
# Think of it as a translator that speaks both English and Mathematics.
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Define our sentences
# We want to see if the AI knows these two mean roughly the same thing.
sentence1 = "Artificial intelligence is changing the world"
sentence2 = "AI is transforming the global landscape"

# 3. Convert sentences to Embeddings (The Magic Trick)
# The model turns the words into a list of 384 numbers.
# To the computer, sentence1 is now just a coordinate in space.
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

# 4. Calculate Similarity
# 'cos_sim' stands for Cosine Similarity. It measures the angle between 
# the two lists of numbers. Small angle = very similar meaning.
similarity = util.cos_sim(embedding1, embedding2)

# 5. Display the result
score = similarity.item() # Convert the tensor to a simple number
print(f"--- AI Similarity Analysis ---")
print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Similarity Score: {score:.4f} (Ranges from 0 to 1)")

# 6. Logic Check
if score > 0.7:
    print("✅ Result: These sentences have a very similar meaning!")
else:
    print("❌ Result: These sentences are about different topics.")