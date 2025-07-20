from gensim.models.fasttext import load_facebook_model

# Step 1: Load the binary model
model = load_facebook_model("sample1_model")  # or "sample1_model.bin" if renamed

# Step 2: Save it in Gensim's native pickle format
model.save("sample1_model.model")