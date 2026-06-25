emb = model.extract_embedding("speaker.wav")

np.save("embedding.npy", emb)