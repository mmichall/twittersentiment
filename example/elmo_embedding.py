from embedding.elmo import elmo_model

embeddings = elmo_model(
    ["the cat is on the mat", "dogs are in the fog"],
    signature="default",
    as_dict=True)["elmo"]

print(embeddings)
