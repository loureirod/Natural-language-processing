import pre_processing as pre

dataset,labels = pre.load_dataset("MR")

one = dataset[25][0]
all = dataset[25][16]

print(pre.cosine_similarity(one,pre.vector_query("one")))

print("End")