# Notes for extracting and visualizing embeddings

from protein_transformer.dataset import VOCAB

# checkpoint = torch.load(chkpt_file_name, map_location=torch.device('cpu'))
v = VOCAB.ints2str(list(range(22)), include_sos_eos=True)

model = None # Load model
a = next(model.encoder.input_embedding.parameters())


# Write out weights as a tab-separated file
with open("emb01.tsv", "w") as f:
    for i in a.detach().numpy():
        f.write("\t".join([str(el.item()) for el in list(i)]) + "\n")

# Write out labels for each
with open("emb01_labels.tsv", "w") as f:
    for l in VOCAB.ints2str(list(range(22)), include_sos_eos=True):
        f.write(f"{l}\n")
