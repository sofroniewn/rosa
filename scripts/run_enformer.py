from enformer_pytorch import GenomeIntervalDataset

class MyGenomeIntervalDataset(GenomeIntervalDataset):
    def __init__(self, **kwargs):
        super(MyGenomeIntervalDataset, self).__init__(**kwargs)

    def __getitem__(self, ind):
        item = super().__getitem__(ind)
        label = self.df.row(ind)[4]
        return label, item


##############################################################################
##############################################################################

if __name__ == "__main__":
    import torch
    import zarr
    from enformer_pytorch import Enformer
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import pyfaidx
    from pathlib import Path

    torch.multiprocessing.freeze_support()

    BASE_PT = "/home/ec2-user/enformer"
    DEVICE = "cuda:0"

    # BASE_PT = "/Users/nsofroniew/Documents/data/multiomics/enformer"
    # DEVICE = "cpu"

    FASTA_PT = BASE_PT + "/Homo_sapiens.GRCh38.dna.toplevel.fa"
    GENE_INTERVALS_PT = BASE_PT + "/Homo_sapiens.GRCh38.genes.bed"
    EMBEDDING_PT = BASE_PT + "/Homo_sapiens.GRCh38.genes.enformer_embeddings.zarr"
    EMBEDDING_PT_TSS = BASE_PT + "/Homo_sapiens.GRCh38.genes.enformer_embeddings_tss.zarr"
    MODEL_PT = "EleutherAI/enformer-official-rough"

    print("Converting fasta file")
    pyfaidx.Faidx(FASTA_PT)
    print("Fasta file done")

    model = Enformer.from_pretrained(MODEL_PT, output_heads=dict(), use_checkpointing = False)
    model.to(DEVICE)

    ds = MyGenomeIntervalDataset(
        bed_file=GENE_INTERVALS_PT,  # bed file - columns 0, 1, 2 must be <chromosome>, <start position>, <end position>
        fasta_file=FASTA_PT,  # path to fasta file
        return_seq_indices=False,  # return nucleotide indices (ACGTN) or one hot encodings
        rc_aug=False,
    )
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0) # type: DataLoader

    # Create zarr files
    SEQ_EMBED_DIM = 896
    EMBED_DIM = 3072
    NUM_GENES = len(ds)
    TSS = int(SEQ_EMBED_DIM // 2)

    paths = (Path(EMBEDDING_PT), Path(EMBEDDING_PT_TSS))
 

    z_embedding_full = zarr.open(
        EMBEDDING_PT,
        mode="w",
        shape=(NUM_GENES, SEQ_EMBED_DIM, EMBED_DIM),
        chunks=(1, SEQ_EMBED_DIM, EMBED_DIM),
        dtype='float32',
    )

    z_embedding_tss = zarr.open(
        EMBEDDING_PT_TSS,
        mode="w",
        shape=(NUM_GENES, EMBED_DIM),
        chunks=(1, EMBED_DIM),
        dtype='float32',
    )

    index = 0
    for labels, batch in tqdm(dl):
        # calculate embedding
        with torch.no_grad():
            output, embeddings = model(batch.to(DEVICE), return_embeddings=True)
            embeddings = embeddings.detach().cpu().numpy()

        tss_embedding = embeddings[:, TSS]

        # save full and reduced embeddings
        batch_size = len(embeddings)
        z_embedding_full[index : index + batch_size] = embeddings
        z_embedding_tss[index : index + batch_size] = tss_embedding
        index += batch_size