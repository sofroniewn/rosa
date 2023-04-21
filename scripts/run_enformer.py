from enformer_pytorch import GenomeIntervalDataset
import numpy as np
from pyensembl import EnsemblRelease
import torch
import torch.nn.functional as F
from enformer_pytorch.data import str_to_one_hot


def gaussian_kernel_1d(kernel_size, sigma):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be an odd number.")
   
    x = torch.arange(start=-kernel_size // 2, end=kernel_size // 2 + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()
   
    return kernel


def gaussian_filter_1d(input_tensor, kernel_size, sigma, padding_mode='reflect'):
    input_tensor = input_tensor.unsqueeze(dim=1).type(torch.float)
    kernel = gaussian_kernel_1d(kernel_size, sigma).type(torch.float).to(DEVICE)
    kernel = kernel.view(1, 1, -1)
    output = F.conv1d(input_tensor, kernel, padding='same')
    return output.squeeze(dim=1)


class MyGenomeIntervalDataset(GenomeIntervalDataset):
    def __init__(self, **kwargs):
        super(MyGenomeIntervalDataset, self).__init__(**kwargs)

    def __getitem__(self, ind):
        item = super().__getitem__(ind)
        label = self.df.row(ind)[4]
        return label, item


class MyGenomeIntervalVariantDataset(GenomeIntervalDataset):
    def __init__(self, **kwargs):
        super(MyGenomeIntervalVariantDataset, self).__init__(**kwargs)

    def __getitem__(self, ind):
        item = super().__getitem__(ind)
        label = self.df.row(ind)[4]
        start = self.df.row(ind)[1]
        pos = self.df.row(ind)[5]
        ref = self.df.row(ind)[6]
        alt = self.df.row(ind)[7]
        loc = pos - start - 1
        assert abs(item[loc] - str_to_one_hot(ref)).max() == 0.0 # confirm ref match
        return label, item, loc, str_to_one_hot(alt)


genome = EnsemblRelease(77)


def get_tss(gene_id, tss, length):
    gene = genome.gene_by_id(gene_id)
    transcripts = [ts for ts in gene.transcripts if ts.biotype == 'protein_coding']
    if len(transcripts) > 0:
        starts = np.array([tss + np.round((ts.start - gene.start) / 128) for ts in gene.transcripts], dtype=int)
        starts = starts[starts>=0]
        starts = starts[starts<length]
    else:
        starts = np.array([tss], dtype=int)
    vector = np.zeros(length)
    vector[starts] = 1.0
    return vector


# def extract_embeddings(embeddings, cage_expression, tss_tensors, sigmas, tss):
#     # Embeddings include
#     #   TSS
#     #   sum over all
#     #   argmax over all
#     #   argmax over TSS
#     #   sum over TSS
#     #   argmax over TSS sigma 3, 8, 16
#     #   sum over TSS sigma 3, 8, 16
#     batch_size = embeddings.shape[0]
#     scaled_cage_expression = cage_expression * tss_tensors

#     tss_emb = embeddings[:, tss]
#     sum_emb = embeddings.sum(dim=1)
#     max_inds = torch.argmax(cage_expression, dim=-1)
#     amax_emb = embeddings[torch.arange(batch_size), max_inds]
    
#     max_inds = torch.argmax(scaled_cage_expression, dim=-1)
#     amax_tss_emb = embeddings[torch.arange(batch_size), max_inds]
#     sum_tss_emb = (embeddings * scaled_cage_expression.unsqueeze(dim=-1)).sum(dim=1)

#     all_emb = [tss_emb, sum_emb, amax_emb, amax_tss_emb, sum_tss_emb]
#     for sigma in sigmas:
#         ks = 2 * int(sigma / 2 * 3) + 1
#         tss_tensors_conv = gaussian_filter_1d(tss_tensors, kernel_size=ks, sigma=sigma)
#         scaled_cage_expression = cage_expression * tss_tensors_conv
#         max_inds = torch.argmax(scaled_cage_expression, dim=-1)

#         amax_tss_emb = embeddings[torch.arange(batch_size), max_inds]
#         sum_tss_emb = (embeddings * scaled_cage_expression.unsqueeze(dim=-1)).sum(dim=1)

#         all_emb.append(amax_tss_emb)
#         all_emb.append(sum_tss_emb)

#     return torch.stack(all_emb, dim=0) # 5 + 2 * len(sigmas)


# def extract_embeddings(embeddings, cage_expression, tss_tensors, sigmas, tss):
#     # Embeddings include
#     #   TSS -1, 0, 1
#     #   argmax over TSS -1, 0, 1
#     batch_size = embeddings.shape[0]
#     len_seq = tss_tensors.shape[1] - 1
#     scaled_cage_expression = cage_expression * tss_tensors

#     tss_emb = embeddings[:, tss]
#     tss_emb_m1 = embeddings[:, tss - 1]
#     tss_emb_1 = embeddings[:, tss + 1]

#     max_inds = torch.argmax(scaled_cage_expression, dim=-1)
#     amax_emb = embeddings[torch.arange(batch_size), max_inds]
#     amax_emb_m1 = embeddings[torch.arange(batch_size), torch.clip(max_inds - 1, 0, len_seq)]
#     amax_emb_1 = embeddings[torch.arange(batch_size), torch.clip(max_inds + 1, 0, len_seq)]
    
#     all_emb = [tss_emb, tss_emb_m1, tss_emb_1, amax_emb, amax_emb_m1, amax_emb_1]
#     return torch.stack(all_emb, dim=0) # 6

def extract_embeddings(embeddings, cage_expression, tss_tensors, sigmas, tss):
    # Embeddings include
    #   argmax over TSS -1, 0, 1 for top 5
    batch_size = embeddings.shape[0]
    len_seq = tss_tensors.shape[1] - 1
    scaled_cage_expression = cage_expression * tss_tensors

    max_inds = torch.argmax(scaled_cage_expression, dim=-1)
    tok_values, topk_inds = torch.topk(scaled_cage_expression, 10, dim=-1)
    topk_inds[tok_values == 0] = tss

    all_emb = []
    for i in range(topk_inds.shape[1]):
        max_inds = topk_inds[:, i]
        amax_emb = embeddings[torch.arange(batch_size), max_inds]
        amax_emb_m2 = embeddings[torch.arange(batch_size), torch.clip(max_inds - 2, 0, len_seq)]
        amax_emb_m1 = embeddings[torch.arange(batch_size), torch.clip(max_inds - 1, 0, len_seq)]
        amax_emb_1 = embeddings[torch.arange(batch_size), torch.clip(max_inds + 1, 0, len_seq)]
        amax_emb_2 = embeddings[torch.arange(batch_size), torch.clip(max_inds + 2, 0, len_seq)]
        all_emb += [amax_emb, amax_emb_m2, amax_emb_m1, amax_emb_1, amax_emb_2]
    return torch.stack(all_emb, dim=0) # 15

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
    import pandas as pd
    import numpy as np
    import polars as pl

    import anndata as ad

    # PATH = '/home/ec2-user/cell_census/tabula_sapiens__sample_single_cell__label_cell_type__processed.h5ad'
    PATH = '/home/ec2-user/cell_census/tabula_sapiens__sample_donor_id__label_cell_type.h5ad'

    adata = ad.read_h5ad(PATH)

    def filter_df_fn(df, shift=0):
        df = df.filter(pl.col("column_5").is_in(list(adata.var_names)))
        df = df.with_columns([
            (pl.col("column_2") + shift),
            (pl.col("column_3") + shift),
        ])
        return df

    torch.multiprocessing.freeze_support()

    BASE_PT = "/home/ec2-user/enformer"
    DEVICE = "cuda:0"

    # BASE_PT = "/Users/nsofroniew/Documents/data/multiomics/enformer"
    # DEVICE = "cpu"

    FASTA_PT = BASE_PT + "/Homo_sapiens.GRCh38.dna.toplevel.fa"
    # GENE_INTERVALS_PT = BASE_PT + "/Homo_sapiens.GRCh38.genes.bed"
    GENE_INTERVALS_PT = BASE_PT + "/Homo_sapiens.GRCh38.genes.clinvar.vcf.bed"
    EMBEDDING_PT = BASE_PT + "/Homo_sapiens.GRCh38.genes.enformer_embeddings.zarr"
    # EMBEDDING_PT_TSS = BASE_PT + "/Homo_sapiens.GRCh38.genes.enformer_embeddings_5x_top10_pc_0.zarr"
    EMBEDDING_PT_TSS = BASE_PT + "/Homo_sapiens.GRCh38.genes.enformer_embeddings_var_0.zarr"
    EMBEDDING_PT_TSS_INDS = BASE_PT + "/Homo_sapiens.GRCh38.genes.enformer_embeddings_inds.zarr"
    MODEL_PT = "EleutherAI/enformer-official-rough"
    TARGET_PT = BASE_PT + '/targets_human.txt'

    print("Converting fasta file")
    pyfaidx.Faidx(FASTA_PT)
    print("Fasta file done")

    model = Enformer.from_pretrained(MODEL_PT, output_heads=dict(human = 5313), use_checkpointing = False)
    model.to(DEVICE)

    ds = MyGenomeIntervalVariantDataset(
        bed_file=GENE_INTERVALS_PT,  # bed file - columns 0, 1, 2 must be <chromosome>, <start position>, <end position>
        fasta_file=FASTA_PT,  # path to fasta file
        return_seq_indices=False,  # return nucleotide indices (ACGTN) or one hot encodings
        rc_aug=False,
        filter_df_fn=filter_df_fn,
    )
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0) # type: DataLoader

    z_inds = zarr.open(EMBEDDING_PT_TSS_INDS)
    genes = list(adata.var_names)
    max_inds_dict = dict(zip(genes, list(z_inds[:])))

    # targets_txt = 'https://raw.githubusercontent.com/calico/basenji/0.5/manuscripts/cross2020/targets_human.txt'
    # df_targets = pd.read_csv(targets_txt, sep='\t')
    df_targets = pd.read_csv(TARGET_PT)
    cage_indices = np.where(df_targets['description'].str.startswith('CAGE:'))[0]

    # Create zarr files
    SEQ_EMBED_DIM = 896
    EMBED_DIM = 3072
    NUM_GENES = len(ds)
    TSS = int(SEQ_EMBED_DIM // 2)

    sigmas = [3, 8, 16, 32, 64]

    paths = (Path(EMBEDDING_PT), Path(EMBEDDING_PT_TSS))
 

    # z_embedding_full = zarr.open(
    #     EMBEDDING_PT,
    #     mode="w",
    #     shape=(NUM_GENES, SEQ_EMBED_DIM, EMBED_DIM),
    #     chunks=(1, SEQ_EMBED_DIM, EMBED_DIM),
    #     dtype='float32',
    # )

    # z_embedding_tss = zarr.open(
    #     EMBEDDING_PT_TSS,
    #     mode="w",
    #     # shape=(5 + 2*len(sigmas), NUM_GENES, EMBED_DIM),
    #     shape=(50, NUM_GENES, EMBED_DIM),
    #     chunks=(1, 1, EMBED_DIM),
    #     dtype='float32',
    # )

    z_embedding_tss = zarr.open(
        EMBEDDING_PT_TSS,
        mode="rw",
        # shape=(5 + 2*len(sigmas), NUM_GENES, EMBED_DIM),
        shape=(NUM_GENES, EMBED_DIM),
        chunks=(1, EMBED_DIM),
        dtype='float32',
    )

    index = 0
    # for labels, batch in tqdm(dl):
    #     batch_size = len(labels)
    #     tss_tensors = []
    #     for label in labels:
    #         tss_tensors.append(torch.from_numpy(get_tss(label, tss=TSS, length=SEQ_EMBED_DIM)))
    #     tss_tensors = torch.stack(tss_tensors, dim=0).to(DEVICE)

    #     # calculate embedding
    #     with torch.no_grad():
    #         output, embeddings = model(batch.to(DEVICE), return_embeddings=True)
            
    #         cage_expression = output['human'][:, :, cage_indices].mean(dim=-1)
    #         # tss_embedding = extract_embeddings(embeddings, cage_expression, tss_tensors, sigmas=sigmas, tss=TSS)

    #         batch_size = embeddings.shape[0]
    #         len_seq = tss_tensors.shape[1] - 1
    #         scaled_cage_expression = cage_expression * tss_tensors

    #         max_inds = torch.argmax(scaled_cage_expression, dim=-1)
        
    #     z_embedding_tss_max_inds[index : index + batch_size] = max_inds.cpu().numpy()
    #     # save full and reduced embeddings
    #     # z_embedding_full[index : index + batch_size] = embeddings
    #     # z_embedding_tss[:, index : index + batch_size] = tss_embedding.cpu().numpy()
    #     index += batch_size

    for labels, batch, pos, alt in tqdm(dl):
        batch_size = len(labels)
        if abs(z_embedding_tss[index : index + batch_size]).max() == 0:
            max_inds = []
            for label in labels:
                max_inds.append(max_inds_dict[label])

            # calculate embedding
            with torch.no_grad():
                # Add mutation
                batch[torch.arange(batch.shape[0]), pos] = alt.squeeze(dim=1)
                output, embeddings = model(batch.to(DEVICE), return_embeddings=True)            
                batch_size = embeddings.shape[0]
                tss_embedding = embeddings[torch.arange(batch_size), max_inds]

            z_embedding_tss[index : index + batch_size] = tss_embedding.cpu().numpy()
        index += batch_size