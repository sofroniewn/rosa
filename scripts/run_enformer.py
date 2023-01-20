import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import polars as pl
import numpy as np
from random import randrange, random
from pathlib import Path
from pyfaidx import Fasta

# helper functions


def exists(val):
    return val is not None


def identity(t):
    return t


def cast_list(t):
    return t if isinstance(t, list) else [t]


def coin_flip():
    return random() > 0.5


# genomic function transforms

seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord("a")] = 0
seq_indices_embed[ord("c")] = 1
seq_indices_embed[ord("g")] = 2
seq_indices_embed[ord("t")] = 3
seq_indices_embed[ord("n")] = 4
seq_indices_embed[ord("A")] = 0
seq_indices_embed[ord("C")] = 1
seq_indices_embed[ord("G")] = 2
seq_indices_embed[ord("T")] = 3
seq_indices_embed[ord("N")] = 4
seq_indices_embed[ord(".")] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord("a")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("c")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("g")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("t")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("n")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("A")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
one_hot_embed[ord("C")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
one_hot_embed[ord("G")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
one_hot_embed[ord("T")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
one_hot_embed[ord("N")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
one_hot_embed[ord(".")] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()


def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.fromstring(t, dtype=np.uint8), seq_strs))
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]


def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]


def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]


def seq_indices_to_one_hot(t, padding=-1):
    is_padding = t == padding
    t = t.clamp(min=0)
    one_hot = F.one_hot(t, num_classes=5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out


# augmentations


def seq_indices_reverse_complement(seq_indices):
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims=(-1,))


def one_hot_reverse_complement(one_hot):
    *_, n, d = one_hot.shape
    assert d == 4, "must be one hot encoding with last dimension equal to 4"
    return torch.flip(one_hot, (-1, -2))


def one_hot_reverse(one_hot):  # NJS ADDED
    *_, n, d = one_hot.shape
    assert d == 4, "must be one hot encoding with last dimension equal to 4"
    return torch.flip(one_hot, (-2,))


# processing bed files


class FastaInterval:
    def __init__(
        self,
        *,
        fasta_file,
        context_length=None,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "path to fasta file must exist"

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

    def __call__(self, chr_name, start, end, return_augs=False):
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        chromosome_length = len(chromosome)

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        if exists(self.context_length) and interval_length < self.context_length:
            extra_seq = self.context_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        seq = ("." * left_padding) + str(chromosome[start:end]) + ("." * right_padding)

        if self.return_seq_indices:
            if self.rc_aug and coin_flip():
                seq = seq_indices_reverse_complement(seq)

            return str_to_seq_indices(seq)

        one_hot = str_to_one_hot(seq)

        rc_aug = self.rc_aug  # and coin_flip() NJS OVERWRITE TO ALWAYS RC

        one_hot = one_hot_reverse(one_hot)  # NJS OVERWRITE TO ALWAYS R

        if rc_aug:
            one_hot = one_hot_reverse_complement(one_hot)

        if not return_augs:
            return one_hot

        # returns the shift integer as well as the bool (for whether reverse complement was activated)
        # for this particular genomic sequence

        rand_shift_tensor = torch.tensor([rand_shift])
        rand_aug_bool_tensor = torch.tensor([rc_aug])

        return one_hot, rand_shift_tensor, rand_aug_bool_tensor


class GenomeIntervalDataset(Dataset):
    def __init__(
        self,
        bed_file,
        fasta_file,
        filter_df_fn=identity,
        chr_bed_to_fasta_map=dict(),
        context_length=None,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
    ):
        super().__init__()
        bed_path = Path(bed_file)
        assert bed_path.exists(), "path to .bed file must exist"

        df = pl.read_csv(str(bed_path), sep="\t", has_headers=False)
        df = filter_df_fn(df)
        self.df = df

        # if the chromosome name in the bed file is different than the keyname in the fasta
        # can remap on the fly
        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map

        self.fasta = FastaInterval(
            fasta_file=fasta_file,
            context_length=context_length,
            return_seq_indices=return_seq_indices,
            shift_augs=shift_augs,
            rc_aug=rc_aug,
        )

        self.return_augs = return_augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end = (interval[0], interval[1], interval[2])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        return self.fasta(chr_name, start, end, return_augs=self.return_augs)


##############################################################################
##############################################################################

if __name__ == "__main__":
    import torch
    from enformer_pytorch import Enformer
    from torch.utils.data import DataLoader
    import zarr
    from tqdm import tqdm
    import pyfaidx

    torch.multiprocessing.freeze_support()

    BASE_PT = "/home/ec2-user/enformer"
    DEVICE = "cuda:0"

    FASTA_PT = BASE_PT + "/Homo_sapiens.GRCh38.dna.toplevel.fa"
    GENE_INTERVALS_PT = BASE_PT + "/Homo_sapiens.GRCh38.genes.bed"
    EMBEDDING_PT = BASE_PT + "/Homo_sapiens.GRCh38.genes.enformer_full_embedding_r.zarr"
    REDUCED_EMBEDDING_PT = (
        BASE_PT + "/Homo_sapiens.GRCh38.genes.enformer_reduced_embedding_r.zarr"
    )
    TSS_EMBEDDING_PT = (
        BASE_PT + "/Homo_sapiens.GRCh38.genes.enformer_tss_embedding_r.zarr"
    )
    MODEL_PT = "EleutherAI/enformer-official-rough"

    print("Converting fasta file")
    pyfaidx.Faidx(FASTA_PT)
    print("Fasta file done")

    model = Enformer.from_pretrained(MODEL_PT, output_heads=dict(human=5313))
    model.to(DEVICE)

    ds = GenomeIntervalDataset(
        bed_file=GENE_INTERVALS_PT,  # bed file - columns 0, 1, 2 must be <chromosome>, <start position>, <end position>
        fasta_file=FASTA_PT,  # path to fasta file
        return_seq_indices=False,  # return nucleotide indices (ACGTN) or one hot encodings
        rc_aug=False,
    )

    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    # Create zarr files
    SEQ_EMBED_DIM = 896
    EMBED_DIM = 3072
    NUM_GENES = len(ds)
    TSS = int(SEQ_EMBED_DIM // 2)

    z_embedding_full = zarr.open(
        EMBEDDING_PT,
        mode="w",
        shape=(NUM_GENES, SEQ_EMBED_DIM, EMBED_DIM),
        chunks=(dl.batch_size, SEQ_EMBED_DIM, EMBED_DIM),
    )

    z_embedding_reduced = zarr.open(
        REDUCED_EMBEDDING_PT,
        mode="w",
        shape=(NUM_GENES, EMBED_DIM),
        chunks=(dl.batch_size, EMBED_DIM),
    )

    z_embedding_tss = zarr.open(
        TSS_EMBEDDING_PT,
        mode="w",
        shape=(NUM_GENES, EMBED_DIM),
        chunks=(dl.batch_size, EMBED_DIM),
    )

    index = 0
    for batch in tqdm(dl):
        # calculate embedding
        with torch.no_grad():
            output, embedding = model(batch.to(DEVICE), return_embeddings=True)

        embedding = embedding.detach().cpu().numpy()
        # reduce embedding along sequence axis
        reduced_embedding = embedding.mean(axis=1)
        tss_embedding = embedding[:, TSS]

        # save full and reduced embeddings
        batch_size = len(embedding)
        z_embedding_full[index : index + batch_size] = embedding
        z_embedding_reduced[index : index + batch_size] = reduced_embedding
        z_embedding_tss[index : index + batch_size] = tss_embedding
        index += batch_size
