import math
from dataclasses import dataclass
from typing import Tuple, Union

import anndata as ad
import pandas as pd
import torch
from enformer_pytorch import FastaInterval
from tqdm import tqdm


@dataclass
class Interval:
    name: str
    start: int
    end: int
    strand: str = "."

    @property
    def negative_strand(self):
        return self.strand == "-"

    @property
    def length(self):
        return self.end - self.start

    def center(self, use_strand=True):
        if use_strand:
            add_offset = 0 if self.negative_strand else 1
        else:
            add_offset = 0
        delta = (self.end + self.start) % 2
        center = (self.end + self.start) // 2
        return center + add_offset * delta

    def resize(self, length, use_strand=True):
        if self.negative_strand and use_strand:
            # negative strand
            start = self.center() - length // 2
            end = self.center() + length // 2 + length % 2
        else:
            # positive strand
            start = self.center() - length // 2 - length % 2
            end = self.center() + length // 2
        return Interval(name=self.name, start=start, end=end, strand=self.strand)

    def trim(self, min_value=0, max_value=math.inf):
        start = max(self.start, min_value)
        end = min(self.end, max_value)
        return Interval(name=self.name, start=start, end=end, strand=self.strand)


def request_interval(gene, sequence_length):
    # Create an interval centered on gene start
    interval = Interval(
        name=gene.contig, start=gene.start, end=gene.start, strand=gene.strand
    )
    # Resize interval to be target sequence length
    interval = interval.resize(sequence_length)
    return interval


def extract_interval(dna, interval):
    # Get length of chromosome
    chromosome_length = len(dna[interval.name])
    # Determin padding needed
    pad_upstream = "N" * max(-interval.start, 0)
    pad_downstream = "N" * max(interval.end - chromosome_length, 0)
    # Make sure interval fits in chromosome
    interval = interval.trim(max_value=chromosome_length)
    # Get sequence corresponding to interval - note pyfaidx wants a 1-based interval
    sequence = str(
        dna.get_seq(
            interval.name, interval.start + 1, interval.end, rc=interval.negative_strand
        ).seq
    ).upper()
    # pad sequence to propper length
    return pad_upstream + sequence + pad_downstream


def get_all_sequences(genome, dna, sequence_length):
    sequences_all = {}
    for gene_id in tqdm(genome.gene_ids()):
        # Extract gene based on id
        gene = genome.gene_by_id(gene_id)
        # Generate an interval of the right length around gene start site
        interval = request_interval(gene, sequence_length)
        # Extract sequence
        sequence = extract_interval(dna, interval)
        sequences_all[gene_id] = sequence
    return sequences_all


def get_all_intervals(genome, sequence_length):
    # BED file format
    # https://genome.ucsc.edu/FAQ/FAQformat.html#format1
    intervals_all = []
    for gene_id in tqdm(genome.gene_ids()):
        # Extract gene based on id
        gene = genome.gene_by_id(gene_id)
        # Generate an interval of the right length around gene start site
        interval = request_interval(gene, sequence_length)
        # Extract sequence
        intervals_all.append(
            {
                "chromosome": interval.name,
                "start": interval.start,
                "end": interval.end,
                "strand": interval.strand,
                "name": gene_id,
                "score": 1000,
            }
        )
    df = pd.DataFrame(
        intervals_all, columns=["chromosome", "start", "end", "name", "score", "strand"]
    )
    return df


class AdataFastaInterval:
    def __init__(self, adata: ad.AnnData, fasta_path: str):
        self.adata = adata
        self.fasta = FastaInterval(
            fasta_file=fasta_path,
            context_length=None,
            return_seq_indices=False,
            shift_augs=None,
            rc_aug=False,
        )

    def __getitem__(self, ind: Union[int, torch.Tensor]) -> torch.Tensor:
        if isinstance(ind, torch.Tensor):
            return torch.stack([self[int(i)] for i in ind], dim=0)
        chr_name, start, end = self.adata.var.iloc[int(ind)][
            ["column_1", "column_2", "column_3"]
        ]
        return self.fasta(chr_name, start, end, return_augs=False)

    def __len__(self) -> int:
        return self.adata.n_vars

    @property
    def shape(self) -> Tuple[int, ...]:
        if len(self) > 0:
            return (len(self),) + self[0].shape
        else:
            return (0, 0, 0)
