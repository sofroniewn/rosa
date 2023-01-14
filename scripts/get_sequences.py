from pyensembl import EnsemblRelease
from rosa.sequences import get_all_sequences, get_all_intervals
import pyfaidx


# https://ftp.ensembl.org/pub/release-77/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.toplevel.fa.gz
FASTA_PT = '/Users/nsofroniew/Documents/data/multiomics/enformer/Homo_sapiens.GRCh38.dna.toplevel.fa'
GENE_INTERVALS_PT = '/Users/nsofroniew/Documents/data/multiomics/enformer/Homo_sapiens.GRCh38.genes.bed'
SEQUENCE_LENGTH = 196_608
INTERVAL_ONLY = True

# Open genome
genome = EnsemblRelease(77)

if INTERVAL_ONLY:
    # Get interval around TSS for all genes in gemone
    intervals = get_all_intervals(genome, SEQUENCE_LENGTH)

    # Save interval to BED file
    intervals.to_csv(GENE_INTERVALS_PT, sep='\t', header=False, index=False)
else:
    # Open dna
    pyfaidx.Faidx(FASTA_PT)
    dna = pyfaidx.Fasta(FASTA_PT)

    # Get sequences around TSS for all genes in gemone
    sequences = get_all_sequences(genome, dna, SEQUENCE_LENGTH)
