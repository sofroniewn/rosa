from pyensembl import EnsemblRelease
from rosa.data.sequences import get_all_sequences, get_all_intervals
import pyfaidx


BASE_PTH = "/home/ec2-user/enformer/"
SEQUENCE_LENGTH = 196_608
INTERVAL_ONLY = True
RELEASE_NUMBER = 109
SPECIES = 'Mouse'
# FILE_NAME = 'Homo_sapiens.GRCh38.genes.bed'
FILE_NAME = 'Mus_musculus.genes.bed'


# Open genome
genome = EnsemblRelease(RELEASE_NUMBER, species=SPECIES)

# Get interval around TSS for all genes in gemone
intervals = get_all_intervals(genome, SEQUENCE_LENGTH)

# Save interval to BED file
intervals.to_csv(BASE_PTH + FILE_NAME, sep="\t", header=False, index=False)


# # https://ftp.ensembl.org/pub/release-77/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.toplevel.fa.gz
# FASTA_PT = "/Users/nsofroniew/Documents/data/multiomics/enformer/Homo_sapiens.GRCh38.dna.toplevel.fa"

# # Open dna
# pyfaidx.Faidx(FASTA_PT)
# dna = pyfaidx.Fasta(FASTA_PT)

# # Get sequences around TSS for all genes in gemone
# sequences = get_all_sequences(genome, dna, SEQUENCE_LENGTH)
