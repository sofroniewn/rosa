from rosa import GeneAnnDataModule
from sklearn import linear_model
import pickle
import numpy as np


TABULA_SAPIENS_BY_CELL_TYPE_WITH_EMBEDS_PT = "/home/ec2-user/cell_census/tabula_sapiens__max__sample_donor_id__label_cell_type.h5ad"
TABULA_SAPIENS_BY_CELL_TYPE_RIDGE_MODEL = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_features_ridge_model_new_norm.sav"
ALPHA = 100
SHUFFLE = False

dm = GeneAnnDataModule(TABULA_SAPIENS_BY_CELL_TYPE_WITH_EMBEDS_PT)
dm.setup()

X_train = dm.train_dataset.gene_embedding.numpy()
y_train = dm.train_dataset.expression.T.numpy()

# Shuffle genes to get a null baseline for model
if SHUFFLE:
    np.random.shuffle(X_train)

model = linear_model.Ridge(alpha=ALPHA).fit(X_train, y_train)

pickle.dump(model, open(TABULA_SAPIENS_BY_CELL_TYPE_RIDGE_MODEL, "wb"))
