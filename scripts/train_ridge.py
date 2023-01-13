from rosa import GeneAnnDataModule
from sklearn import linear_model
import pickle


TABULA_SAPIENS_BY_CELL_TYPE_WITH_EMBEDS_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_features_with_embeds_new.h5ad"
TABULA_SAPIENS_BY_CELL_TYPE_RIDGE_MODEL = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_features_ridge_model_new.sav"
ALPHA = 100

dm = GeneAnnDataModule(TABULA_SAPIENS_BY_CELL_TYPE_WITH_EMBEDS_PT)
dm.setup()

X_train = dm.train_dataset.gene_embedding
y_train = dm.train_dataset.expression.T
model = linear_model.Ridge(alpha=ALPHA).fit(X_train, y_train)

pickle.dump(model, open(TABULA_SAPIENS_BY_CELL_TYPE_RIDGE_MODEL, 'wb'))