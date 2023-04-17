
from sklearn.metrics import roc_auc_score, average_precision_score
from models.select_model import get_model
from utils.utils import seed_everything

def get_trainer(opt, get_layer_feature=False):
    seed_everything(seed=opt["seed"])
    
    trainer = get_model(opt, get_layer_feature=get_layer_feature)
        
    score_funcs = {'auc': roc_auc_score, 
                   'ap': average_precision_score}

    return trainer, score_funcs


