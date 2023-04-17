import warnings
from utils.utils import seed_everything
from functions import *
from options.parser import *

warnings.filterwarnings('ignore')
args=parser.parse_args()  
DEVICE = "cuda"
seed_everything(seed=opt["seed"])

from sklearn.metrics import roc_auc_score, average_precision_score
from models.select_model import get_model
from utils.utils import seed_everything


def get_trainer(opt):
    seed_everything(seed=opt["seed"])

    trainer = get_model(model_name=opt["model_name"], in_chans=1)

    score_funcs = {'auc': roc_auc_score,
                   'ap': average_precision_score}

    return trainer, score_funcs



train_dl, val_dl, test_dls = get_datasets(train_path=opt["datasets"]["train_path"],
                                        test_path1=opt["datasets"]["test_path1"],
                                        test_path2=opt["datasets"]["test_path2"],
                                        bs=opt["batch_size"], num_workers=opt["num_workers"])


trainer, score_funcs = get_trainer(opt)
 
df_results = trainer.train(train_loader=train_dl, 
                        val_loader=val_dl, 
                        test_loaders=test_dls,
                        epochs=opt["epoch"],
                        score_funcs=score_funcs, 
                        device=DEVICE, 
                        is_bilateral=opt["IS_BILATERAL"],
                        desc=opt["DESCE"],
                        )


# df_results.to_csv("Data/csv_data/csv_results/bilateral_PHYSBOnet.csv", index=False)
print("Test normal ...")
trainer.test(test_loader=test_dls[0], 
                score_funcs=score_funcs, 
                is_bilateral=opt["IS_BILATERAL"], 
                device=DEVICE)
trainer.test(test_loader=test_dls[1], 
                score_funcs=score_funcs, 
                is_bilateral=opt["IS_BILATERAL"], 
                device=DEVICE)

print("Test ema ...")
trainer.test_ema(test_loader=test_dls[0], 
                score_funcs=score_funcs, 
                is_bilateral=opt["IS_BILATERAL"], 
                device=DEVICE)
trainer.test_ema(test_loader=test_dls[1], 
                score_funcs=score_funcs, 
                is_bilateral=opt["IS_BILATERAL"], 
                device=DEVICE)