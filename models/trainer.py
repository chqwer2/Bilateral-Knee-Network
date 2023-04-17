import os
import gc
import collections
import copy
import time
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

# from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.utils import moveTo
from Data.test_utils.tta import tta_inference
import copy
from torch.nn.parallel import DataParallel, DistributedDataParallel
import cv2


tta_list = [0, 1, 5, 15, 11]

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def get_bare_model(network):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(network, (DataParallel, DistributedDataParallel)):
        network = network.module
    return network

def update_E(netG, netEMA, decay=0.999):
    model_params = OrderedDict(netG.named_parameters())
    shadow_params = OrderedDict(netEMA.named_parameters())
    for name, param in model_params.items():
        # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        # shadow_variable -= (1 - decay) * (shadow_variable - variable)
        shadow_params[name].sub_((1. - decay) * (shadow_params[name] - param))

    model_buffers = OrderedDict(netG.named_buffers())
    shadow_buffers = OrderedDict(netEMA.named_buffers())

    # check if both model contains the same set of keys
    assert model_buffers.keys() == shadow_buffers.keys()

    for name, buffer in model_buffers.items():
        # buffers are copied
        shadow_buffers[name].copy_(buffer)
        

def _run_epoch(opt, cls_models, data_loader,
               device, results, score_funcs, 
               is_bilateral=False, 
               prefix="", 
               desc=None, netEMA=None
               ):
    """
    Train one epoch
    
    Args:
    - model: the PyTorch model
    - optimizer: the object that will update the weights of the network
    - data_loader: DataLoader object that returns tuples of (input, label) pairs.
    - loss_func: the loss function that takes in two arguments, the model outputs and the labels, and returns a score
    - device: the compute lodation to perform training
    - score_funcs: a dictionary of scoring functions to use to evalue the performance of the model
    - is_bilateral: if True, train bilateral view model 
    - prefix: a string to pre-fix to any scores placed into the _results_dictionary
    - desc: a description to use for the prgress bar
    
    Return: loss of the current epoch
    """
    # running_loss = 0.0
    running_loss = []
    y_true = []
    y_pred = []
    # start = time.time()

    [model, netEMA, loss_func, lr_schedule, optimizer] = cls_models

    # for inputs, labels, grades, landmark in tqdm(data_loader, desc=desc, leave=False):
    for inputs_dict in tqdm(data_loader, desc=desc, leave=False):
        labels_left, labels_right = torch.split(inputs_dict["labels"], split_size_or_sections=1, dim=1)
        grades_left, grades_right = torch.split(inputs_dict["grades"], split_size_or_sections=1, dim=1)

        labels = torch.cat([labels_left, labels_right], dim=0)
        grades = torch.cat([grades_left, grades_right], dim=0)

        if not is_bilateral:
            inputs_left, inputs_right = inputs_dict["images"]
            inputs = torch.cat([inputs_left, inputs_right], dim=0)
        else:
            inputs = inputs_dict["images"]

        lmk_left, lmk_right = inputs_dict["lmk"]
        lmk = torch.cat([lmk_left, lmk_right], dim=0)

        if len(labels.shape) == 1:
            labels = labels.view(labels.shape[0], -1)   #
        labels = labels.float()

        # print("inputs:", inputs[0].shape, labels.shape)  # 16, 32

        if len(grades.shape) == 1:
            grades = grades.view(grades.shape[0], -1)
        grades = grades.float()

        inputs = moveTo(inputs, device)
        grades = moveTo(grades, device)
        labels = moveTo(labels, device)
        lmk = moveTo(lmk, device)
        outputs = model(inputs, None)
        loss = loss_func(outputs, grades, labels, lmk)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Momentum
            model.update_gem()

        if prefix == "train":
            update_E(model, netEMA, decay=0.99)  # 0.99 ....


            # running_loss += loss.item()
        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            # Convert the outputs to the probability
            y_hat = torch.sigmoid(outputs[0]).detach().cpu().numpy()

            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())

    # End training epoch

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    results[f"{prefix} loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():

        try:
            results[f"{prefix} {name}"].append(score_func(y_true.flatten(), y_pred.flatten()))
        except:
            results[f"{prefix} {name}"].append(float("NaN"))

    gc.collect()
    
    return np.mean(running_loss) # loss of one epoch

def _run_val_epoch(opt, cls_models, dataloader,
                   device, results, score_funcs, 
                   is_bilateral=False, 
                   prefix="val",  
                   landmark=False, 
                   desc=None, tta=[0]
                   ):

    [model, netEMA, loss_func, lr_schedule, optimizer] = cls_models
    model.eval()
    running_loss = []
    y_true = []
    y_pred = []
    tta_inferencer = tta_inference()

    if landmark == True:
        # for inputs, labels, grades, landmark in tqdm(dataloader, desc=desc, leave=False):
        for inputs_dict in tqdm(dataloader, desc=desc, leave=False):
            labels_left, labels_right = torch.split(inputs_dict["labels"], split_size_or_sections=1, dim=1)
            grades_left, grades_right = torch.split(inputs_dict["grades"], split_size_or_sections=1, dim=1)

            labels = torch.cat([labels_left, labels_right], dim=0)
            grades = torch.cat([grades_left, grades_right], dim=0)

            if not is_bilateral:
                inputs_left, inputs_right = inputs_dict["images"]
                inputs = torch.cat([inputs_left, inputs_right], dim=0)

            lmk_left, lmk_right = inputs_dict["lmk"]
            lmk = torch.cat([lmk_left, lmk_right], dim=0)

            if len(labels.shape) == 1:
                labels = labels.view(labels.shape[0], -1)   #
            labels = labels.float()

            # print("inputs:", inputs[0].shape, labels.shape)  # 16, 32

            if len(grades.shape) == 1:
                grades = grades.view(grades.shape[0], -1)
            grades = grades.float()

            inputs = moveTo(inputs, device)
            grades = moveTo(grades, device)
            labels = moveTo(labels, device)
            lmk = moveTo(lmk, device)
            with torch.no_grad():
                for i in tta:
                    outputs_list = []
                    inputs_tta = tta_inferencer(inputs, i)
                    if prefix == "val":
                        outputs_list.append(model(inputs_tta, None)[0].detach().cpu().numpy())  # logit
                    else:
                        outputs_list.append(netEMA(inputs_tta, None)[0].detach().cpu().numpy())  # logit

            outputs = np.mean(outputs_list, 0)


            loss = loss_func([outputs, None], grades, labels, lmk)
            
            running_loss.append(loss.item())

            if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
                # Convert the outputs to the probability
                y_hat = sigmoid(outputs)

                y_true.extend(labels.tolist())
                y_pred.extend(y_hat.tolist())
    else:
        for inputs_dict in tqdm(dataloader, desc=desc, leave=False):
            labels_left, labels_right = torch.split(inputs_dict["labels"], split_size_or_sections=1, dim=1)
            grades_left, grades_right = torch.split(inputs_dict["grades"], split_size_or_sections=1, dim=1)

            labels = torch.cat([labels_left, labels_right], dim=0)
            grades = torch.cat([grades_left, grades_right], dim=0)

            if not is_bilateral:
                inputs_left, inputs_right = inputs_dict["images"]
                inputs = torch.cat([inputs_left, inputs_right], dim=0)
            else:
                inputs = inputs_dict["images"]

            if len(labels.shape) == 1:
                labels = labels.view(labels.shape[0], -1)   #
            labels = labels.float()

            if len(grades.shape) == 1:
                grades = grades.view(grades.shape[0], -1)
            grades = grades.float()

            inputs = moveTo(inputs, device)
            grades = moveTo(grades, device)
            labels = moveTo(labels, device)
            with torch.no_grad():
                outputs_list = []
                for i in tta:

                    inputs_tta = tta_inferencer(inputs, i)

                    if prefix == "val":
                        outputs_list.append(model(inputs_tta, None)[0].detach().cpu().numpy())  # logit

                    else:
                        outputs_list.append(netEMA(inputs_tta, None)[0].detach().cpu().numpy())  # logit

            outputs = np.mean(outputs_list, 0)

            if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
                # Convert the outputs to the probability
                y_hat = sigmoid(outputs)

                y_true.extend(labels.tolist())
                y_pred.extend(y_hat)
        
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    results[f"{prefix} loss"].append(0)
    for name, score_func in score_funcs.items():

        try:
            results[f"{prefix} {name}"].append(score_func(y_true.flatten(), y_pred.flatten()))
        except:
            results[f"{prefix} {name}"].append(float("NaN"))

    gc.collect()
    
    return 0


# @torch.inference_mode()
def _run_test_epoch(opt, cls_models, data_loader,
               device, results, score_funcs,
               is_bilateral=False,
               prefix="",
               desc=None, tta=[0],  cam=False
               ):

    [model, netEMA, loss_func, lr_schedule, optimizer] = cls_models

    model.eval()
    netEMA.eval()

    # running_loss = 0.0
    running_loss = []
    y_true = []
    y_pred = []
    # start = time.time()
    tta_inferencer = tta_inference()

    os.makedirs("cam_results", exist_ok=True)
    total_n = 0
    # for inputs, labels, grades in tqdm(data_loader, desc=desc, leave=False):
    for inputs_dict in tqdm(data_loader, desc=desc, leave=False):

        labels_left, labels_right = torch.split(inputs_dict["labels"], split_size_or_sections=1, dim=1)
        grades_left, grades_right = torch.split(inputs_dict["grades"], split_size_or_sections=1, dim=1)

        labels = torch.cat([labels_left, labels_right], dim=0)
        grades = torch.cat([grades_left, grades_right], dim=0)


        if not is_bilateral:
            inputs_left, inputs_right = inputs_dict["images"]
            inputs = torch.cat([inputs_left, inputs_right], dim=0)
        else:
            inputs = inputs_dict["images"]

        if len(labels.shape) == 1:
            labels = labels.view(labels.shape[0], -1)  #
        labels = labels.float()

        # print("inputs:", inputs[0].shape, labels.shape)  # 16, 32

        if len(grades.shape) == 1:
            grades = grades.view(grades.shape[0], -1)
        grades = grades.float()

        inputs = moveTo(inputs, device)
        grades = moveTo(grades, device)
        labels = moveTo(labels, device)


        with torch.no_grad():
            outputs_list = []
            for i in tta:
                inputs_tta = tta_inferencer(inputs, i)
                outputs = netEMA(inputs_tta, None, return_features=cam)

                outputs_list.append(outputs[0].detach().cpu().numpy())

        outputs = np.mean(outputs_list, 0)

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            # Convert the outputs to the probability
            y_hat = sigmoid(outputs)

            y_true.extend(labels.tolist())
            y_pred.extend(y_hat)

    # End training epoch

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    results[f"{prefix} loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():

        try:
            results[f"{prefix} {name}"].append(score_func(y_true.flatten(), y_pred.flatten()))
        except:
            results[f"{prefix} {name}"].append(float("NaN"))

    gc.collect()

    return roc_auc_score(y_true.flatten(), y_pred.flatten())

def returnCAM(feature_conv):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    feature_conv = feature_conv.cpu().numpy()

    for idx in range(bz):
        cam = feature_conv[idx]
        cam = cam - cam.min(-1, keepdims=True).min(-2, keepdims=True)

        cam_img = cam /  cam.max(-1, keepdims=True).max(-2, keepdims=True)
        cam_img = np.uint8(255 * cam_img)   # )

        # print("cam_img shape:", cam_img.shape)
        cam_output = []
        for img in cam_img:
            cam_output.append(cv2.resize(img, size_upsample))
        output_cam.append(np.asarray(cam_output))

    return output_cam


def _train_binary_classifier(opt, cls_models, train_loader,
                             val_loader=None,
                             score_funcs=None,
                             epochs=10, device="gpu",
                             checkpoints_dir=None,
                             patience=10,
                             verbose=True,
                             is_bilateral=False,
                            ):
    """"
    Train a binary classifier

    Args:
    - model: the PyTorch model / "Module" to train
    - loss_func: the loss function that takes in batch in two arguments,
                    the model outputs and the labels, and returns a score
    - train_loader: PyTorch DataLoader object that returns tuples of (input, label) pairs
    - val_loader: Optional PyTorch DataLoader to evaluate on after every epoch
    - score_funcs (dict): A dictionary of scoring functions to use to evalue the performance of the model
    - epochs: the number of training epochs to perform
    - device: the compute lodation to perform training
    - checkpoints_dir (str): The checkpoints directory where you save the current training weight
    - lr_schedule: the learning rate schedule used to alter \eta as the model trains.
                    If this is not None than the user must also provide the optimizer to use.
    - optimizer: the method used to alter the gradients for learning.
    - patience: How long to wait after last time validation loss improved.
                    Default: 3
    - verbose: If True, prints a message for each validation loss improvement.
                    Default: True

    Return: DataFrame of results
    """
    if score_funcs == None:
        score_funcs = {} # empty set
    results = collections.defaultdict(list)
    results["epoch"] = []

    results_val = collections.defaultdict(list)
    results_val["epoch"] = []

    [model, netEMA, loss_func, lr_schedule, optimizer] = cls_models

    if optimizer == None:
        optimizer = torch.optim.AdamW(model.parameters())
        del_opt = True
    else:
        del_opt = False

    start = time.time()
    best_epoch_auc = 0.5
    best_epoch_auc_ema = 0.5
    model.to(device)


    normal_bst_weights = collections.deque(maxlen=2)
    ema_bst_weights = collections.deque(maxlen=2)
    update_E(model, netEMA, decay=0)

    for epoch in range(epochs):
        gc.collect()
        model.train()
        train_loss = _run_epoch(opt, cls_models, train_loader,
                                device, results, score_funcs,
                                is_bilateral=is_bilateral,
                                prefix="train",
                                desc=f"Training epoch {epoch+1} of {epochs}")

        # Add the results the the current epoch
        results["epoch"].append(epoch)



        if val_loader is not None:
            val_loss = _run_val_epoch(opt, cls_models, val_loader,
                                      device, results_val, score_funcs,
                                      is_bilateral=is_bilateral,
                                      prefix="val",
                                      landmark=False, tta=tta_list,
                                      desc=f"Validating epoch {epoch+1} of {epochs}")
            val_auc = results_val["val auc"][epoch]
            print("[Epoch: %i] [Train Loss: %.5f] [Val Loss: %.5f] [Val AUC: %.3f] [Val AP: %.3f]"
                  %(epoch+1, train_loss, val_loss, val_auc, results_val["val ap"][epoch]))

            val_loss_ema = _run_val_epoch(opt, cls_models, val_loader,
                                          device, results_val, score_funcs,
                                          is_bilateral=is_bilateral,
                                          prefix="valema",
                                          landmark=False, tta=tta_list,
                                          desc=f"Validating epoch {epoch+1} of {epochs}")
            val_auc_ema = results_val["valema auc"][epoch]
            print("[Epoch: %i] [Train Loss: %.5f] [Val EMA Loss: %.5f] [Val AUC: %.3f] [Val AP: %.3f]"
                  %(epoch+1, train_loss, val_loss_ema, val_auc_ema, results_val["valema ap"][epoch]))

        # Deep copy the model
        if best_epoch_auc < val_auc:
            print(f"Epoch {epoch + 1} AUC Improved ({best_epoch_auc:.3f} --> {val_auc:.3f})")
            best_epoch_auc = val_auc
            best_normal_wts = copy.deepcopy(model.state_dict())
            # PATH = f"{checkpoints_dir}/E{epoch+1}_AUC{best_epoch_auc:.3f}.pth"
            # torch.save(model.state_dict(), PATH)
            normal_bst_weights.append({f"E{epoch+1}_AUC{best_epoch_auc:.3f}": best_normal_wts})


        if best_epoch_auc_ema < val_auc_ema:
            print(f"Epoch {epoch + 1} EMA_AUC Improved ({best_epoch_auc_ema:.3f} --> {val_auc_ema:.3f})")
            best_epoch_auc_ema = val_auc_ema

        # The convention is to update the learning rate after every epoch
        if lr_schedule is not None:
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val loss"][-1])
            else:
                lr_schedule.step()

    if del_opt:
        del optimizer

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUC: {:.3f}".format(best_epoch_auc))

    # Finish training
    # Return the results to a dataframe
    print("Save weights ...")
    for normal_bst_weight in normal_bst_weights:
        for k, v in normal_bst_weight.items():
            PATH = f"{checkpoints_dir}/{k}.pth"
            torch.save(v, PATH)

    for ema_bst_weight in ema_bst_weights:
        for k, v in ema_bst_weight.items():
            PATH = f"{checkpoints_dir}/{k}.pth"
            torch.save(v, PATH)

    return pd.DataFrame.from_dict(results), best_normal_wts

def replace_key_name_for_feature_only(state_dict):

    for name in list(state_dict.keys()):

        if "stages." in name:
            state_dict[name.replace("stages.", "stages_")] = state_dict.pop(name)
        if "stem." in name:
            state_dict[name.replace("stem.", "stem_")] = state_dict.pop(name)
        if "head." in name:
            state_dict.pop(name)

    return state_dict


class Trainer():
    def __init__(self, opt, cls_model, lmk_model,
                 checkpoints_dir="./checkpoints",
                 ) -> None:

        # CLS

        # cls_model = cls_model, lmk_model = lmk_model)
        self.opt = opt
        self.model = cls_model['model']
        self.netEMA = copy.deepcopy(self.model).eval()
        for param in self.netEMA.parameters():
            param.detach_()

        update_E(self.model, self.netEMA, decay=0)


        self.optimizer = cls_model['optimizer']
        self.loss_funcs = cls_model['loss_funcs']
        self.lr_schedule = cls_model['lr_schedule']
        self.checkpoints_dir = checkpoints_dir

        # Landmark
        self.lmk_model = lmk_model['model']
        self.lmk_netEMA = copy.deepcopy(self.lmk_model).eval()
        for param in self.lmk_netEMA.parameters():
            param.detach_()

        update_E(self.lmk_model, self.lmk_netEMA, decay=0)

        self.lmk_optimizer = lmk_model['optimizer']
        self.lmk_loss_funcs = lmk_model['loss_funcs']
        self.lmk_lr_schedule = lmk_model['lr_schedule']

        self.checkpoints_dir = checkpoints_dir


    def train(self,  train_loader, val_loader,
              is_bilateral=False,
              epochs:int = 10,
              score_funcs:dict = None,
              device:str = 'cpu',
              desc="test",
            ):

        # Name for checkpoint
        checkpoint_name = desc

        checkpoints_dir = f"{self.checkpoints_dir}/{checkpoint_name}"
        os.makedirs(checkpoints_dir, exist_ok=True)

        df_results, best_normal_wts, best_ema_wts = _train_binary_classifier(
            opt=self.opt,
            cls_models=[self.model, self.netEMA,
                       self.loss_funcs, self.lr_schedule,self.optimizer],
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            checkpoints_dir=checkpoints_dir,
            score_funcs=score_funcs,
            device=device,
            is_bilateral=is_bilateral)

        self.model.load_state_dict(best_normal_wts)
        self.netEMA.load_state_dict(best_ema_wts)
        print("Load best model weights")
        return df_results

    def test(self, test_loader, score_funcs:dict,
             best_model_wts:str=None,
             is_bilateral=False,
             device="cpu", cam=False, tta=tta_list):


        print("Testing Model...")

        if best_model_wts:
            checkpoints = torch.load(best_model_wts,
                                     map_location=torch.device(device))

            self.model.load_state_dict(checkpoints)  #replace_key_name_for_feature_only(checkpoints))

            # try:
            #     self.model.load_state_dict(checkpoints)
            #
            # except:
            #     self.model.load_state_dict(replace_key_name_for_feature_only(checkpoints))


        if score_funcs == None:
            score_funcs = {} # empty set
        results = collections.defaultdict(list)

        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            test_loss = _run_test_epoch(
                                   opt=self.opt,
                                   cls_models=[self.model, self.model,
                                   self.loss_funcs, self.lr_schedule,self.optimizer],
                                   data_loader=test_loader,
                                   device=device,
                                   results=results,
                                   score_funcs=score_funcs,
                                   is_bilateral=is_bilateral,
                                   prefix="test", tta=tta, cam=cam)
        results = pd.DataFrame.from_dict(results)
        print(results)

    def test_ema(self, test_loader, score_funcs:dict,
             best_model_wts:str=None,
             is_bilateral=False,
             device="cpu", cam=False, tta=tta_list):

        print("Testing Model...")

        if best_model_wts:
            checkpoints = torch.load(best_model_wts,
                                     map_location=torch.device(device))
            self.netEMA.load_state_dict(checkpoints)

        if score_funcs == None:
            score_funcs = {} # empty set
        results = collections.defaultdict(list)

        self.netEMA.eval()
        self.netEMA.to(device)
        with torch.no_grad():
            test_loss = _run_test_epoch(
                opt=self.opt,
                cls_models=[self.netEMA, self.netEMA,
                            self.loss_funcs, self.lr_schedule, self.optimizer],
                            data_loader=test_loader,
                            device=device,
                            results=results,
                            score_funcs=score_funcs,
                            is_bilateral=is_bilateral,
                            prefix="testema", tta=tta_list, cam=cam)
        results = pd.DataFrame.from_dict(results)
        print(results)

            

