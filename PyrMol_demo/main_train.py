import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import json
import operator
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam

from data_fun import create_dataloader_our
from model_fun import Version3_MultiSub_Contrastive
from schedular import NoamLR
from utils import get_func,remove_nan_label,scaffold_split,fix_train_random_seed


def evaluate_sesstion(dataloader, model, device, metric_fn, metric_dtype, task,save_results=True):
    model.eval()
    all_smiles = []
    all_preds, all_labels = [], []

    for smiles, bg, labels in dataloader:
        bg, labels = bg.to(device), labels.type(metric_dtype)
        pred = model(bg).cpu().detach()

        if task == 'classification':
            pred = torch.sigmoid(pred)
        elif task == 'multiclass':
            pred = torch.softmax(pred, dim=1)

        all_preds.append(pred)
        all_labels.append(labels)
        all_smiles.extend(smiles)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    num_task = all_preds.size(1)
    if num_task > 1:
        metric = 0
        for i in range(num_task):
            metric_i = metric_fn(*remove_nan_label(all_preds[:, i], all_labels[:, i]))
            if not np.isnan(metric_i):
                metric += metric_i
            else:
                print(f'Only one class for task {i}')
        metric /= num_task
    else:
        metric = metric_fn(all_preds, all_labels.reshape(all_preds.shape))
    if save_results:
        all_preds = all_preds.squeeze().tolist()
        all_labels = all_labels.squeeze().tolist()
        results_df = pd.DataFrame({'smiles': all_smiles,'pred': all_preds,'label': all_labels})
        # return results_df,metric.item()
        return results_df, metric
    else:
        # return metric.item()
        return metric

model_dict = {
    "Version3_MultiSub_Contrastive": Version3_MultiSub_Contrastive
}

def train_sesstion(config,model_name,contrastive_weight = False,seeds=[0,100,200,300,400]):
    # epochs = 1
    epochs = 300
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    if model_name == "Version3_MultiSub_Contrastive":
        save_path = f"./{model_name}/{config['data_name']}"

    os.makedirs(save_path,exist_ok=True)
    results = []
    for seed in seeds:
        fix_train_random_seed(seed)
        for fold in range(5):
            patience = 0
            model_args = config
            model = model_dict[model_name](model_args).to(device)


            loss_fn = get_func(config['loss_fn'])
            metric_fn = get_func(config['metric_fn'])
            if config['loss_fn'] in []:
                loss_dtype = torch.long
            else:
                loss_dtype = torch.float32

            if config['metric_fn'] in []:
                metric_dtype = torch.long
            else:
                metric_dtype = torch.float32

            if config['metric_fn'] in ['auc','acc']:
                best = 0
                op = operator.ge
            else:
                best = np.inf
                op = operator.le
            best_epoch = 0
            testloader = create_dataloader_our(config, f'{seed}_fold_{fold}_test.csv', shuffle=False,
                                               train=False)
            if not os.path.exists(os.path.join(save_path, f'{seed}_best_fold{fold}.pt')):
                # trainloader = valloader = testloader
                trainloader = create_dataloader_our(config, f'{seed}_fold_{fold}_train.csv', shuffle=True)
                valloader = create_dataloader_our(config, f'{seed}_fold_{fold}_valid.csv', shuffle=False,train=False)
                print(f'dataset size, train: {len(trainloader.dataset)}, \
                                        val: {len(valloader.dataset)}, \
                                        test: {len(testloader.dataset)}')

                optimizer = Adam(model.parameters())
                scheduler = NoamLR(
                    optimizer=optimizer,
                    warmup_epochs=[config['warmup']],
                    total_epochs=[epochs],
                    steps_per_epoch=len(trainloader.dataset) // config['batch_size'],
                    init_lr=[config['init_lr']],
                    max_lr=[config['max_lr']],
                    final_lr=[config['final_lr']]
                )
                for epoch in tqdm(range(epochs)):
                    model.train()
                    total_loss = 0
                    total_contrastive = 0
                    total_task = 0
                    for _, bg, labels in trainloader:
                        bg, labels = bg.to(device), labels.type(loss_dtype).to(device)

                        # 计算预测值
                        pred = model(bg)

                        # 检查 pred 是否包含 NaN
                        if torch.isnan(pred).any():
                            raise ValueError(f"NaN detected in predictions at epoch {epoch}")

                        num_task = pred.size(1)
                        if num_task > 1:
                            task_loss = 0
                            for i in range(num_task):
                                valid_pred, valid_labels = remove_nan_label(pred[:, i], labels[:, i])
                                if torch.isnan(valid_pred).any() or torch.isnan(valid_labels).any():
                                    raise ValueError(
                                        f"NaN detected in valid_pred or valid_labels at epoch {epoch}, task {i}")
                                loss_i = loss_fn(valid_pred, valid_labels)
                                if torch.isnan(loss_i).any():
                                    continue
                                else:
                                    task_loss += loss_i

                        else:
                            valid_pred, valid_labels = remove_nan_label(pred, labels.reshape(pred.shape))
                            if torch.isnan(valid_pred).any() or torch.isnan(valid_labels).any():
                                raise ValueError(f"NaN detected in valid_pred or valid_labels at epoch {epoch}")
                            task_loss = loss_fn(valid_pred, valid_labels)

                        total_task += task_loss.item()
                        # contrastive loss
                        if contrastive_weight:
                            contrastive_loss = model.compute_contrastive_loss2(temperature = config['temperature'])
                            total_contrastive += contrastive_loss.item()

                            # 检查 loss 是否包含 NaN
                            if torch.isnan(task_loss) or torch.isnan(contrastive_loss):
                                print(f"Warning: NaN loss value detected in step {epoch}.")
                                print(
                                    f"Loss of mandate: {task_loss.item()}, comparative loss: {contrastive_loss.item()}")
                                continue

                            loss = task_loss + contrastive_weight * contrastive_loss
                            total_loss += loss.item()
                        else:
                            # 检查 loss 是否包含 NaN
                            if torch.isnan(task_loss).any():
                                raise ValueError(f"NaN detected in loss at epoch {epoch}")
                            loss = task_loss
                            total_loss += loss.item()



                        optimizer.zero_grad()
                        loss.backward()

                        # 检查梯度是否包含 NaN
                        for name, param in model.named_parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                raise ValueError(f"NaN detected in gradients of {name} at epoch {epoch}")

                        # 进行梯度裁剪
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

                        optimizer.step()
                        scheduler.step()

                    total_loss = total_loss / len(trainloader.dataset)

                    # val
                    model.eval()
                    val_metric = evaluate_sesstion(valloader,model,device,metric_fn,metric_dtype,config['task'],save_results=False)
                    if contrastive_weight:
                        total_task /= len(trainloader.dataset)
                        total_contrastive /= len(trainloader.dataset)
                        print(
                            f"train_loss:{total_loss},task_loss:{total_task}, contrastive_loss:{total_contrastive}, val_metri:{val_metric}")
                    else:
                        print(
                            f"train_loss:{total_loss}, val_metri:{val_metric}")

                    if op(val_metric, best):
                        best = val_metric
                        best_epoch = epoch
                        torch.save(model.state_dict(), os.path.join(save_path, f'{seed}_best_fold{fold}.pt'))
                        patience = 0
                    else:
                        patience += 1
                    if patience == 20:
                        break
                model.load_state_dict(torch.load(os.path.join(save_path, f'{seed}_best_fold{fold}.pt')))
                results_df,test_metric = evaluate_sesstion(testloader,model,device,metric_fn,metric_dtype,config['task'])
                print(
                    f'best epoch {best_epoch} for fold {fold}, val {config["metric_fn"]}:{best}, test: {test_metric}')
            else:
                model.load_state_dict(torch.load(os.path.join(save_path,f'{seed}_best_fold{fold}.pt')))
                results_df,test_metric = evaluate_sesstion(testloader, model, device, metric_fn, metric_dtype, config['task'])
            results_df.to_csv(f"{save_path}/seed{seed}_fold{fold}output.csv", index=False)

            '''# evaluate on testset
            model = Model(model_args).to(device)
            state_dict = torch.load(os.path.join(save_path,f'./best_fold{fold}.pt'))
            model.load_state_dict(state_dict)
            model.eval()
            test_metric = evaluate(testloader,model,device,metric_fn,metric_dtype,data_args['task'])
            results.append(test_metric)
            print(f'best epoch {best_epoch} for fold {fold}, val {train_args["metric_fn"]}:{best}, test: {test_metric}')'''
            #wandb.finish()
            results.append(test_metric)
            print(f'Fold{fold},best test {config["metric_fn"]}:{test_metric}')
    results.append(f'{np.mean(results)}+/-{np.std(results)}')
    results_df = pd.DataFrame(results, columns=[f"{config['metric_fn']}"])
    results_df.to_csv(f"{save_path}/results.csv")

if __name__=='__main__':

    model_name = "Version3_MultiSub_Contrastive"
    config_path = f'./dataset_configs.json'
    configs = json.load(open(config_path, 'r'))

    # for dataset_name in [ "bace", "bbbp","freesolv", "lipophilicity", "esol","toxcast", "clintox", "sider", "tox21"]:
    for dataset_name in ["bace"]:
        print(f"""Train on {dataset_name}""")
        config = configs[dataset_name]
        config['contrastive_weight'] = 0.1 # 是否使用对比学习，设置为0，则为不使用
        print(config)
        train_sesstion(config, model_name, contrastive_weight=config['contrastive_weight'], seeds=[0, 100, 200, 300, 400])
