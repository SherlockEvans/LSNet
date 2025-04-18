import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
import shutil
from typing import Dict, List, Union
from tqdm import tqdm
from ifocalloss import *
# from lsnetwork import *



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, Dataset_ASVspoof2021_eval, genSpoof_list)
from data_utils2021 import genSpoof_list2021
from evaluation import calculate_tDCF_EER
from evaluation2021 import calculate_tDCF_EER2021
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)
torch.cuda.is_available()

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "partASVspoof2019.{}".format(track)
    prefix_2021 = "ASVspoof2021.{}".format(track)

    database_path = Path(config["database_path"])
    database_logical_path = Path(config["database_logical_path"])
    dev_trial_path = (database_logical_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    # dev_trial_path = (database_path /
    #                   "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt".format(
    #                       track, prefix_2019))
    eval_trial_path = (database_logical_path /
                       "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
                         track, prefix_2019))
    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"],
        config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    os.makedirs(model_tag, exist_ok=True)
    # make directory for metric logging
    model_save_path = model_tag / "weights"
    metric_path = model_tag / "metrics"
    swa_eval_score_path = model_tag / config["eval_output"]
    pre_eval_score_path = model_tag / "pre_model_eval_scores.txt"
   

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)
    teachermodel = TeacherModel(args, device)
    teachermodel.load_state_dict(torch.load(config["teachermodel_path"], map_location=device))

    # define lossmodel
    lossmodel = None
    lossmodel_optimzer = None
    if config["loss"] == "ocsoftmax":
        lossmodel = OCSoftmax(config["enc_dim"], r_real=config["r_real"], r_fake=config["r_fake"],
                              alpha=config["alpha"]).to(device)
        lossmodel.train()
        lossmodel_optimzer = torch.optim.SGD(lossmodel.parameters(), lr=args.lr)
    elif config["loss"] == "scokdwcedoc":
        lossmodel = DOCSoftmax(config["enc_dim"], r_real=config["r_real"], r_fake=config["r_fake"],
                              alpha=config["alpha"]).to(device)
        lossmodel.train()
        lossmodel_optimzer = torch.optim.SGD(lossmodel.parameters(), lr=args.lr)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, database_logical_path, args.seed, config)

    # sys.exit(0)


    # evaluates pretrained model and exit script
    if args.eval:
        teachermodel.load_state_dict(torch.load(config["model_path"], map_location=device))
        model = teachermodel
        nb_params_1 = sum([param.view(-1).size()[0] for param in model.parameters()])
        print("no. pre-model params:{}".format(nb_params_1))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                pre_eval_score_path, eval_trial_path, lossmodel, config)
        # calculate_tDCF_EER(cm_scores_file=eval_score_path,
        #                    asv_score_file=database_logical_path / config["asv_score_path"],
        #                    output_file=model_tag / "t-DCF_EER.txt")
        print("DONE.")
        eval_eer, eval_tdcf = calculate_tDCF_EER(
            cm_scores_file=pre_eval_score_path,
            asv_score_file=database_logical_path / config["asv_score_path"],
            output_file=model_tag / "pre_model_t-DCF_EER.txt")
        print('eval_eer:{}'.format(eval_eer))
        print('eval_tdcf:{}'.format(eval_tdcf))
        sys.exit(0)


    if args.eval2021LA:
        # define dataloaders
        eval_trial_path_2021LA = (database_logical_path /
                                  "ASVspoof2021_LA_cm_protocols/trial_metadata.txt")
        eval_database_path2021LA = "D:\database\ASV spoof 2021\LA\ASVspoof2021_LA_eval"
        d_label_eval, file_eval = genSpoof_list2021(dir_meta=eval_trial_path_2021LA,
                                                is_train=False,
                                                is_eval=True)
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval,
                                             labels=d_label_eval,
                                             base_dir=eval_database_path2021LA)
        eval_loader_2021LA = DataLoader(eval_set,
                                        batch_size=config["batch_size"],
                                        shuffle=False,
                                        drop_last=False,
                                        pin_memory=True)
        model.load_state_dict(torch.load(config["model_path"], map_location=device))

        nb_params_1 = sum([param.view(-1).size()[0] for param in model.parameters()])
        print("no. pre-model params:{}".format(nb_params_1))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader_2021LA, model, device,
                                pre_eval_score_path, eval_trial_path_2021LA, lossmodel, config, is_2021eval=True)
        # calculate_tDCF_EER(cm_scores_file=eval_score_path,
        #                    asv_score_file=database_logical_path / config["asv_score_path"],
        #                    output_file=model_tag / "t-DCF_EER.txt")
        print("DONE.")
        # eval_eer, eval_tdcf = calculate_tDCF_EER2021(
        #     cm_scores_file=pre_eval_score_path,
        #     asv_score_file=database_logical_path / config["asv_score_path"],
        #     output_file=model_tag / "loaded_model_t-DCF_EER.txt")
        # print('eval_eer:{}'.format(eval_eer))
        # print('eval_tdcf:{}'.format(eval_tdcf))
        sys.exit(0)


    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    else:
        shutil.rmtree(model_save_path)
        os.mkdir(model_save_path)

    copy(args.config, model_tag / "config.conf")


    if not os.path.exists(metric_path):
        os.makedirs(metric_path, exist_ok=True)
    else:
        shutil.rmtree(metric_path)
        os.mkdir(metric_path)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)


    best_dev_eer = 100
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")


    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        if config["loss"] == "ocsoftmax":
            adjust_learning_rate(args, lossmodel_optimzer, epoch)
        elif config["loss"] == "scokdwcedoc":
            adjust_learning_rate(args, lossmodel_optimzer, epoch)
        running_loss = train_epoch(trn_loader, model, teachermodel, optimizer, device,
                                   scheduler, lossmodel, lossmodel_optimzer, config)
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", dev_trial_path, lossmodel, config)
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            asv_score_file=database_logical_path/config["asv_score_path"],
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("Loss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}\nDONE.".format(
            running_loss, dev_eer, dev_tdcf))
       

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            # print("Saving epoch {} for swa".format(epoch))

            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))
            if config["loss"] == "ocsoftmax":
                torch.save(lossmodel, model_save_path / "epoch_{}_loss.pt".format(epoch))
            elif config["loss"] == "scokdwcedoc":
                torch.save(lossmodel, model_save_path / "epoch_{}_loss.pt".format(epoch))

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                eval_score_path_1 = metric_path / "eval_score_{:03d}epo.txt".format(epoch)
                produce_evaluation_file(eval_loader, model, device,
                                        eval_score_path_1, eval_trial_path, lossmodel, config)
                eval_eer, eval_tdcf = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path_1,
                    asv_score_file=database_logical_path / config["asv_score_path"],
                    output_file=metric_path / "t-DCF_EER_{:03d}epo.txt".format(epoch))

                log_text = "epoch{:03d}, ".format(epoch)
                log_text += "best eer:{:.4f}% , ".format(eval_eer)
                log_text += "best tdcf:{:.4f}".format(eval_tdcf)
                if eval_eer < best_eval_eer:
                    best_eval_eer = eval_eer
                if eval_tdcf < best_eval_tdcf:
                    best_eval_tdcf = eval_tdcf
                    torch.save(model.state_dict(),
                               model_save_path / "midbest.pth")
                    if config["loss"] == "ocsoftmax":
                        torch.save(lossmodel, model_save_path / "midlossbest.pt")
                    elif config["loss"] == "scokdwcedoc":
                        torch.save(lossmodel, model_save_path / "midlossbest.pt")
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            optimizer_swa.update_swa()
            n_swa_update += 1
            optimizer_swa.swap_swa_sgd()
            optimizer_swa.bn_update(trn_loader, model, device=device)

       

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, device, swa_eval_score_path,
                            eval_trial_path, lossmodel, config)
    eval_eer, eval_tdcf = calculate_tDCF_EER(cm_scores_file=swa_eval_score_path,
                                             asv_score_file=database_logical_path /
                                             config["asv_score_path"],
                                             output_file=model_tag / "t-DCF_EER.txt")
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("swa EER: {:.3f}, min t-DCF: {:.5f}\n".format(eval_eer, eval_tdcf))
    f_log.write("=" * 5 + "\n")
    f_log.close()

    torch.save(model.state_dict(), model_save_path / "swa.pth")
    if config["loss"] == "ocsoftmax":
        torch.save(lossmodel, model_save_path / "swaloss.pt")
    elif config["loss"] == "scokdwcedoc":
        torch.save(lossmodel, model_save_path / "swaloss.pt")

    print("before final EER: {:.3f}, min t-DCF: {:.5f}".format(best_eval_eer, best_eval_tdcf))
    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(model.state_dict(), model_save_path / "best.pth")
        if config["loss"] == "ocsoftmax":
            torch.save(lossmodel, model_save_path / "bestloss.pt".format(epoch))
        elif config["loss"] == "scokdwcedoc":
            torch.save(lossmodel, model_save_path / "bestloss.pt".format(epoch))
    print("best EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))


def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args.lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(database_path: str, database_logical_path: str, seed: int, config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    prefix_2019 = "partASVspoof2019.{}".format(track)
    prefix_2021 = "ASVspoof2021.{}".format(track)

    # trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    # dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    # eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

    trn_database_path = database_path
    dev_database_path = database_path
    eval_database_path = database_path

    trn_list_path = (database_logical_path /
                     "ASVspoof2019_{}_cm_protocols/{}.cm.train.trl.txt".format(
                         track, prefix_2019))
    dev_trial_path = (database_logical_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (database_logical_path /
                       "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
                        track, prefix_2019))


    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    d_label_dev, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                          is_train=False,
                                          is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            labels=d_label_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    d_label_eval, file_eval = genSpoof_list(dir_meta=eval_trial_path,
                                            is_train=False,
                                            is_eval=True)
    print("no. eval files:", len(file_eval))
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             labels=d_label_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(data_loader: DataLoader, model, device: torch.device,
                            save_path: str, trial_path: str, lossmodel, config: argparse.Namespace, is_2021eval=False) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    model = model.to(device)
    for batch_x, batch_y, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        # print('batch_x', batch_x)
        # print('batch_y', batch_y)
        with torch.no_grad():
            # feat, batch_out = model(batch_x)
            if config["loss"] == "ocsoftmax":
                feat, batch_out = model(batch_x)
                batch_loss, batch_score = lossmodel(feat, batch_y)
            # if config["loss"] == "wce" or config["loss"] == "scokdwcedoc"
            else:
                feat, batch_out = model(batch_x)
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()


        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        if is_2021eval:
            for fn, score in zip(fname_list, score_list):
                fh.write("{} {}\n".format(fn, score))
        else:
            for fn, score, trl in zip(fname_list, score_list, trial_lines):
                _, utt_id, _, tag, label = trl.strip().split(' ')
                # print(fn)
                # print(utt_id)
                assert fn == utt_id
                fh.write("{} {} {} {}\n".format(utt_id, tag, label, score))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,teachermodel,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    lossmodel,
    loss_optim,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    mseloss = nn.MSELoss()
    focalloss = i-FocalLoss()
    


    model = model.to(device)
    teachermodel = teachermodel.to(device)
    for batch_x, batch_y in tqdm(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        # if ii == 3:
        #     print("batch_x's shape:", batch_x.shape)  #torch.Size([12, 64600])
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        feat, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))

        if config["loss"] == "scokdifloss":
            t_feat, t_score = teachermodel(batch_x)
            batch_loss = focalloss(batch_out, batch_y)            

            # begin ！
            beta = 0.5
            ts_loss = mseloss(t_score, batch_out)
            batch_loss = beta * ts_loss + (1-beta) * batch_loss
            # end
            optim.zero_grad()
            batch_loss.backward()
            optim.step()

        running_loss = running_loss + batch_loss.item() * batch_size


        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss = running_loss/num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        default='./config/AASIST.conf',
                        help="configuration file",
                        )
    parser.add_argument("--output_dir",
                        dest="output_dir",
                        type=str,
                        help="output directory for results",
                        default="./exp_result",
                        )
    parser.add_argument("--seed",
                        type=int,
                        default=688,
                        help="random seed (default: 1234)")
    parser.add_argument("--eval",
                        action="store_true",
                        # default="true",
                        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--eval2021LA",
                        action="store_true",
                        default="true",
                        help="when this flag is given, evaluates given model and exit")

    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")

    main(parser.parse_args())
