"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
import utils
from cnn_net import ConvNet2D as net
from data import DoAPredDataset

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default='model_faulty',
                    help="Directory containing params.json")



def train(model, optimizer, loss_fn, dataloader, metric, params):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metric: (function) a function that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            #TODO: fix the dict_ addition
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            # compute model output and loss
            # import pdb;pdb.set_trace()
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch.double())

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metric on this batch
                summary_batch = {}
                # summary_batch['metric'] = metric(output_batch, labels_batch)
                summary_batch['metric'] =metrics(output_batch, labels_batch)              
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metric: (function) a function that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {}
        # summary_batch['metric'] = metrics(output_batch, labels_batch)
        summary_batch['metric'] = metrics(output_batch, labels_batch)
        #accuracy_text(outputs, labels_text, dict_)

        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
    """
    best_val_acc = 0.0
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6, last_epoch=-1)      
    scheduler = StepLR(optimizer, step_size=25, gamma=0.8)

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        # val_metrics = evaluate(model, loss_fn, train_dataloader, metrics, params)
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        scheduler.step()

        val_acc = val_metrics['metric']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

def freeze_layers(model):
    n = 7 #Freeze first 7 layers
    for i ,child in enumerate(model.children()):
        if( i < n ):
            for param in child.parameters():
                param.requires_grad = False
        else:
            break
    #Check frozen layers
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    
    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # create dataloaders

    train_dataset = DoAPredDataset("/data/ssharma497/beam_hw/radar_sim/", exps = ["Experiment_1", "Experiment_2", "Experiment_3"], faulty=True)
    test_dataset = DoAPredDataset("/data/ssharma497/beam_hw/radar_sim/", exps = ["Experiment_1", "Experiment_2", "Experiment_3"], train = False, faulty=True)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
    
    logging.info("- done.")
    
    model = net(params).double().cuda() if params.cuda else net(params).double()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = nn.BCELoss()
    metrics = utils.accuracy
    logging.info("Load the trained model")
    checkpoint = utils.load_checkpoint("model"+"/best.pth.tar", model)
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

    freeze_layers(model)
    # exit()
    train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, loss_fn, metrics, params, args.model_dir)

    # # Define the model and optimizer
    # model = net(params).cuda() if params.cuda else net(params)
    # optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # # fetch loss function and metrics
    # loss_fn = nn.CrossEntropyLoss()
    # # metrics = utils.accuracy
    # metrics = utils_layer_pred.accuracy
    # #accuracy_text(outputs, labels_text, dict_)

    # # Train the model
    # logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    # train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, loss_fn, metrics, params, args.model_dir)