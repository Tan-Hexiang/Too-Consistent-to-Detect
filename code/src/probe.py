# Impletement of probe-based uncertainty estimation, including CCS and Supervised Probe
# 

import os
import copy
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import json
import numpy
from datetime import datetime
import logging

class ProbeMethod:
    def __init__(self):
        self.best_probe = None
        self.probe = None

    def train(self):
        raise NotImplementedError

    def estimate(self):
        raise NotImplementedError

    def generate_hd(self):
        raise NotImplementedError

    def load(self, path):
        self.best_probe = torch.load(path)
        self.probe = torch.load(path)

    def save(self, path):
         # 如果目录不存在，则创建目录
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        
        torch.save(self.best_probe, path)
    
    def _get_hd_for_reponse(self, model, tokenizer, prompt, response):
        # get hd for llama chat model
        if prompt != None:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content":response}
            ]
        else:
            print("Should provide prompt")
        with torch.no_grad():
            tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
            output = model(input_ids=tokenized_chat, output_hidden_states=True)
        # tuple:layer (batch_size, sequence_length, hidden_size)
        # hd: (layer, sequence, hd_dim 4096)
        hd = torch.stack([x.squeeze(dim=0) for x in output['hidden_states']], dim=0).cpu()
        return hd


class SupervisedProbe(ProbeMethod):
    def __init__(self):
        super().__init__()
        # Binary Cross-Entropy Loss for binary classification problems
        self.loss_func = torch.nn.BCELoss()
    
    def _single_train(self, train_dataloader, optimizer, device='cuda'):
        """
        Performs a single training epoch.

        Parameters:
        - train_dataloader: DataLoader for training data
        - lr: learning rate
        - weight_decay: weight decay (L2 penalty)
        - device: device to use ('cuda' or 'cpu')

        Returns:
        - average_loss: Average training loss over the epoch
        """
        self.probe.train()
        self.probe.to(device)
       
        
        # Log total loss
        total_loss = []
        
        # Training loop over batches
        for step, (hd, label) in enumerate(train_dataloader):
            hd, label = hd.to(device), label.to(device)
            # Forward pass
            outputs = self.probe(hd)
            # Compute loss
            loss = self.loss_func(outputs.squeeze(), label)
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Log loss
            total_loss.append(loss.item())
        average_loss = sum(total_loss) / len(total_loss)
        return average_loss
        
    def estimate(self, hd_dev, batch_size=1000, device='cuda'):
        """
        Evaluates the probe on hd_dev data and returns the predicted scores.

        Parameters:
        - hd_dev: numpy array of shape (n_dev, hd_dim), input features
        - batch_size: batch size to use during evaluation
        - device: device to use ('cuda' or 'cpu')

        Returns:
        - dev_scores: torch tensor of predicted scores of shape (n_dev,)
        """
        self.probe.eval()
        self.probe.to(device)

        # Convert hd_dev to torch tensor
        hd_dev_tensor = torch.tensor(hd_dev, dtype=torch.float32)

        # Create dev dataset and dataloader
        dev_dataset = torch.utils.data.TensorDataset(hd_dev_tensor)
        dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

        # Collect all predicted scores
        all_scores = []

        with torch.no_grad():
            for batch in dev_dataloader:
                hd = batch[0].to(device)
                scores = self.probe(hd)
                all_scores.append(scores)
        # Concatenate all scores
        dev_scores = torch.cat(all_scores, dim=0).squeeze()
        return dev_scores

    def train(self,
        hd_train, label_train, 
        hd_dev, label_dev, 
        save_dir, 
        epoch=1000, 
        lr=1e-3, 
        batch_size=1000, 
        device="cuda", 
        weight_decay=0.01, 
        writer=None):
        """
        Trains the probe using the provided training data, and evaluates on dev data.

        Parameters:
        - hd_train: numpy array of shape (n_train, hd_dim), input features for training
        - label_train: numpy array of shape (n_train, 1), labels for training
        - hd_dev: numpy array of shape (n_dev, hd_dim), input features for validation
        - label_dev: numpy array of shape (n_dev, 1), labels for validation
        - save_dir: directory where to save the trained model
        - epoch: number of epochs to train
        - lr: learning rate
        - batch_size: batch size for training and evaluation
        - device: device to use ('cuda' or 'cpu')
        - weight_decay: weight decay (L2 penalty)
        - writer: writer for logging (e.g., TensorBoard writer)
        """



        # Set the device
        self.device = device

        # Set the writer
        self.writer = writer

        # Set the save directory
        self.save_dir = save_dir

        if os.path.exists(self.save_dir):
            print("Checkpoint already exists: {}".format(self.save_dir))

        # Convert training data to torch tensors
        hd_train_tensor = torch.tensor(hd_train, dtype=torch.float32)
        # Squeeze labels to ensure correct shape
        label_train_tensor = torch.tensor(label_train, dtype=torch.float32).squeeze()

        # init
        self.probe = MLPProbe(hd_train_tensor.shape[-1])

        # Create train dataset and dataloader
        train_dataset = torch.utils.data.TensorDataset(hd_train_tensor, label_train_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


        # Set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float('inf')
        for train_num in tqdm(range(epoch)):
            # Training phase
            train_loss = self._single_train(train_dataloader, optimizer, device)
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, train_num)
            
            # Validation phase
            dev_score = self.estimate(hd_dev, batch_size, device)
            with torch.no_grad():
                # Ensure label_dev is a torch tensor on the correct device
                label_dev_tensor = torch.tensor(label_dev, dtype=torch.float32).squeeze().to(device)
                dev_loss = self.loss_func(dev_score, label_dev_tensor)
            
            # Log and save the model if it has improved
            if self.writer:
                self.writer.add_scalar('Loss/val', dev_loss.item(), train_num)
            if dev_loss.item() < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = dev_loss.item()
            print("Epoch {}: train_loss = {}, val_loss = {}".format(train_num, train_loss, dev_loss.item()))
        print("Best validation loss: {}".format(best_loss))
        self.save(path=self.save_dir+"/best_probe.pth")
        # !estimate uses probe, not best_probe
        self.probe = copy.deepcopy(self.best_probe)
        return best_loss

    
    def generate_hd(self, model, tokenizer, question, response, token=-1, prompt_template = "Briefly answer the question.\nQuestion: {query}\nAnswer: "):
        # create prompt for sp, and generate hd
        prompt = prompt_template.format(query=question)
        # hd: (layer, sequence, hd_dim 4096)
        hd = self._get_hd_for_reponse(model, tokenizer, prompt, response)
        num_layers = hd.shape[0] 
        all_hds = []
        for layer in range(0,num_layers,1):
            single_hd = numpy.array(hd[layer, token].squeeze())
            all_hds.append(single_hd)
        return all_hds

    def get_hd_for_message(self, model, tokenizer, message):
        with torch.no_grad():
            tokenized_chat = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')
            output = model(input_ids=tokenized_chat, output_hidden_states=True)
        # tuple:layer (batch_size, sequence_length, hidden_size)
        # hd: (layer, sequence, hd_dim 4096)
        hd = torch.stack([x.squeeze(dim=0) for x in output['hidden_states']], dim=0).cpu()
        # single_hd = hd[layer, token].squeeze()  numpy.array(single_hd)
        return hd

    def single_estimate(self, question, response, model, tokenizer, layer=16, token=-1):
        hd = torch.tensor(self.generate_hd(model, tokenizer, question, response, layer, token), dtype=torch.float32)
        hd = hd.to('cuda') 
        self.probe.to('cuda')
        score = self.probe(hd)
        return score.item()

class MLPProbe(nn.Module):
    # torch version code of "The Internal State of an LLM Knows When It's Lying"
    def __init__(self, input_dim):
        super(MLPProbe, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x