import copy
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time

from src.model import init_cnn


class Server(object):
    def __init__(self, model, args):
        self.args = args
        # federated learning related arguments
        self.r = args.r
        self.k = args.k
        self.b = args.b
        self.nc = args.nc
        # defense related arguments
        self.filter = args.filter
        # model & dataset related arguments
        self.global_model = model
        self.dataset = args.dataset
        # poisoning attacks related arguments
        self.m = args.m
        self.dp = args.dp
        self.mp = args.mp
        self.s = args.s
        self.lamda = args.lamda
        self.device = args.device

        self.test_data = None
        self.loss = nn.CrossEntropyLoss()

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.uploaded_updates = []

        self.rs_test_acc = []
        self.rs_train_loss = []

        self.init_model()
        self.init_clients()

    def init_clients(self):
        for i in range(self.k):
            client = Client(i,self.global_model, self.args)
            self.clients.append(client)

    def init_model(self):
        self.global_model.apply(init_cnn)

    def send_model(self):
        for client in self.clients:
            client.receive_model(self.global_model)

    def load_data(self, data_dict):
        for client in self.clients:
            client.load_data(data_dict['train'][client.id])
        self.test_data = DataLoader(data_dict['test'], batch_size=self.b, shuffle=True)

    def select_clients(self):
        # select all
        self.selected_clients = self.clients

    def receive_model(self):
        self.uploaded_ids = []
        self.uploaded_models = []
        self.uploaded_weights = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)
            self.uploaded_weights.append(client.num_samples)


    def model_to_update(self):
        self.uploaded_updates = []
        global_params = torch.cat([param.data.view(-1) for param in self.global_model.parameters()])
        for model in self.uploaded_models:
            model_params = torch.cat([param.data.view(-1) for param in model.parameters()])
            update = model_params - global_params
            self.uploaded_updates.append(update)  

    def update_to_model(self):
        self.uploaded_models = []
        global_params = torch.cat([param.data.view(-1) for param in self.global_model.parameters()])
        for update in self.uploaded_updates:
            model = copy.deepcopy(self.global_model)
            model_params = global_params + update

            start_idx = 0
            for param in model.parameters():
                param_size = param.numel()
                param.data.copy_(model_params[start_idx:start_idx + param_size].view_as(param))
                start_idx += param_size
            self.uploaded_models.append(model)

    def aggregate_model(self):
        total_updates = torch.zeros_like(self.uploaded_updates[0])
        total_weights = sum(self.uploaded_weights)      # 归一化
        for weight, update in zip(self.uploaded_weights, self.uploaded_updates):
            total_updates += (weight / total_weights) * update
        
        global_params = torch.cat([param.data.view(-1) for param in self.global_model.parameters()])
        global_params += total_updates

        start_idx = 0
        for param in self.global_model.parameters():
            param_size = param.numel()
            param.data.copy_(global_params[start_idx:start_idx + param_size].view_as(param))
            start_idx += param_size

    def poisoning_attack(self):
        pass

    def filter_update(self):
        pass

    def calculate_metrics(self):
        num_train_samples = 0
        total_train_loss = 0
        for client in self.clients:
            train_loss, train_num = client.train_metrics()
            num_train_samples += train_num
            total_train_loss += train_loss
        total_train_loss = total_train_loss / num_train_samples

        num_test_correct = 0
        num_test_samples = 0
        self.global_model.eval()
        with torch.no_grad():
            for x, y in self.test_data:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)
                num_test_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                num_test_samples += y.shape[0]
        total_test_acc = num_test_correct / num_test_samples
        return total_train_loss, total_test_acc

    def get_results(self):
        return self.rs_train_loss, self.rs_test_acc

    def evaluate(self):
        train_loss, test_acc = self.calculate_metrics()
        self.rs_train_loss.append(train_loss)
        self.rs_test_acc.append(test_acc)
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Test Accurancy: {:.4f}".format(test_acc))


    def train(self):
        time_list = []

        for i in range(self.r + 1):
            start_time = time.time()
            # send_model
            self.send_model()

            # evaluate global model
            print(f"\n------------- Round number: {i} -------------")
            print("Evaluate global model")
            s_t = time.time()
            self.evaluate()
            print("evaluate time: {}s".format(time.time() - s_t))

            # select client
            self.select_clients()

            # client train model
            for client in self.selected_clients:
                s_t = time.time()
                client.train()
                print("client {} train time: {}s".format(client.id, time.time() - s_t))
            self.receive_model()
            self.model_to_update()

            # poisoning 
            self.poisoning_attack()

            # defending
            self.filter_update()

            # aggregate
            # self.update_to_model()

            s_t = time.time()
            self.aggregate_model()
            print("aggregation time: {}s".format(time.time() - s_t))

            time_list.append(time.time() - start_time)
            print('-'*15, 'time cost', '-'*15, time_list[-1], 's')

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(time_list[1:])/len(time_list[1:]))
            


class Client(object):
    def __init__(self, id, model, args):
        self.id = id
        self.args = args
        # federated learning related arguments
        self.e = args.e
        self.b = args.b
        self.lr = args.lr
        self.ld = args.ld
        self.ldg = args.ldg
        self.nc = args.nc
        # model & dataset related arguments
        self.model = copy.deepcopy(model)
        self.dataset = args.dataset
        # poisoning attacks related arguments
        self.dp = args.dp
        self.ls = args.ls
        self.lt = args.lt
        self.mp = args.mp
        self.lamda = args.lamda
        # other arguments
        self.device = args.device

        self.num_samples = 0
        self.train_data = None
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)   # model必须提前准备好
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=self.ldg
        )

    def receive_model(self, new_model):
        for param, new_param in zip(self.model.parameters(), new_model.parameters()):
            param.data = new_param.data.clone()

    def load_data(self, dataset):
        self.num_samples = len(dataset)
        self.train_data = DataLoader(dataset, batch_size=self.b, shuffle=True, num_workers=8)

    def train(self):
        self.model.train()
        for epoch in range(self.e):
            s_t1 = time.time()
            t_list = []
            s_t3 = time.time()
            for i, (x, y) in enumerate(self.train_data):
                print("load data time {}".format(time.time() - s_t3))
                s_t2 = time.time()
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                self.optimizer.zero_grad()
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                t_list.append(time.time() - s_t2)
                s_t3 = time.time()
            print("client {} epoch {} timecost/batch {} batch_num {} train time: {}s".format(self.id, epoch, np.mean(t_list), len(t_list), time.time() - s_t1))
        
        if self.ld:
            self.learning_rate_scheduler.step()
    
    def train_metrics(self):
        self.model.eval()

        train_num = 0
        total_loss = 0
        with torch.no_grad():
            for x, y in self.train_data:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                total_loss += loss.item() * y.shape[0]
        return total_loss, train_num

                



