import os
import copy
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import h5py
import logging
import json
from pathlib import Path

from src.model import init_cnn
from src.utils import read_client_data
from src.attack_methods import min_max_attack, LIE_attack, sign_flip_attack, enhanced_sign_flip_attack, global_sign_flip_attack, random_attack, CAMP_attack, scale_attack, init_MPAF_model, poisonedfl_attack
from src.defend_methods import krum, median, trimmed, multi_krum, selective_mean, dpd, lbfgs_torch, fld_distance, detection, detection1, flame, maud_norm_filter, maud_cosine_filter

logger = logging.getLogger('client')

class Server(object):
    def __init__(self, model, times, args):
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
        self.ls = args.ls
        self.lt = args.lt
        self.mp = args.mp
        self.s = args.s
        self.lamda = args.lamda
        self.trmean_ratio = args.trmean_ratio
        self.device = args.device
        self.times = times
        self.vector_s = None # for CAMP attack
        self.dpd_mode = args.dpd_mode
        self.noise_level = args.noise_level
        self.MPAF_model = None

        self.test_data = None
        self.loss = nn.CrossEntropyLoss()

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.uploaded_updates = []

        self.local_updates = []
        self.old_updates = []
        self.weight_record = []
        self.update_record = []
        self.last_update = None
        self.weight = None
        self.last_weight = None
        self.malicious_score = torch.zeros((1, self.k))
        self.poisonedfl_state = {}
        self._round_idx = 0
        self.maud_accumulated = {}  # MAUD: per-client accumulated update history
        self.maud_window = args.maud_window

        self.C_t = 0 # clipping value

        self.rs_test_acc = []
        self.rs_train_loss = []
        self.rs_asr = []

        self.init_model()
        self.init_clients()

    def init_clients(self):
        for i in range(self.k):
            client = Client(i,self.global_model, self.args)
            self.clients.append(client)

    def init_model(self):
        self.global_model.apply(init_cnn)
        if self.mp == 'MPAF':
            self.MPAF_model = init_MPAF_model(self.global_model)
            logger.info("Initialized MPAF model") 

    def send_model(self):
        for client in self.clients:
            client.receive_model(self.global_model)

    def load_data(self):
        for client in self.clients:
            if self.dp == 'lf' and client.id < self.m:
                client.load_data(is_lf=True, ls=self.ls, lt=self.lt)
            else:
                client.load_data()

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
        # 通过聚合uploaded_updates中的本地更新获得全局更新
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
        
        if self.filter == 'flame':
            for param in self.global_model.parameters():
                temp = torch.normal(mean=0, std=self.C_t * self.noise_level, size=param.size()).to(self.device)
                param.data += temp

    def poisoning_attack(self):
        if self.mp == 'min-max':
            self.uploaded_updates, self.uploaded_weights, _ = min_max_attack(
                self.uploaded_updates, self.uploaded_weights, self.m
            )
        elif self.mp == 'LIE':
            self.uploaded_updates, self.uploaded_weights = LIE_attack(
                self.uploaded_updates, self.uploaded_weights, self.m, self.k
            )
        elif self.mp == 'sign_flip':
            self.uploaded_updates, self.uploaded_weights = sign_flip_attack(
                self.uploaded_updates, self.uploaded_weights, self.m
            )
        elif self.mp == 'enhanced_sign_flip':
            self.uploaded_updates, self.uploaded_weights, _ = enhanced_sign_flip_attack(
                self.uploaded_updates, self.uploaded_weights, self.m
            )
        elif self.mp == 'global_sign_flip':
            self.uploaded_updates, self.uploaded_weights, _ = global_sign_flip_attack(
                self.uploaded_updates, self.uploaded_weights, self.m
            )
        elif self.mp == 'random':
            self.uploaded_updates, self.uploaded_weights, _ = random_attack(
                self.uploaded_updates, self.uploaded_weights, self.m
            )
        elif self.mp == 'CAMP':
            if self.vector_s is None and self.args.CAMP_mode != 'clipping_v4':
                print(self.uploaded_updates[0].device)
                self.vector_s = torch.sign(torch.randn_like(self.uploaded_updates[0]))
                self.vector_s[self.vector_s == 0] = 1 # to avoid zero vectors
            self.uploaded_updates, self.uploaded_weights, self.vector_s = CAMP_attack(
                self.uploaded_updates, self.uploaded_weights, self.m, 
                self.args.CAMP_mode, self.filter, self.vector_s, self.args.lamda, self.args.pk,
                self.uploaded_models, self.noise_level, self.m
            )
        elif self.mp == 'PoisonedFL':
            self.uploaded_updates, self.uploaded_weights, self.poisonedfl_state = poisonedfl_attack(
                updates=self.uploaded_updates,
                weights=self.uploaded_weights,
                num_attackers=self.m,
                state=self.poisonedfl_state,
                round_idx=self._round_idx,
                scaling_factor=1e5,     # 你也可以做成 args
                adjust_period=50,       # 同上
                global_model_vec=None,
                global_model_vec_prev_period=None,
                last_global_grad=None,
                jitter_ratio=0.0
            )
        elif self.mp == 'scale':
            self.uploaded_updates = scale_attack(self.uploaded_updates, self.m, self.s)

    def filter_update(self, epoch):
        if self.filter == 'avg':
            pass
            # update_norms = [torch.norm(update) for update in self.uploaded_updates]
            # median_norm = torch.median(torch.stack(update_norms))
            # clipped_updates = []
            # for update in self.uploaded_updates:
            #     current_norm = torch.norm(update)
            #     if current_norm > median_norm:
            #         clipped_update = update * (median_norm / current_norm)
            #         clipped_updates.append(clipped_update)
            #     else:
            #         clipped_updates.append(update)
            # self.uploaded_updates = clipped_updates
        elif self.filter == 'krum':
            selected_id = krum(self.uploaded_updates, self.m)
            self.uploaded_ids = [self.uploaded_ids[selected_id]]
            self.uploaded_updates = [self.uploaded_updates[selected_id]]
            self.uploaded_weights = [self.uploaded_weights[selected_id]]
            logger.info("Krum select client: {}".format(selected_id))
        elif self.filter == 'median':
            selected_update = median(self.uploaded_updates)
            self.uploaded_updates = [selected_update]
            self.uploaded_weights = [1]
        elif self.filter == 'trmean':
            selected_update, trim_ratios = trimmed(self.uploaded_updates, self.trmean_ratio, track_trimmed=True)
            atk_trim = trim_ratios[:self.m].tolist()
            ben_trim = trim_ratios[self.m:].tolist()
            logger.info(f"[trmean] attacker trim ratios: {[f'{r:.3f}' for r in atk_trim]}")
            if atk_trim: logger.info(f"[trmean] attacker avg trim: {sum(atk_trim)/len(atk_trim):.4f}, benign avg trim: {sum(ben_trim)/len(ben_trim):.4f}")
            self.uploaded_updates = [selected_update]
            self.uploaded_weights = [1]
        elif self.filter == 'multi-krum':
            selected_indices = multi_krum(self.uploaded_updates, self.m)
            self.uploaded_ids = [self.uploaded_ids[i] for i in selected_indices]
            self.uploaded_updates = [self.uploaded_updates[i] for i in selected_indices]
            self.uploaded_weights = [self.uploaded_weights[i] for i in selected_indices]
            logger.info("Multi-krum select clients: {}".format(selected_indices))
        elif self.filter == 'sad':
            selected_update = selective_mean(self.uploaded_updates, self.args)
            self.uploaded_updates = [selected_update]
            self.uploaded_weights = [1]
        elif self.filter == 'dpd':
            # tmp1 = copy.deepcopy(self.uploaded_updates[0])
            # print("before dpd , update norm:", torch.norm(tmp1))
            dpd(self.uploaded_updates, self.dpd_mode, self.noise_level)
            # print("after dpd , update norm:", torch.norm(self.uploaded_updates[0]))
            # print("different update norm:", torch.norm(tmp1 - self.uploaded_updates[0]))
        elif self.filter == 'FLDetector':
            N = 5
            self.weight = torch.cat([param.data.view(-1) for param in self.global_model.parameters()])
            self.local_updates = [uploaded_update.cpu() * -1 for uploaded_update in self.uploaded_updates]
            
            if epoch > 0:
                update = self.last_weight - self.weight

            if epoch > 1:
                self.update_record.append(update.cpu() - self.last_update.cpu())
            
            if epoch > N + 1:
                del self.weight_record[0]
                del self.update_record[0]

            if epoch > 0:
                self.last_update = update


            if epoch > N + 1:
                hvp = lbfgs_torch(self.weight_record, self.update_record, self.weight - self.last_weight)

                distance = fld_distance(self.old_updates, self.local_updates, hvp)
                distance = distance.view(1, -1)

                self.malicious_score = torch.cat((self.malicious_score, distance), dim=0)
                if self.malicious_score.shape[0] > N + 1:
                    if detection1(np.sum(self.malicious_score[-N:].numpy(), axis=0)):
                        label = detection(np.sum(self.malicious_score[-N:].numpy(), axis=0), self.m, self.k)
                    else:
                        label = np.ones(self.k)
                    self.uploaded_ids = [id for id, l in zip(self.uploaded_ids, label) if l == 1]
                    self.uploaded_updates = [update for update, l in zip(self.uploaded_updates, label) if l == 1]
                    self.uploaded_weights = [weight for weight, l in zip(self.uploaded_weights, label) if l == 1]
                    logger.info("FLDetector detect mean clients idx: {}".format(list(np.where(label==0)[0])))
            
            if epoch > 0:
                self.weight_record.append(self.weight.cpu() - self.last_weight.cpu())
            self.last_weight = self.weight
            self.old_updates = self.local_updates
        
        elif self.filter == 'maud-norm':
            selected_indices, self.maud_accumulated = maud_norm_filter(
                self.uploaded_updates, self.uploaded_ids, self.maud_accumulated, self.maud_window)
            self.uploaded_ids = [self.uploaded_ids[i] for i in selected_indices]
            self.uploaded_updates = [self.uploaded_updates[i] for i in selected_indices]
            self.uploaded_weights = [self.uploaded_weights[i] for i in selected_indices]
        elif self.filter == 'maud-cosine':
            selected_indices, self.maud_accumulated = maud_cosine_filter(
                self.uploaded_updates, self.uploaded_ids, self.maud_accumulated, self.maud_window)
            self.uploaded_ids = [self.uploaded_ids[i] for i in selected_indices]
            self.uploaded_updates = [self.uploaded_updates[i] for i in selected_indices]
            self.uploaded_weights = [self.uploaded_weights[i] for i in selected_indices]
        elif self.filter == 'flame':
            selected_indices, self.C_t = flame(self.uploaded_models, self.uploaded_updates, self.m)
            self.uploaded_ids = [self.uploaded_ids[i] for i in selected_indices]
            self.uploaded_updates = [self.uploaded_updates[i] for i in selected_indices]
            self.uploaded_weights = [self.uploaded_weights[i] for i in selected_indices]
            logger.info("Flame select clients: {}".format(selected_indices))



    def calculate_metrics(self):
        num_train_samples = 0
        total_train_loss = 0
        for client in self.clients:
            train_loss, train_num = client.train_metrics()
            num_train_samples += train_num
            total_train_loss += train_loss
        total_train_loss = total_train_loss / num_train_samples

        num_test_samples = 0
        total_test_acc = 0
        for client in self.clients:
            correct_num, test_num = client.test_metrics()
            num_test_samples += test_num
            total_test_acc += correct_num
        total_test_acc = total_test_acc / num_test_samples
        return total_train_loss, total_test_acc

    def calculate_metrics_with_asr(self):
        num_train_samples = 0
        total_train_loss = 0
        for client in self.clients:
            train_loss, train_num = client.train_metrics()
            num_train_samples += train_num
            total_train_loss += train_loss
        total_train_loss = total_train_loss / num_train_samples

        num_test_samples = 0
        total_test_acc = 0
        # ASR
        total_ls = 0
        total_ls_to_lt = 0
        for client in self.clients:
            correct_num, test_num, ls_num, ls_to_lt_num = client.test_metrics_with_asr()
            num_test_samples += test_num
            total_test_acc += correct_num
            total_ls += ls_num
            total_ls_to_lt += ls_to_lt_num
        total_test_acc = total_test_acc / num_test_samples
        asr = total_ls_to_lt / total_ls if total_ls > 0 else 0
        return total_train_loss, total_test_acc, asr   

    def save_results(self):
        result_path = self.args.sp
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        if (len(self.rs_test_acc)):
            file_path = os.path.join(self.args.sp, self.args.sn + '.h5')
            logger.info("Saving results to {}".format(file_path))

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('asr', data=self.rs_asr)
                hf.create_dataset('args', data=str(self.args))
        
        return self.rs_train_loss, self.rs_test_acc

    def evaluate(self, epoch):
        if self.dp == 'lf':
            train_loss, test_acc, asr = self.calculate_metrics_with_asr()
            self.rs_train_loss.append(train_loss)
            self.rs_test_acc.append(test_acc)
            self.rs_asr.append(asr)
            logger.info("Averaged Train Loss: {:.4f}".format(train_loss))
            logger.info("Test Accurancy: {:.4f}".format(test_acc))
            logger.info("Attack Success Rate: {:.4f}".format(asr))
        else:
            train_loss, test_acc = self.calculate_metrics()
            self.rs_train_loss.append(train_loss)
            self.rs_test_acc.append(test_acc)
            logger.info("Averaged Train Loss: {:.4f}".format(train_loss))
            logger.info("Test Accurancy: {:.4f}".format(test_acc))

        # # 结构化日志记录
        # log_entry = {
        #     "epoch": epoch,
        #     "timestamp": time.time(),
        #     "train_loss": float(train_loss),
        #     "test_acc": float(test_acc),
        # }
        
        # # 追加写入JSON文件
        # with open(self.args.log_file, "a") as f:
        #     f.write(json.dumps(log_entry) + "\n")


    def train(self):
        time_list = []

        for i in range(self.r + 1):
            start_time = time.time()
            
            # send_model
            self.send_model()

            # evaluate global model
            logger.info(f"Round number: {i} --------------------------")
            logger.info("Evaluate global model")
            # s_t = time.time()
            self.evaluate(i)
            # logger.info("evaluate time: {}s".format(time.time() - s_t))

            # select client
            self.select_clients()

            # client train model
            for client in self.selected_clients:
                client.train()
            self.receive_model()

            if self.mp == 'MPAF':
                assert self.MPAF_model is not None
                for j in range(self.m):
                    self.uploaded_models[j] = self.MPAF_model

            self.model_to_update()

            self._round_idx = i
            # poisoning 
            self.poisoning_attack()

            # defending
            self.filter_update(epoch=i)

            # aggregate
            # self.update_to_model()

            self.aggregate_model()

            time_list.append(time.time() - start_time)
            # logger.info('-'*15, 'time cost', '-'*15, time_list[-1], 's')

        logger.info("Run over")
        self.save_results()
        logger.info("Best accuracy: {}".format(max(self.rs_test_acc)))
        logger.info("Average time cost per round: {}".format(sum(time_list[1:])/len(time_list[1:])))
            


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
        self.train_loader = None
        self.test_loader = None
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)   # model必须提前准备好
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=self.ldg
        )

    def receive_model(self, new_model):
        for param, new_param in zip(self.model.parameters(), new_model.parameters()):
            param.data = new_param.data.clone()

    def flip_labels(self, data, ls, lt):
        flipped_data = []
        for x, y in data:
            if y.item() == ls:
                y = torch.tensor(lt, dtype=torch.int64)
            flipped_data.append((x, y))
        return flipped_data

    def load_data(self, is_lf=False, ls=None, lt=None):
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        # from collections import Counter
        # print(f"Client {self.id} original label distribution: {Counter([y.item() for _, y in train_data])}")
        if is_lf and (ls is not None) and (lt is not None):
            train_data = self.flip_labels(train_data, ls, lt)
        # print(f"Client {self.id} after label flipping distribution: {Counter([y.item() for _, y in train_data])}")
        self.train_loader =  DataLoader(train_data, batch_size=self.b, shuffle=True, drop_last=True)
        
        self.num_samples = len(train_data)
        
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        self.test_loader =  DataLoader(test_data, batch_size=self.b, shuffle=True, drop_last=False)

    def train(self):
        # s_t = time.time()
        # train_loader = self.load_data(is_train=True)
        # logger.info('client ', self.id, ' load data time: ', time.time() - s_t)
        # self.num_samples = len(train_loader.dataset)
        self.model.train()
        t_list = []
        for epoch in range(self.e):
            
            s_t = time.time()
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                self.optimizer.zero_grad()
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
            t_list.append(time.time() - s_t)
        # logger.info('client ', self.id, ' train time / epoch ',np.mean(t_list))
        
        if self.ld:
            self.learning_rate_scheduler.step()
    
    def train_metrics(self):
        # train_loader = self.load_data(is_train=True)
        self.model.eval()

        train_num = 0
        total_loss = 0
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                total_loss += loss.item() * y.shape[0]
        return total_loss, train_num

    def test_metrics(self):
        # test_loader = self.load_data(is_train=False)
        self.model.eval()

        test_num = 0
        correct_num = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_num += y.shape[0]
                correct_num += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return correct_num, test_num

    def test_metrics_with_asr(self):
        # test_loader = self.load_data(is_train=False)
        self.model.eval()

        test_num = 0
        correct_num = 0
        ls_num = 0
        ls_to_lt_num = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                preds = torch.argmax(output, dim=1)
                test_num += y.shape[0]
                correct_num += (torch.sum(preds == y)).item()

                ls_mask = (y == self.ls)
                ls_num += torch.sum(ls_mask).item()
                ls_to_lt_num += torch.sum(preds[ls_mask] == self.lt).item() 


        return correct_num, test_num, ls_num, ls_to_lt_num



