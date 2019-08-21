import numpy as np
import torch
import time
import torch.nn as nn
from torch.utils import data

from MyDataset import MyDataset
from MyTestDataset import MyTestDataset


class Flexible_MLP:
    #hidden_sizes has two elements per layer: an input size and an output size.
    def __init__(self, one_sided_context_size, hidden_sizes, activations, output_size, load_and_test, model_name=None):
        self.load_and_test = load_and_test
        self.model_name = model_name
        if self.load_and_test == True:
            self.net = torch.load('/home/ubuntu/hw1 Speech Processing/' + repr(self.model_name) + '.pt')
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.one_sided_context_size = one_sided_context_size

            #Just bumping up input layer size for context
            # hidden_sizes = list(hidden_sizes)
            # hidden_sizes[0] = hidden_sizes[0]*(2*one_sided_context_size + 1)
            #Bumping up all layer sizes for context
            # hidden_sizes = ([hidden_size * (2*one_sided_context_size + 1) for hidden_size in hidden_sizes])
            scaled_hidden_sizes = []
            factor = (2*one_sided_context_size + 1)
            # reduce = False
            # for i, hidden_size in enumerate(hidden_sizes):
            #     if reduce is False and i != 0 and hidden_size < hidden_sizes[i-1]:
            #         reduce = True
            #     if reduce:
            #         factor = max(factor - 2, 0)
            #     scaled_hidden_sizes.append(hidden_size * factor)
            # hidden_sizes = scaled_hidden_sizes
            hidden_sizes = np.array(hidden_sizes)
            hidden_sizes[0] = hidden_sizes[0] * factor

            hidden_layers = []
            for i in range(len(hidden_sizes)-1):
                hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                # Batchnorm layer here
                hidden_layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
                if activations[i] == 0:
                    hidden_layers.append(nn.Sigmoid())
                elif activations[i] == 1:
                    hidden_layers.append(nn.Tanh())
                elif activations[i] == 2:
                    hidden_layers.append(nn.ReLU())
                else:
                    hidden_layers.append(nn.Sigmoid())
                #Dropout layer here; p between 0.1 and 0.5
                # hidden_layers.append(nn.Dropout(p=0.1))

            #don't multiply output size by (2*context+1) because we are only predicting the middle phoneme.
            hidden_layers.append(nn.Linear(hidden_sizes[-1], output_size))
            self.net = nn.Sequential(*hidden_layers)
            self.net.apply(init_weights)

            # between 32 and 256, usually; 100 is good
            self.batch_size = 100
            # 4 * num_gpus; num_gpus = 16 for p2.xlarge
            self.num_workers = 8
            self.train_data_params = {'batch_size': self.batch_size,
                      'shuffle': True,
                      'num_workers': self.num_workers,
                      'pin_memory': True}
            self.val_data_params = {'batch_size': self.batch_size,
                      'shuffle': False,
                      'num_workers': self.num_workers,
                        'pin_memory': True}
            self.test_data_params = {'batch_size': self.batch_size,
                      'shuffle': False,
                      'num_workers': self.num_workers,
                     'pin_memory': True}


    def train(self, train, val, iters, gpu, lr):
        device = torch.device('cuda' if gpu else 'cpu')
        self.net = self.net.to(device)
        # train_utt_lengths = torch.Tensor(list(map(len, train[0])))
        # train_utt_tensor = Variable(torch.zeros((len(train[0], train_utt_lengths.max())))).float()
        # for index, (utt, utt_len) in enumerate(zip(train[0], train_utt_lengths)):
        #     train_utt_tensor[index, :utt_len] = torch.Tensor(utt).float()
        # train_utt_lengths, perm_index = train_utt_lengths.sort(0, descending=True)
        # train_utt_tensor = train_utt_tensor[perm_index]

        print('Creating the training generator.')
        training_gen_start_time = time.time()
        training_generator = data.DataLoader(MyDataset(train, self.one_sided_context_size), **self.train_data_params)
        print('Creating the training generator took ' + repr(time.time() - training_gen_start_time) + ' seconds.')
        # print('Length of training generator is ' + repr(sum(1 for _ in training_generator)))
        # training_generator = data.DataLoader(MyDataset(train), **params, collate_fn=MyCollateFn(dim=0))
        print('Creating the validation generator.')
        validation_generator = data.DataLoader(MyDataset(val, self.one_sided_context_size), **self.val_data_params)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-5)
        # optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

        print('Beginning training.')
        for epoch in range(iters):
            print('Device is ' + repr(device))
            if torch.cuda.is_available():
                print('Pytorch confirms that we are using the gpu')
            start = time.time()
            # Training
            count = 0
            # grab_batch_start_time = time.time()
            for local_batch, local_labels in training_generator:
                # grab_batch_end_time = time.time()
                self.net.train()
                # if(count % 100 == 0):
                #     print('Grabbing a batch took ' + repr((grab_batch_end_time - grab_batch_start_time)) + ' seconds.')
                #     print('Training on batch ' + repr(count) + ' for epoch ' + repr(epoch))
                if(count % 50000 == 0):
                    print('So far, training on ' + repr(count) + ' batches has taken ' + repr((time.time()-start)/60) + ' minutes.')
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                # local_labels = local_labels.narrow(1, self.one_sided_context_size, 1).long().view(-1)
                local_labels = local_labels.long().view(-1)
                # local_batch, local_labels = torch.from_numpy(local_batch).float().to(device), torch.from_numpy(local_labels).long().to(device)
                # forward_start_time = time.time()
                local_output = self.net(local_batch)
                #predicting for just the center frame
                local_loss = self.criterion(local_output.float(), local_labels)
                # forward_end_time = time.time()
                # if(count % 100 == 0):
                #     print('Forward pass took ' + repr((forward_end_time - forward_start_time)) + ' seconds.')
                #predicting for every frame in context
                # local_loss = 0
                # for c in range(2*self.one_sided_context_size + 1):
                #     local_loss += self.criterion(local_output.narrow(1, c*138, 138), local_labels.narrow(1, c, 1))
                # predicting with no context frames
                # local_loss = self.criterion(local_output, local_labels)

                # backward_start_time = time.time()
                local_loss.backward()
                # backward_end_time = time.time()
                # if(count % 100 == 0):
                #     print('Backward pass took ' + repr((backward_end_time - backward_start_time)) + ' seconds.')
                #Use the below line to control for exploding gradients
                # step_start_time = time.time()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.25)
                optimizer.step()
                optimizer.zero_grad()
                # step_end_time = time.time()
                # if(count % 100 == 0):
                #     print('Step took ' + repr(step_end_time - step_start_time) + ' seconds.')
                count+=1
                # grab_batch_start_time = time.time()

            if epoch % 1 == 0:
                print("After epoch ", repr(epoch))
                # compute the accuracy of the prediction
                local_output = local_output.cpu().detach()
                # train_prediction = local_output.argmax(axis=1)
                train_prediction = torch.argmax(local_output, dim=1)
                train_accuracy = (train_prediction.numpy() == local_labels.cpu().numpy()).mean()

                self.net.eval()
                with torch.set_grad_enabled(False):
                    for val_local_batch, val_local_labels in validation_generator:
                        val_local_batch, val_local_labels = val_local_batch.to(device), val_local_labels.to(device)
                        val_local_labels = val_local_labels.long().view(-1)
                        val_local_output = self.net(val_local_batch)
                        val_local_loss = 0
                        # for c in range(2*self.one_sided_context_size + 1):
                        #     val_local_loss += self.criterion(val_local_output.narrow(1, c*138, (c+1)*138), val_local_labels.narrow(1, c, 1))
                        val_local_loss = self.criterion(val_local_output, val_local_labels)

                # val_local_output = np.split(val_local_output.cpu().detach(), (2*self.one_sided_context_size+1))
                val_local_output = val_local_output.cpu().detach()
                val_prediction = torch.argmax(val_local_output, dim=1)
                val_accuracy = (val_prediction.numpy() == val_local_labels.cpu().numpy()).mean()
                print("Training loss :", local_loss.cpu().detach().numpy())
                print("Training accuracy :", train_accuracy)
                print("Validation loss :", val_local_loss.cpu().detach().numpy())
                print("Validation accuracy :", val_accuracy)
                stop = time.time()
                print("This epoch took " + repr((stop-start)/60) + " minutes.")
                backup_file = 'model_' + repr(epoch) + '.pt'
                torch.save(self.net.state_dict(), backup_file)
                scheduler.step(val_local_loss)
        self.net = self.net.cpu()
        print('Finished training.')

    def test(self, test, gpu):
        out_file = open("output.csv", "w+")
        out_file.write("id,label\n")
        out_file.close()
        device = torch.device("cuda:0" if gpu else "cpu")
        self.net = self.net.to(device)
        self.net.eval()
        params = {'batch_size': 100,
                  'shuffle': False,
                  'num_workers': 6}
        print('Creating the testing generator.')
        test_generator = data.DataLoader(MyTestDataset(test, self.one_sided_context_size), **params)
        index = 0
        for local_batch in test_generator:
            local_batch = local_batch.to(device)
            local_output = self.net(local_batch)
            local_output = local_output.cpu().detach()
            predictions = torch.argmax(local_output, dim=1)
            out_file = open("output.csv", "a+")
            for prediction in predictions:
                out = repr(index) + "," + repr(prediction.item()) + "\n"
                out_file.write(out)
                index += 1
            out_file.close()
        self.net = self.net.cpu()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    # if type(m) == nn.ReLU:
    #     torch.nn.init.kaiming_normal(m.weight, nonlinearity='relu')
    #     m.bias.fill.data.fill_(0.01)