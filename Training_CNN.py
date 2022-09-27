import numpy as np
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/gcc")


class Training_CNN:

    def __init__(self, model, loss_fn, optimiser, dataset, batch_size, ratio, num_gpu, device, filename, num_workers):

        self.train_data_loader = None
        self.val_data_loader = None
        self.model = model
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.device = device
        self.filename = filename

        n_total = len(dataset)
        n_train = int(n_total * ratio)
        n_val = n_total - n_train
        train_data, val_data = torch.utils.data.random_split(dataset=dataset, lengths=[n_train, n_val])

        self.train_data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        self.val_data_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    def train(self, epochs):

        loss_list = []
        loss_val_list = []

        list_snr = []
        list_rt_60 = []
        list_signal_type = []
        list_ir_type = []

        for i in range(epochs):
            print(f"Epoch {i + 1}/{epochs}")

            # Train
            loss, rt_60, snr, signal_type, ir_type = self.train_single_epoch()
            loss_list.append(loss)

            list_rt_60.extend(rt_60)
            list_snr.extend(snr)
            list_signal_type.extend(signal_type)
            list_ir_type.extend(ir_type)

            # Validate
            loss_val = self.validate_single_epoch()
            loss_val_list.append(loss_val)

            torch.save(self.model.state_dict(), self.filename)

            writer.add_scalars(
                main_tag='loss',
                tag_scalar_dict={'training': loss, 'validation': loss_val},
                global_step=i)
            writer.close()

            print("---------------------------")
        print("Finished training")

        return loss_list, loss_val_list, list_rt_60, list_snr, list_signal_type, list_ir_type

    def train_single_epoch(self):

        losses = []
        list_rt_60 = []
        list_snr = []
        list_signal_type = []
        list_ir_type = []

        for bulk_sample, bulk_target, parameters in self.train_data_loader:

            sample, label = bulk_sample.to(self.device), bulk_target.to(self.device)

            # forward pass
            prediction = self.model(sample)

            # calculate loss
            loss = self.loss_fn(prediction, torch.squeeze(label, dim=-1))

            losses.append(loss.item())

            # backpropagate error and update weights
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        loss = np.mean(np.array(losses))
        print(f"loss: {loss.item()}")
        return loss.item(), list_rt_60, list_snr, list_signal_type, list_ir_type

    def validate_single_epoch(self):

        losses = []
        for bulk_sample, bulk_target, parameters in self.val_data_loader:

            sample, label = bulk_sample.to(self.device), bulk_target.to(self.device)

            # forward pass
            prediction = self.model(sample)

            # calculate loss
            loss = self.loss_fn(prediction, torch.squeeze(label, dim=-1))
            losses.append(loss.item())

            # backpropagate error and update weights
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        print(f"val:  {loss.item()}")
        return loss.item()
