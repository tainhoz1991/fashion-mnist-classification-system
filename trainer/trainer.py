import numpy as np
import torch
from torchvision.utils import make_grid
from base import TrainerBase
from utils import inf_loop
from logger import MetricTracker


class Trainer(TrainerBase):
    """
    Trainer class
    """

    # metric_fn_list is a list of functions used to calculate metrics such as accuracy, ...
    def __init__(self, model, loss_fn, metric_fn_list, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, batch_len=None):
        super().__init__(model, loss_fn, metric_fn_list, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if batch_len is None:
            # epoch-based training
            self.batch_len = len(self.data_loader)
        else:
            # iteration-based training
            # by this way we don't distinguish one epoch is whole dataloader
            # rather than we traverse batch list of data_loader and repeat from beginning
            # until our loop reach number of "batch_len" times
            # that means one epoch might run over whole data_loader 1 or 2 or 2,5 times
            # for example
            # === Iteration-based training with inf_loop: FIXED number of steps, independent of dataset size ===
            # data_loader = [[0,1,2,3], [4,5,6,7], [8,9,0]]
            # batch_len = 8
            # 1/8 | pass #1 | batch_dataloader 1/3 -> [0, 1, 2, 3]
            # 2/8 | pass #1 | batch_dataloader 2/3 -> [4, 5, 6, 7]
            # 3/8 | pass #1 | batch_dataloader 3/3 -> [8, 9]
            # 4/8 | pass #2 | batch_dataloader 1/3 -> [0, 1, 2, 3]
            # 5/8 | pass #2 | batch_dataloader 2/3 -> [4, 5, 6, 7]
            # 6/8 | pass #2 | batch_dataloader 3/3 -> [8, 9]
            # 7/8 | pass #3 | batch_dataloader 1/3 -> [0, 1, 2, 3]
            # 8/8 | pass #3 | batch_dataloader 2/3 -> [4, 5, 6, 7]
            self.data_loader = inf_loop(data_loader)
            self.batch_len = batch_len
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # set up tracking metrics based on list of metric functions in model.metric
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fn_list], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fn_list], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A epoch log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        # traverse data_loader list
        for batch_idx, (input, target) in enumerate(self.data_loader):
            input, target = input.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            # len_epoch = number of batches, (epoch - 1) * self.len_epoch + batch_idx = count the total of number of gradients
            self.writer.set_step((epoch - 1) * self.batch_len + batch_idx)
            self.train_metrics.update('loss', loss.item())

            # calculate metrics (accuracy) of each batch
            for met in self.metric_fn_list:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch, self._progress(batch_idx), loss.item()))
                self.writer.add_image('input', make_grid(input.cpu(), nrow=8, normalize=True))

            if batch_idx == self.batch_len:
                break

        epoch_log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            epoch_log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return epoch_log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A validation log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss_fn(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                for met in self.metric_fn_list:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.batch_len
        return base.format(current, total, 100.0 * current / total)
