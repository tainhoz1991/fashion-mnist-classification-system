import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorBoardWriter
import mlflow
import os
from datetime import datetime
from torchinfo import summary
from mlflow.models import infer_signature
import numpy as np

class TrainerBase:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss_fn, metric_fn_list, optimizer, config):
        self.config = config

        # set up mlflow tracking server
        mlflow_tracker_server_url = os.getenv('MLFLOW_TRACKING_URI')
        mlflow.set_tracking_uri(mlflow_tracker_server_url)

        # initiate a logger name "trainer"
        self.logger = config.get_logger('trainer', config['trainer']['log_level'])

        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn_list = metric_fn_list
        self.optimizer = optimizer

        # Create sample input and predictions
        sample_input = np.random.uniform(size=[1, 1, 28, 28]).astype(np.float32)

        # Get model output - convert tensor to numpy
        with torch.no_grad():
            output = model(torch.tensor(sample_input))
            sample_output = output.numpy()

        # Infer signature automatically
        self.signature = infer_signature(sample_input, sample_output)

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            # inf = positive infinit (+~)
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorBoardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
        username = os.getlogin()
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H:%M:%S")
        runner = f"{username}---{formatted_datetime}"

        with mlflow.start_run(run_name=runner) as run:
            # Log model summary.
            summary_model_path = f"{self.config.save_dir}/model_summary.txt"
            with open(summary_model_path, "w", encoding="utf-8") as f:
                f.write(str(summary(self.model)))
            mlflow.log_artifact(summary_model_path)

            params = {
                "epochs": self.epochs,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "batch_size": self.config['data_loader']['args']['batch_size'],
                "loss_function": self.loss_fn.__name__,
                "metric_function": [metric_fn.__name__ for metric_fn in self.metric_fn_list],
                "optimizer": self.config["optimizer"]["type"],
            }
            # Log training parameters.
            mlflow.log_params(params)

            not_improved_count = 0
            for epoch in range(self.start_epoch, self.epochs + 1):
                result = self._train_epoch(epoch)

                # save logged information into log dict
                log = {'epoch': epoch}
                log.update(result)

                # print logged information to the screen
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                best = False
                if self.mnt_mode != 'off':
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        self.logger.warning("Warning: Metric '{}' is not found. "
                                            "Model performance monitoring is disabled.".format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                         "Training stops.".format(self.early_stop))
                        break

                if epoch % self.save_period == 0:
                    self._save_checkpoint(run, mlflow, log, epoch, save_best=best)

        ranked_checkpoints = mlflow.search_logged_models(
            filter_string=f"source_run_id='{run.info.run_id}'",
            order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
            output_format="list",
        )
        best_checkpoint = ranked_checkpoints[0]
        self.logger.info("Best checkpoint: {}".format(best_checkpoint))

    def _save_checkpoint(self, run, mlflow, log, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model_arch = type(self.model).__name__
        state = {
            'model_arch': model_arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        # save model to mlflow tracking server
        logged_model = mlflow.pytorch.log_model(pytorch_model=self.model, name=f"model-epoch-{epoch}",
                                                signature=self.signature)
        mlflow.log_metric(key="loss", value=f"{log["loss"]:2f}",
                          step=epoch, model_id=logged_model.model_id)
        mlflow.log_metric(key="accuracy", value=f"{log["accuracy"]:2f}",
                          step=epoch, model_id=logged_model.model_id)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            # save best model to mlflow tracking server
            logged_model = mlflow.pytorch.log_model(pytorch_model=self.model, name="model-best",
                                                    signature=self.signature)
            mlflow.log_metric(key="loss", value=f"{log["loss"]:2f}", step=epoch, model_id=logged_model.model_id)
            mlflow.log_metric(key="accuracy", value=f"{log["accuracy"]:2f}",
                              step=epoch, model_id=logged_model.model_id)
            # register model
            model_uri = f"runs:/{run.info.run_id}/model-best"
            registered_model = mlflow.register_model(model_uri=model_uri, name="FashionMNISTModel")
            self.logger.info(f"Registered model name {registered_model.name} version {registered_model.version} ...")

            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model_arch'] != self.config['model_arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
