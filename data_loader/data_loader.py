from torchvision import datasets, transforms
from base import DataLoaderBase
from utils import calculate_mean_std


class FashionMNISTDataLoader(DataLoaderBase):
    """
        FashionMNIST data loading demo using BaseDataLoader
        """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 config=None):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])
        self.data_dir = data_dir
        self.dataset = datasets.FashionMNIST(self.data_dir, train=training, download=True,
                                             transform=transforms.ToTensor())
        if training:
            mean, std = calculate_mean_std(self.dataset)
            normalized_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist())
            ])
            self.dataset.transform = normalized_tf
        else:
            normalized_tf = _init_transform(config)
            self.dataset.transform = normalized_tf

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def _init_transform(config_section):
    # config_section is something like config['transform']
    tf_list = []
    for tf_conf in config_section['args']['transforms']:
        tf_type = tf_conf['type']
        tf_args = tf_conf.get('args', {})
        tf_list.append(getattr(transforms, tf_type)(**tf_args))
    return getattr(transforms, config_section['type'])(tf_list)
