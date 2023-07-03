import pandas as pd
import numpy as np


class Zscore:
    def __init__(self, sample_percentage=None, channel_dim=None, num_channels=1, replace=False):
        self.sample_percentage = sample_percentage
        self.channel_dim = channel_dim
        self.num_channels = num_channels
        self.replace = replace
        if self.channel_dim is not None:
            self.samples = pd.DataFrame(columns=["channel{}".format(channel) for channel in range(num_channels)])
        else:
            self.samples = pd.DataFrame(columns=["channel0"])

    def sample(self, data):
        new_row = {}
        if self.channel_dim is not None:
            for channel in range(self.num_channels):
                samples = self._choose_samples(np.moveaxis(data, self.channel_dim, 0)[channel, ...])
                new_row["channel{}".format(channel)] = samples
        else:
            samples = self._choose_samples(data)
            new_row["channel0"] = samples
        self.samples = pd.concat([self.samples, pd.DataFrame([new_row])], ignore_index=True)

    def get_zscore(self):
        mean, std = [], []
        for channel in range(self.num_channels):
            channel_mean = np.asarray(self.samples["channel{}".format(channel)].to_list()).flatten().mean()
            channel_std = np.asarray(self.samples["channel{}".format(channel)].to_list()).flatten().std()
            mean.append(channel_mean)
            std.append(channel_std)
        if self.num_channels == 1:
            mean = mean[0]
            std = std[0]
        return {"mean": mean, "std": std}

    def _choose_samples(self, data):
        if self.sample_percentage is not None:
            num_samples = int(data.size * self.sample_percentage)
            data_samples = np.random.choice(data.flatten(), num_samples, replace=self.replace)
        else:
            data_samples = data.flatten()
        return data_samples
    

if __name__ == "__main__":
    normalizer = Zscore(sample_percentage=0.1)

    for _ in range(10):
        data = np.random.random((100, 100))
        normalizer.sample(data)

    print(normalizer.get_zscore())