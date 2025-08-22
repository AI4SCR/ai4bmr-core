import numpy as np


class StatsRecorder:
    """http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html"""

    def __init__(self, data=None, exclude_zeros: bool = False):
        self.max_height = None
        self.max_width = None
        self.maxs = None
        self.mins = None
        self.mean = None
        self.std = None
        self.n_observations = 0
        self.exclude_zeros = exclude_zeros

        if data is not None:
            self.update(data)

    def update(self, data: np.ndarray):
        assert data.ndim == 3

        if self.exclude_zeros:
            data = data.copy()
            data[data == 0] = np.nan

        if self.mean is None:
            self.max_height = data.shape[-2]
            self.max_width = data.shape[-1]

            self.maxs = np.nanmax(data, axis=(1, 2))
            self.mins = np.nanmin(data, axis=(1, 2))

            self.mean = np.nanmean(data, axis=(1, 2))
            self.std = np.nanstd(data, axis=(1, 2))
            self.n_observations = data[0].size
        else:
            self.max_height = max(self.max_height, data.shape[-2])
            self.max_width = max(self.max_width, data.shape[-1])

            self.maxs = np.stack((self.maxs, np.nanmax(data, axis=(1, 2)))).max(axis=0)
            self.mins = np.stack((self.mins, np.nanmin(data, axis=(1, 2)))).min(axis=0)

            new_mean = np.nanmean(data, axis=(1, 2))
            new_std = np.nanstd(data, axis=(1, 2))

            m = self.n_observations * 1.0
            n = data[0].size

            tmp = self.mean

            self.mean = m / (m + n) * tmp + n / (m + n) * new_mean
            self.std = (
                    m / (m + n) * self.std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                        tmp - new_mean) ** 2
            )
            self.std = np.sqrt(self.std)

            self.n_observations += n
