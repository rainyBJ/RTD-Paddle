import paddle


class DistributedSampler(paddle.io.DistributedBatchSampler):
    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0,
                 drop_last=False):
        super().__init__(
            dataset=dataset,
            batch_size=1,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last)


def LongTensor(x):
    if isinstance(x, int):
        return paddle.to_tensor([x], dtype="int64")
    if isinstance(x, list):
        x = paddle.to_tensor(x, dtype="int64")
    return x