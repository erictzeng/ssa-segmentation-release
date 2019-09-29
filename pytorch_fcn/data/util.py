import torch


class Dispenser:
    def __init__(self, loader):
        self.loader = loader
        self.loader_iter = iter(loader)

    def next_batch(self):
        try:
            return next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            return next(self.loader_iter)


class JointDispenser:
    def __init__(self, *loaders):
        self.dispensers = [Dispenser(loader) for loader in loaders]

    def next_batch(self):
        batches = []
        for dispenser in self.dispensers:
            batch = dispenser.next_batch()
            batches.append(batch)
        zipped = zip(*batches)
        concatenated = [torch.cat(item, 0) for item in zipped]
        if len(concatenated) == 1:
            return concatenated[0]
        else:
            return tuple(concatenated)
