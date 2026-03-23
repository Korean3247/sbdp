from torch.utils.data import Dataset, DataLoader, Subset


class IndexedDataset(Dataset):
    """Wraps a dataset to return (image, label, sample_id)."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, idx


def make_subset_loader(
    dataset: IndexedDataset,
    selected_ids: list[int] | None,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    if selected_ids is not None:
        subset = Subset(dataset, selected_ids)
    else:
        subset = dataset
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
