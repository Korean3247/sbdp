from torch.utils.data import Dataset, DataLoader, Subset


class IndexedDataset(Dataset):
    """Wraps a dataset to return (data, label, sample_id).

    For image datasets: returns (image, label, idx)
    For text datasets: returns ({"input_ids": ..., "attention_mask": ...}, label, idx)
    """

    def __init__(self, dataset: Dataset, is_text: bool = False):
        self.dataset = dataset
        self.is_text = is_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.is_text:
            input_ids, attention_mask, label = self.dataset[idx]
            data = {"input_ids": input_ids, "attention_mask": attention_mask}
            return data, label, idx
        else:
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
