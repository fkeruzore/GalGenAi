from datasets import Dataset, load_dataset, load_from_disk
from galgenai.data.hsc import get_dataset_and_loaders

nx = 64


def test_data():
    try:
        dataset_raw = load_from_disk("./data/hsc_mmu_mini")
    except FileNotFoundError:
        _iter = load_dataset(
            "MultimodalUniverse/hsc", split="train", streaming=True
        )
        dataset_raw = Dataset.from_list(list(_iter.take(128)))
    dataset, train_loader, test_loader = get_dataset_and_loaders(
        dataset_raw, nx=nx
    )

    shape = dataset[0][0].shape
    assert shape[0] == 5
    assert shape[1] == nx
    assert shape[2] == nx


if __name__ == "__main__":
    test_data()
