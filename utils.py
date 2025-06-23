from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
import torch.nn.functional as F
import torch, wget, os, gzip, pickle, random, tarfile
import numpy as np
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
from typing import Union, Tuple, List, Callable

# ------------ load imdb data ------------
def load_imdb(final=False, val=5000, seed=0, voc=None, char=False):
    """
    Returns:
        :x_train (List[List[int]]): Integer-encoded training sentences.
        :x_val (List[List[int]]): Integer-encoded validation sentences.
        :y_train (List[int]): Labels for training sentences.
        :y_val (List[int]): Labels for validation sentences.
        :i2w (Dict[int, str]): Mapping from integer to word.
        :w2i (Dict[str, int]): Mapping from word to integer.
        :numcls (int): Number of distinct labels/classes.
    """
    IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
    IMDB_FILE = 'imdb.{}.pkl.gz'

    cst = 'char' if char else 'word'

    imdb_url = IMDB_URL.format(cst)
    imdb_file = IMDB_FILE.format(cst)

    if not os.path.exists(imdb_file):
        wget.download(imdb_url)

    with gzip.open(imdb_file) as file:
        sequences, labels, i2w, w2i = pickle.load(file)

    if voc is not None and voc < len(i2w):
        nw_sequences = {}

        i2w = i2w[:voc]
        w2i = {w: i for i, w in enumerate(i2w)}

        mx, unk = voc, w2i['.unk']
        for key, seqs in sequences.items():
            nw_sequences[key] = []
            for seq in seqs:
                seq = [s if s < mx else unk for s in seq]
                nw_sequences[key].append(seq)

        sequences = nw_sequences

    if final:
        return (sequences['train'], labels['train']), (sequences['test'], labels['test']), (i2w, w2i), 2

    # Make a validation split
    random.seed(seed)

    x_train, y_train = [], []
    x_val, y_val = [], []

    val_ind = set( random.sample(range(len(sequences['train'])), k=val) )
    for i, (s, l) in enumerate(zip(sequences['train'], labels['train'])):
        if i in val_ind:
            x_val.append(s)
            y_val.append(l)
        else:
            x_train.append(s)
            y_train.append(l)

    return (x_train, y_train), \
           (x_val, y_val), \
           (i2w, w2i), 2


# ------------ load wikipedia data ------------
def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load the enwik8 dataset from the Hutter challenge.

    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py

    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return Tuple[Tensor[int]]: Tuple of tensors containing integer-encoded charachters of the english alphabet. The tensors are:
        - Train data
        - Test data
        - Validation data
    """
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

# ------------ create batches for sentences ------------
def create_sentence_batches(sequences, labels, batch_size=32, pad_token=0, device='cpu'):
    """
    sequences: List[List[int]] -> integer encoded sentences
    labels: List[int] -> labels for each sentence
    Returns: List[Tuple[Tensor, Tensor]] -> List of batches. Each batch is a tuple of sentences and its labels.
    """
    # sort data and labels by length
    sorted_data: List[Tuple[list,int]] = sorted(list(zip(sequences,labels)),key=lambda x: len(x[0]), reverse=True)

    batches = []
    for i in range(0, len(sorted_data), batch_size):
        batch: List[tuple] = sorted_data[i : i + batch_size]
        # seperate sentences from labels
        seqs, lbls = zip(*batch)
        # pad sequences
        seqs = pad_sequence([torch.tensor(s, dtype=torch.long) for s in seqs], batch_first=True, padding_value=pad_token)
        # convert to tensors and append to final list
        batches.append((seqs.to(device), torch.tensor(lbls, dtype=torch.long, device=device)))

    return batches

# ------------ load CIFAR10 normal and corrupted data ------------
def load_CIFAR10(
        normal_data_root:str, 
        corrupt_data_path:str, 
        transformations: Callable,
        corrupt_types:Union[Tuple[str], List[str]]=["gaussian_noise"],
        download_normal:bool=False,
        device = 'cpu'
        ):
    """ 
    Returns transformed normal and corrupted CIFAR10 in hashmaps.
    Args:
        normal_data_root (str): Path where normal CIFAR10 is saved or will be saved.
        corrupt_data_path (str): Path of the folder containing .npy files of corrupted 
            CIFAR10 (First extract .tar file of corrupted CIFAR10).
        corrupt_types (tuple or list, optional): Sequence of corruption types to load from corrupted CIFAR-10 
            (matching the .npy filenames). Defaults to ["gaussian_noise"]. Labels are always returned.
        transformations (callable): transformations to apply on normal CIFAR10.
        download_normal (bool, optional): If True, download normal CIFAR10 at normal_data_root,
            else load from disk. Defaults to False.
        device: cpu or cuda
    Returns:
        tuple: Two dicts containing transformed normal and corrupted CIFAR10 datasets.
    """
    normal_data = {
        "train" : datasets.CIFAR10(root=normal_data_root, train=True, download=download_normal, transform=transformations),
        "test" : datasets.CIFAR10(root=normal_data_root, train=False, download=download_normal, transform=transformations)
    }
    
    # get corrupt data.
    corrupt_data = {"labels":np.load(f"{corrupt_data_path}/labels.npy")}
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device).view(1,3,1,1)
    std = torch.tensor([0.247, 0.243, 0.261]).to(device).view(1,3,1,1)
    for corrupt_type in corrupt_types:
        imgs = np.load(f"{corrupt_data_path}/{corrupt_type}.npy")  # get imgs from folder
        imgs = torch.from_numpy(imgs.astype('float32') / 255.).to(device)  # move to GPU
        corrupt_data[corrupt_type] = (imgs.permute(0, 3, 1, 2) - mean) / std     # to [N, C, H, W]
    
    return normal_data, corrupt_data


# load shuffled tiny imagenet
def load_experimental_TinyImageNet(
        normal_data_path: str,
        corrupt_data_path: str, 
        corrupt_types=["motion_blur"],
        num_train_imgs=40,
        num_test_imgs=4,
    ):
    """ 
    Returns transformed normal and corrupted Tiny ImageNet mixed and shuffled together. 
    Preferably save the data after execution.
    Args:
        normal_data_path (str): Path where normal Tiny ImageNet is saved.
        corrupt_data_path (str): Path of the directory containing folders of Tiny ImageNet corruptions.
        corrupt_types (tuple or list): Sequence of corruption types to load from corrupted Tiny ImageNet
            (matching to the folder names). Defaults to ["motion_blur"].
        device: cpu or cuda. Defaults to "cpu".
    Returns:
        train and test experiment data. Each is a tuple of tensors: (images, labels, corruption type)
    """
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    class_names = sorted(os.listdir(os.path.join(normal_data_path, "train")))
    label_map = {cls: i for i, cls in enumerate(class_names)}
    corruption_map = {c: i for i, c in enumerate(corrupt_types)}
    corruption_map["normal"] = len(corrupt_types)

    train_imgs, test_imgs = [], []
    train_labels, test_labels = [], []
    train_corrupts, test_corrupts = [], []

    # collect corrupted data
    if len(corrupt_types) != 0:
        for corrupt_type in corrupt_types:
            for severity in range(1, 6):
                path = os.path.join(corrupt_data_path, corrupt_type, str(severity))
                dataset = datasets.ImageFolder(root=path, transform=transform)
                loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

                # Load all data in one go
                imgs, labels = next(iter(loader))

                # Split train/test
                imgs_per_class = {i: [] for i in range(len(class_names))}
                for img, label in zip(imgs, labels):
                    class_name = dataset.classes[label]
                    true_label = label_map[class_name]
                    imgs_per_class[true_label].append(img)

                for true_label in imgs_per_class:
                    imgs_list = imgs_per_class[true_label]
                    train_imgs.extend(imgs_list[:num_train_imgs])
                    test_imgs.extend(imgs_list[num_train_imgs:num_train_imgs+num_test_imgs])

                    train_labels.append(torch.full((num_train_imgs,), true_label))
                    test_labels.append(torch.full((num_test_imgs,), true_label))

                    train_corrupts.append(torch.full((num_train_imgs,), corruption_map[corrupt_type]))
                    test_corrupts.append(torch.full((num_test_imgs,), corruption_map[corrupt_type]))

    # collect normal data
    for cls in class_names:
        img_path = os.path.join(normal_data_path, "train", cls, "images")
        img_files = os.listdir(img_path)
        train_imgs.extend([
            transform(Image.open(os.path.join(img_path, f)).convert("RGB")) for f in img_files[:num_train_imgs*5]
            ])
        test_imgs.extend([
            transform(Image.open(os.path.join(img_path, f)).convert("RGB")) for f in img_files[num_train_imgs*5:(num_train_imgs*5)+(num_test_imgs*5)]
            ])
        
        train_labels.append(torch.full((num_train_imgs*5,), label_map[cls]))
        test_labels.append(torch.full((num_test_imgs*5,), label_map[cls]))

    train_corrupts.append(torch.full((num_train_imgs*5*len(class_names),), corruption_map["normal"]))
    test_corrupts.append(torch.full((num_test_imgs*5*len(class_names),), corruption_map["normal"]))

    # organize data
    train_data = (torch.stack(train_imgs), torch.cat(train_labels), torch.cat(train_corrupts))
    test_data = (torch.stack(test_imgs), torch.cat(test_labels), torch.cat(test_corrupts))

    # shuffling
    perm = torch.randperm(train_data[0].size(0))
    train_data = (train_data[0][perm],train_data[1][perm],train_data[2][perm])
    perm = torch.randperm(test_data[0].size(0))
    test_data = (test_data[0][perm],test_data[1][perm],test_data[2][perm])

    return train_data, test_data

# ------------ from imgs to patches ------------
def img_to_patches(img, patch_size):
    """Returns image as a tensor of its flattened patches"""
    C, H, W = img.shape
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, C * patch_size * patch_size)  # flatten patches
    return patches  # (n, c * p^2)

def batch_to_patches(batch, patch_size):
    """Returns batch of images, where each image is a tensor of its flattened patches"""
    B, C, H, W = batch.shape
    patches = batch.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * patch_size * patch_size)
    return patches  # (b, n, c * p^2)

# ------------ testing function ------------
def test_model(test_batches, model, device='cpu', print_metrics=False):
    model.eval()
    total_loss, correct_predictions, total_samples = 0, 0, 0
    with torch.no_grad():           # Disable gradient computation during evaluation
        
        for x_batch, y_batch in test_batches:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            predictions = model(x_batch)
            loss = F.cross_entropy(predictions, y_batch)

            #calculate metrics
            predicted_classes = torch.argmax(predictions, dim=1)
            correct_predictions += (predicted_classes == y_batch).sum().item()
            total_samples += y_batch.size(0)
            total_loss += loss.item() * y_batch.size(0)

    avg_loss = total_loss/total_samples
    avg_accuracy = correct_predictions / total_samples
    if print_metrics:
        print(f"Val-Loss: {avg_loss:.4f}, Val-Accuracy:{avg_accuracy:.2f}")
    return avg_loss, avg_accuracy

# ------------ training function ------------
def train_model(
        train_batches, model, optimizer, 
        test_batches=None, num_epochs=5, device='cpu',
        learning_rate=0.01, save_path="vit_saved.pth", print_metrics=False
        ):
    train_accs, train_losses, test_accs, test_losses = [], [], [], []
    for epoch in range(1, num_epochs+1):
        print(f"------- Epoch {epoch} -------")
        total_loss, correct_predictions, total_samples = 0, 0, 0

        model.train()
        for x_batch, y_batch in train_batches:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            predictions = model(x_batch)                    # (B, num_classes)
            loss = F.cross_entropy(predictions, y_batch)    # avg loss over batch

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate metrics
            predicted_classes = torch.argmax(predictions, dim=1)
            correct_predictions += (predicted_classes == y_batch).sum().item()
            total_samples += y_batch.size(0)
            total_loss += loss.item() * y_batch.size(0)

        avg_loss = total_loss/total_samples
        avg_accuracy = correct_predictions/total_samples
        train_accs.append(avg_accuracy)
        train_losses.append(avg_loss)

        if print_metrics:
            print(f"Train-Loss: {avg_loss:.3f}, Train-Accuracy:{avg_accuracy:.2f}")
        if test_batches is not None:
            test_loss, test_acc = test_model(test_batches, model, device=device, print_metrics=print_metrics)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
    
    # save trained model
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    if test_batches is not None:
        return train_losses, train_accs, test_losses, test_accs
    return train_losses, train_accs


