from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from torch.nn.functional import cross_entropy
import torch, wget, os, gzip, pickle, random, tarfile
import numpy as np
from torchvision import datasets
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
def load_CIFAR10(normal_data_root:str, 
                 corrupt_data_path:str, 
                 transformations: Callable,
                 corrupt_types:Union[Tuple[str], List[str]]=["gaussian_noise"],
                 download_normal:bool=False,
                 device = 'cpu'
                 ):
    """ 
    Downloads or loads normal CIFAR10. Extracts corrupted CIFAR-10 into a folder named after the .tar file 
        (without extension). Returns transformed normal and corrupted CIFAR10 in hashmaps.
    Args:
        normal_data_root (str): Path where normal CIFAR10 is saved or will be saved.
        corrupt_data_path (str): Path of the .tar file storing corrupted CIFAR10.
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
    # load corrupted .tar data file in a directory
    with tarfile.open(corrupt_data_path, "r") as tar:
        tar.extractall()
    
    # get foldername where corrupt data is stored from .tar file.
    corrupt_foldername = os.path.splitext(os.path.basename(corrupt_data_path))[0]
    
    corrupt_data = {"labels":np.load(f"{corrupt_foldername}/labels.npy")}
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device).view(1,3,1,1)
    std = torch.tensor([0.247, 0.243, 0.261]).to(device).view(1,3,1,1)
    for corrupt_type in corrupt_types:
        imgs = np.load(f"{corrupt_foldername}/{corrupt_type}.npy")  # get imgs from folder
        imgs = torch.from_numpy(imgs.astype('float32') / 255.).to(device)  # move to GPU
        corrupt_data[corrupt_type] = (imgs.permute(0, 3, 1, 2) - mean) / std     # to [N, C, H, W]
    
    return normal_data, corrupt_data

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

# ------------ training function ------------
def train_model(train_batches, model, optimizer, learning_rate=0.01, epochs=5, print_metrics=False):
    optimizer.param_groups[0]['lr'] = learning_rate

    # loop over epochs
    accuracies = []
    for epoch in range(1,epochs+1):
        if print_metrics:
            print(f"------- Epoch {epoch} -------")
        train_loss, train_correct_predictions, train_samples = 0, 0, 0

        # Training loop for the model
        model.train()
        for x_batch, y_batch in train_batches:  # x_shape=(batch_size, words), y_shape=(sentences)

            # make predictions
            predictions = model(x_batch)    # shape=(batch_size, num_classes)

            # compute loss
            loss = cross_entropy(predictions, y_batch)    # shape=integer -- avg loss of all sentences in batch

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate metrics
            predicted_classes = torch.argmax(predictions, dim=1)
            train_correct_predictions += (predicted_classes == y_batch).sum().item()
            train_samples += y_batch.size(0)
            train_loss += loss.item()

        if print_metrics:
            avg_loss = train_loss/len(train_batches)
            train_accuracy = train_correct_predictions / train_samples
            print(f"Train-Loss: {avg_loss:.4f}, Train-Accuracy:{train_accuracy:.2f}")
            accuracies.append(train_accuracy)
    print()
    
    # save trained model
    save_path = "sentiment_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return accuracies


# ------------ testing function ------------
def test_model(val_batches, model, print_metrics=False):
    # Evaluation loop for the model on validation set
    model.eval()
    val_loss, val_correct_predictions, val_samples = 0, 0, 0
    with torch.no_grad():           # Disable gradient computation during evaluation
        for x_batch, y_batch in val_batches:
            predictions = model(x_batch)
            loss = cross_entropy(predictions, y_batch)

            #calculate metrics
            predicted_classes = torch.argmax(predictions, dim=1)
            val_correct_predictions += (predicted_classes == y_batch).sum().item()
            val_samples += y_batch.size(0)
            val_loss += loss.item()

    avg_loss = val_loss/len(val_batches)
    val_accuracy = val_correct_predictions / val_samples
    if print_metrics:
        print(f"Val-Loss: {avg_loss:.4f}, Val-Accuracy:{val_accuracy:.2f}")
    return val_accuracy


