from ... import DataRegister, DataLoader, DataSaver, DatasetGenerator


class Token:
    """base google vocab token"""
    cls = '[CLS]'
    sep = '[SEP]'
    pad = '[PAD]'
    unk = '[UNK]'
    mask = '[MASK]'
    unused = '[unused%d]'
