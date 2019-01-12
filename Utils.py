import torch

def getDevice():
    '''Get devices available on system. Will choose GPU over cpu
    
    :return: GPU if GPU available, else CPU
    :rtype: String
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device