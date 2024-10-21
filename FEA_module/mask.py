import torch

def get_mask_tf(sample, predict_len):
    
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))  # lenght of series indexes
    for channel in range(mask.shape[1]):
        mask[:, channel][-predict_len:] = 0

    return mask

if __name__ == '__main__':
    x = torch.randn((100,14))
    mask = get_mask_tf(x,5)