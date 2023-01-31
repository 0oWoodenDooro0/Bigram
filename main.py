import torch
import torch.nn.functional as F

with open('names.txt', 'r') as file:
    names = [x.replace('\n', '') for x in file.readlines()]

b = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(names))))

char_to_index = {c: i + 1 for i, c in enumerate(chars)}
char_to_index['.'] = 0

index_to_char = {i + 1: c for i, c in enumerate(chars)}
index_to_char[0] = '.'

xs, ys = [], []
for name in names[:]:
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs[:], chs[1:]):
        index1 = char_to_index[ch1]
        index2 = char_to_index[ch2]
        b[index1, index2] += 1
        xs.append(index1)
        ys.append(index2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

generator = torch.Generator().manual_seed(57373543)
hidden_layer = torch.randn((27, 27), generator=generator, requires_grad=True)

for _ in range(100):
    input_layer = F.one_hot(xs, num_classes=27).float()
    logit = input_layer @ hidden_layer
    counts = logit.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean()
    print(loss.item())
    hidden_layer.grad = None
    loss.backward()

    hidden_layer.data += -1 * hidden_layer.grad
