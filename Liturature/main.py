import numpy as np

def onehot(inputs, size):
    newMatrix = []
    for num in inputs:
        array = np.zeros((1, size))
        array[0][num] = 1
        newMatrix.append(array[0])

    return np.array(newMatrix)


def main():
    words = open("words.txt", "r").read().splitlines()

    N = np.zeros((27, 27), dtype=np.int32)
    chars = sorted(set("".join(words)))

    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi["."] = 0

    itos = {i:s for s,i in stoi.items()}

    for w in words:
        ch = ["."] + list(w) + ["."]
        for c1, c2 in zip(ch, ch[1:]):
            x = stoi[c1]
            y = stoi[c2]
        
            N[x, y] += 1

    P = N + 1.0
    P /= P.sum(axis=1).reshape((27, 1))
    g = np.random.default_rng()

    for i in range(0, 5):
        o = []
        ix = 0
        while True:
            ix = g.multinomial(1, P[ix], size=1)[0].tolist().index(1)
            if ix == 0:
                break
            o.append(itos[ix])

        print("".join(o))

    logLiklyhood = 0.0
    n = 0
    for w in words:
        ch = ["."] + list(w) + ["."]
        for c1, c2 in zip(ch, ch[1:]):
            x = stoi[c1]
            y = stoi[c2]
            logLiklyhood += np.log(P[x, y])
            n += 1

    print(-logLiklyhood / n)

    inputs = []
    outputs = []

    for w in words:
        ch = ["."] + list(w) + ["."]
        for c1, c2 in zip(ch, ch[1:]):
            x = stoi[c1]
            y = stoi[c2]
            
            inputs.append(x)
            outputs.append(y)

    inputs = onehot(inputs, 27)
    weights = np.random.rand(27, 27)
    
    pred = inputs.dot(weights)
    pred.reshape(len(inputs), 27)
    logits = np.exp(pred)
    prob = logits / logits.sum(axis=1).reshape(len(inputs), 1)

    print(prob[0])

main()
