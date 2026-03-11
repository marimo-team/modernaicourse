import copy
import math
import torch
import mugrade
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

mnist_test = datasets.MNIST(".", train=False, download=True)
X_test = mnist_test.data.reshape(-1, 784).float() / 255.0
y_test = mnist_test.targets


def test_Linear(Linear):
    torch.manual_seed(0)
    layer = Linear(10, 20)
    assert(hasattr(layer, "weight"))
    assert(isinstance(layer.weight, nn.Parameter))
    assert(layer.weight.shape == (20, 10))

    ref_layer = nn.Linear(10, 20, bias=False)
    ref_layer.weight.data = layer.weight.data.clone()

    X = torch.randn(50, 10)
    assert(layer(X).shape == (50, 20))
    assert(torch.allclose(ref_layer(X), layer(X), atol=1e-6))

    X = torch.randn(7, 9, 10)
    assert(layer(X).shape == (7, 9, 20))
    assert(torch.allclose(ref_layer(X), layer(X), atol=1e-6))

    layer = Linear(100, 1000)
    assert(torch.allclose(layer.weight.std(), torch.tensor(math.sqrt(2 / 100)), atol=3e-3))


def submit_Linear(Linear):
    layer = Linear(4, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, -1.0, 0.5, 2.0],
                                         [-0.5, 0.0, 1.0, -1.5],
                                         [2.0, 1.5, -0.5, 0.25]]))
    X = torch.tensor([[1.0, 2.0, -1.0, 0.5],
                      [0.0, -1.0, 2.0, 3.0]])
    mugrade.submit(layer(X).detach().numpy())
    mugrade.submit(layer(torch.stack([X, -X])).detach().numpy())
    mugrade.submit(type(layer.weight))


def test_CrossEntropyLoss(CrossEntropyLoss):
    torch.manual_seed(1)
    logits = torch.randn(100, 50)
    y = torch.randint(0, 50, (100,))
    ref = nn.CrossEntropyLoss()(logits, y)
    mine = CrossEntropyLoss()(logits, y)
    assert(mine.ndim == 0)
    assert(torch.allclose(ref, mine, atol=1e-6))

    logits = torch.tensor([[1000.0, 1001.0, 999.5], [1.0, -2.0, 0.5]])
    y = torch.tensor([1, 2])
    ref = nn.CrossEntropyLoss()(logits, y)
    mine = CrossEntropyLoss()(logits, y)
    assert(torch.allclose(ref, mine, atol=1e-6))


def submit_CrossEntropyLoss(CrossEntropyLoss):
    logits = torch.tensor([[2.0, 1.0, 0.0],
                           [0.0, 2.0, 1.0],
                           [1.5, -0.5, 0.25]])
    y = torch.tensor([0, 2, 1])
    loss = CrossEntropyLoss()(logits, y)
    mugrade.submit(loss.item())
    mugrade.submit(type(loss))

    logits = torch.tensor([[1000.0, 1002.0], [999.0, 998.0]])
    y = torch.tensor([1, 0])
    mugrade.submit(CrossEntropyLoss()(logits, y).item())


def test_SGD(SGD):
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(6, 4, bias=False), nn.ReLU(), nn.Linear(4, 3, bias=False))
    ref_model = copy.deepcopy(model)

    opt = SGD(model.parameters(), learning_rate=0.05)
    ref_opt = optim.SGD(ref_model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    X = torch.randn(12, 6)
    y = torch.randint(0, 3, (12,))

    for _ in range(3):
        opt.zero_grad()
        ref_opt.zero_grad()

        loss = loss_fn(model(X), y)
        ref_loss = loss_fn(ref_model(X), y)

        loss.backward()
        ref_loss.backward()

        opt.step()
        ref_opt.step()

    for p, p_ref in zip(model.parameters(), ref_model.parameters()):
        assert(torch.allclose(p, p_ref, atol=1e-6))

    loss = loss_fn(model(X), y)
    loss.backward()
    opt.zero_grad()
    for p in model.parameters():
        if p.grad is not None:
            assert(torch.allclose(p.grad, torch.zeros_like(p.grad)))


def submit_SGD(SGD):
    w = nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
    b = nn.Parameter(torch.tensor([0.25, -0.75]))
    opt = SGD([w, b], learning_rate=0.2)

    w.grad = torch.tensor([[0.1, -0.4], [-0.2, 0.3]])
    b.grad = torch.tensor([0.5, -0.25])
    opt.step()

    mugrade.submit(w.detach().numpy())
    mugrade.submit(b.detach().numpy())

    opt.zero_grad()
    mugrade.submit(w.grad.detach().numpy())
    mugrade.submit(b.grad.detach().numpy())


def test_DataLoader(DataLoader):
    X = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    y = torch.arange(12)
    loader = DataLoader(X, y, batch_size=5)

    assert(iter(loader) is loader)

    batches = list(loader)
    assert(len(batches) == 3)

    for i, (Xb, yb) in enumerate(batches):
        start = 5 * i
        end = min(5 * (i + 1), len(X))
        assert(torch.equal(Xb, X[start:end]))
        assert(torch.equal(yb, y[start:end]))

    batches2 = list(loader)
    assert(len(batches2) == 3)
    for (Xb1, yb1), (Xb2, yb2) in zip(batches, batches2):
        assert(torch.equal(Xb1, Xb2))
        assert(torch.equal(yb1, yb2))


def submit_DataLoader(DataLoader):
    X = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    y = torch.tensor([3, 0, 4, 1, 5, 2, 6])
    loader = DataLoader(X, y, batch_size=3)
    batches = list(loader)

    mugrade.submit(len(batches))
    for Xb, yb in batches:
        mugrade.submit(Xb.numpy())
        mugrade.submit(yb.numpy())


def test_epoch(epoch):
    torch.manual_seed(3)
    X = torch.randn(11, 4)
    y = torch.randint(0, 3, (11,))
    loss_fn = nn.CrossEntropyLoss()

    model = nn.Linear(4, 3, bias=False)
    params_before = [p.detach().clone() for p in model.parameters()]
    loader = list(zip(X.split(4), y.split(4)))

    eval_loss, eval_err = epoch(model, loader, loss_fn)
    assert(isinstance(eval_loss, float))
    assert(isinstance(eval_err, float))

    with torch.no_grad():
        logits = model(X)
        ref_loss = loss_fn(logits, y).item()
        ref_err = (logits.argmax(-1) != y).float().mean().item()

    assert(abs(eval_loss - ref_loss) < 1e-6)
    assert(abs(eval_err - ref_err) < 1e-6)

    for p_before, p_after in zip(params_before, model.parameters()):
        assert(torch.allclose(p_before, p_after))

    model = nn.Linear(4, 3, bias=False)
    ref_model = copy.deepcopy(model)
    opt = optim.SGD(model.parameters(), lr=0.1)
    ref_opt = optim.SGD(ref_model.parameters(), lr=0.1)
    loader = list(zip(X.split(3), y.split(3)))

    train_loss, train_err = epoch(model, loader, loss_fn, opt)

    total_loss = 0.0
    total_err = 0.0
    N = 0
    for Xb, yb in loader:
        ref_opt.zero_grad()
        logits = ref_model(Xb)
        l = loss_fn(logits, yb)
        total_loss += l.item() * len(yb)
        total_err += (logits.argmax(-1) != yb).float().sum().item()
        N += len(yb)
        l.backward()
        ref_opt.step()

    ref_train_loss = total_loss / N
    ref_train_err = total_err / N

    assert(abs(train_loss - ref_train_loss) < 1e-6)
    assert(abs(train_err - ref_train_err) < 1e-6)

    for p, p_ref in zip(model.parameters(), ref_model.parameters()):
        assert(torch.allclose(p, p_ref, atol=1e-6))


def submit_epoch(epoch):
    X = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0],
                      [2.0, -1.0]], dtype=torch.float32)
    y = torch.tensor([0, 1, 1, 0])
    loss_fn = nn.CrossEntropyLoss()

    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -1.0],
                                         [-0.5, 0.25]]))

    loader = list(zip(X.split(2), y.split(2)))

    eval_loss, eval_err = epoch(model, loader, loss_fn)
    mugrade.submit(eval_loss)
    mugrade.submit(eval_err)

    opt = optim.SGD(model.parameters(), lr=0.2)
    train_loss, train_err = epoch(model, loader, loss_fn, opt)
    mugrade.submit(train_loss)
    mugrade.submit(train_err)
    mugrade.submit(model.weight.detach().numpy())


def test_eval_linear_model(eval_linear_model):
    model = eval_linear_model()
    assert(isinstance(model, nn.Module))


    with torch.no_grad():
        logits = model(X_test)
    assert(logits.shape == (10000, 10))
    err = (logits.argmax(-1) != y_test).float().mean().item()
    assert(err < 0.1)


def submit_eval_linear_model(eval_linear_model):
    model = eval_linear_model()

    with torch.no_grad():
        logits = model(X_test)
        err = (logits.argmax(-1) != y_test).float().mean().item()

    mugrade.submit(err < 0.1)
    mugrade.submit(logits.shape)
    mugrade.submit(type(model))


def test_TwoLayerNN(TwoLayerNN):
    torch.manual_seed(4)
    model = TwoLayerNN(8, 16, 5)
    assert(hasattr(model, "linear1"))
    assert(hasattr(model, "linear2"))

    ref_model = nn.Sequential(nn.Linear(8, 16, bias=False), nn.ReLU(), nn.Linear(16, 5, bias=False))
    with torch.no_grad():
        ref_model[0].weight.copy_(model.linear1.weight)
        ref_model[2].weight.copy_(model.linear2.weight)

    X = torch.randn(13, 8)
    assert(torch.allclose(model(X), ref_model(X), atol=1e-6))

    X = torch.randn(2, 7, 8)
    assert(torch.allclose(model(X), ref_model(X), atol=1e-6))



def submit_TwoLayerNN(TwoLayerNN):
    model = TwoLayerNN(3, 4, 2)
    with torch.no_grad():
        model.linear1.weight.copy_(torch.tensor([[1.0, -1.0, 0.5],
                                                 [-0.5, 0.0, 1.0],
                                                 [0.25, -0.75, 0.5],
                                                 [1.5, 0.5, -1.0]]))
        model.linear2.weight.copy_(torch.tensor([[1.0, -2.0, 0.5, 0.0],
                                                 [-1.0, 0.5, 1.5, -0.5]]))

    X = torch.tensor([[1.0, 2.0, -1.0],
                      [0.5, -0.5, 0.25],
                      [-1.0, 0.0, 1.0]])
    mugrade.submit(model(X).detach().numpy())
    mugrade.submit(model(torch.stack([X, -X])).detach().numpy())
    mugrade.submit(len(list(model.parameters())))


def test_eval_two_layer_nn(eval_two_layer_nn):
    model = eval_two_layer_nn()
    assert(isinstance(model, nn.Module))

    X_sub, y_sub = X_test[:2000], y_test[:2000]

    with torch.no_grad():
        logits = model(X_sub)
    assert(logits.shape == (2000, 10))
    err = (logits.argmax(-1) != y_sub).float().mean().item()
    assert(err < 0.03)


def submit_eval_two_layer_nn(eval_two_layer_nn):
    model = eval_two_layer_nn()

    with torch.no_grad():
        logits = model(X_test[:1000])
        err = (logits.argmax(-1) != y_test[:1000]).float().mean().item()

    mugrade.submit(err < 0.04)
    mugrade.submit(logits.shape)
    mugrade.submit(type(model))


def test_MultiLayerNN(MultiLayerNN):
    torch.manual_seed(5)
    hidden_dims = [7, 5, 4]
    model = MultiLayerNN(6, 3, hidden_dims)

    assert(hasattr(model, "linears"))
    assert(isinstance(model.linears, nn.ModuleList))
    assert(len(model.linears) == len(hidden_dims) + 1)

    dims = [6] + hidden_dims + [3]
    ref_modules = []
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        ref_linear = nn.Linear(din, dout, bias=False)
        with torch.no_grad():
            ref_linear.weight.copy_(model.linears[i].weight)
        ref_modules.append(ref_linear)
        if i < len(dims) - 2:
            ref_modules.append(nn.ReLU())

    ref_model = nn.Sequential(*ref_modules)

    X = torch.randn(9, 6)
    assert(torch.allclose(model(X), ref_model(X), atol=1e-6))

    X = torch.randn(2, 4, 6)
    assert(torch.allclose(model(X), ref_model(X), atol=1e-6))


def submit_MultiLayerNN(MultiLayerNN):
    model = MultiLayerNN(2, 1, [3, 2])

    with torch.no_grad():
        model.linears[0].weight.copy_(torch.tensor([[1.0, -1.0],
                                                    [0.5, 0.25],
                                                    [-0.75, 0.5]]))
        model.linears[1].weight.copy_(torch.tensor([[1.0, -0.5, 0.25],
                                                    [-1.0, 0.5, 1.0]]))
        model.linears[2].weight.copy_(torch.tensor([[2.0, -1.5]]))

    X = torch.tensor([[1.0, 2.0],
                      [-1.0, 0.5],
                      [0.25, -0.75]])

    mugrade.submit(model(X).detach().numpy())
    mugrade.submit(model(torch.stack([X, -X])).detach().numpy())
    mugrade.submit(len(model.linears))
