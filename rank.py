import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from lambeq.training import PennyLaneModel

from scenario import get_scenarios
from dataset import get_dataset, diagram2circuit


def attempt(params, fixed, loss_func, num_iter, lr, lr_decay, eps, tol, debug):
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=eps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay)
    pbar = tqdm(total=num_iter, unit='batches', leave=False) if debug else None
    for _ in range(num_iter):
        loss = loss_func(params)
        mx = torch.max(torch.abs(torch.cat(list(map(torch.flatten, params)))))
        if debug:
            pbar.set_postfix(loss=f'{loss.item():.6f}', mx=f'{mx:.2f}')
            pbar.update()
        if loss.isnan() or loss.isinf():
            return False
        if loss < tol:
            return True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        with torch.no_grad():
            for p, f in zip(params, fixed):
                p[~f.isnan()] = f[~f.isnan()]
    return False


def approximate(param_shapes, loss_func, num_iter, lr, lr_decay, eps, tol, dtype=torch.float, debug=False):
    for _ in range(3):
        params = [torch.randn(shape, dtype=dtype, requires_grad=True) for shape in param_shapes]
        fixed = [torch.full(shape, torch.nan, dtype=dtype) for shape in param_shapes]
        success = attempt(params, fixed, loss_func, num_iter, lr, lr_decay, eps, tol, debug)
        if success:
            return params
    return None


def outer_product(vectors):
    if len(vectors) == 1:
        return vectors[0]
    tensor = outer_product(vectors[1:])
    view_shape = (1,) * tensor.dim() + (vectors[0].shape[0],)
    reshaped_vector = vectors[0].view(view_shape)
    return tensor.unsqueeze(-1) * reshaped_vector


def loss(tensor, factors):
    r = factors[0].shape[1]
    approx = torch.zeros_like(tensor)
    for i in range(r):
        approx += outer_product([fac[:,i] for fac in factors])
    return torch.sqrt((torch.abs(approx - tensor) ** 2).sum())


def canonical_rank(tensor, max_r=100):
    tensor = torch.tensor(tensor)
    d = len(tensor.shape)
    for r in range(1, max_r + 1):
        factor_shapes = [(tensor.shape[i], r) for i in range(d)]
        factors = approximate(factor_shapes,
                              lambda factors: loss(tensor, factors),
                              num_iter=1000,
                              lr=0.2,
                              lr_decay=0.99,
                              eps=1e-3,
                              tol=1e-2,
                              dtype=tensor.dtype,
                              debug=True)
        if factors is not None:
            return r
    return max_r + 1


if __name__ == '__main__':
    train_filename = "train_n400_act2-3_sent1-3_dir2"
    test_filename = "test_n100_act5_sent10_dir2"
    model_filename = "model_train_act2-3_test_act5_dir2"

    train_scenarios = get_scenarios(train_filename)
    test_scenarios = get_scenarios(test_filename)
    train_dataset = get_dataset(train_filename)
    test_dataset = get_dataset(test_filename)

    model = PennyLaneModel.from_checkpoint(f"data/{model_filename}.lt")

    ranks = []
    for scenario in test_scenarios:
        circuit = diagram2circuit(scenario.story_diagram()).to_pennylane()
        circuit.initialise_concrete_params(model.symbol_weight_map)
        tensor = circuit.eval().detach().numpy()
        ranks.append(canonical_rank(tensor))

    plt.hist(ranks)
    plt.xlabel('canonical rank distribution')
    plt.show()
