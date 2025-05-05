import torch
import numpy as np
import matplotlib.pyplot as plt

from lambeq import PytorchTrainer
from lambeq.training import PennyLaneModel

from scenario import get_scenarios
from dataset import get_dataset


if __name__ == "__main__":
    train_name = "train_n800_act2-3_sent1-5_dir2_foll"
    test_name = "test_n200_act5_sent10_dir2_foll"
    model_name = "model_train_act2-3_test_act5_dir2_foll"

    train_scenarios = get_scenarios(train_name, n_scenarios=800, min_actors=2, max_actors=3, min_sentences=1, max_sentences=5,
                                    binary=True, enable_following=True)
    test_scenarios = get_scenarios(test_name, n_scenarios=200, min_actors=5, max_actors=5, min_sentences=10, max_sentences=10,
                                   binary=True, enable_following=True)
    train_dataset = get_dataset(train_name, train_scenarios)
    test_dataset = get_dataset(test_name, test_scenarios)

    model = PennyLaneModel.from_diagrams(train_dataset.data + test_dataset.data)
    model.initialise_weights()

    trainer = PytorchTrainer(
        model=model,
        loss_function=torch.nn.functional.mse_loss,
        optimizer=torch.optim.Adam,
        learning_rate=0.03,
        epochs=50,
        evaluate_functions={
            'acc': lambda output, target: (torch.argmax(output, dim=1) == torch.argmax(target, dim=1)).double().mean()},
        evaluate_on_train=True,
        use_tensorboard=False,
        verbose='text',
        device=-1)
    trainer.fit(train_dataset, test_dataset)

    model.save(f"data/{model_name}.lt")

    fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
    ax_tl.set_title('Train dataset')
    ax_tr.set_title('Test dataset')
    ax_bl.set_xlabel('Epoch')
    ax_br.set_xlabel('Epoch')
    ax_bl.set_ylabel('Accuracy')
    ax_tl.set_ylabel('Loss')

    colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    range_ = np.arange(1, trainer.epochs + 1)
    ax_tl.plot(range_, trainer.train_epoch_costs, color=next(colours))
    ax_bl.plot(range_, trainer.train_eval_results['acc'], color=next(colours))
    ax_tr.plot(range_, trainer.val_costs, color=next(colours))
    ax_br.plot(range_, trainer.val_eval_results['acc'], color=next(colours))
    plt.show()
