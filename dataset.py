import sys
import os
import pickle

from discopy.grammar.pregroup import Ty as GTy, Box as GBox, Functor
from lambeq import *
from lambeq.backend.grammar import Ty as LTy
from lambeq.backend.converters.discopy import from_discopy

from actor import actor_names


def diagram2circuit(diagram):
    def ob(ty):
        res = GTy()
        for t in ty:
            if str(t) in actor_names:
                res @= GTy("actor")
            elif str(t) == "answer":
                res @= GTy("bool")
            else:
                raise NotImplementedError(t)
        return res

    def ar(box):
        if box.name in ["same_dir", "opp_dir", "follows", "unfollows"]:
            return (ob(box.dom) @ GBox("", GTy(), GTy("ancilla")) >>
                    GBox(box.name, ob(box.dom) @ GTy("ancilla"), ob(box.cod) @ GTy("ancilla")) >>
                    ob(box.cod) @ GBox("", GTy("ancilla"), GTy()))
        else:
            return GBox(box.name, ob(box.dom), ob(box.cod))

    functor = Functor(ob, ar)
    remove_cups = RemoveCupsRewriter()
    ansatz = Sim4Ansatz({
        LTy("ancilla"): 1,
        LTy("actor"): 1,
        LTy("bool"): 1,
    }, n_layers=3)
    diagram = from_discopy(functor(diagram))
    diagram = remove_cups(diagram).normal_form()
    return ansatz(diagram)


def get_dataset(name, scenarios=None, batch_size=32, shuffle=True):
    if not os.path.exists(f"data/{name}.ds.pkl"):
        os.makedirs("data", exist_ok=True)
        circuits = [diagram2circuit(scenario.build_diagram()) for scenario in scenarios]
        labels = [[1, 0] if scenario.answer else [0, 1] for scenario in scenarios]
        dataset = Dataset(circuits, labels, batch_size=batch_size, shuffle=shuffle)
        print(f"Built dataset with {len(scenarios)} scenarios", file=sys.stderr)
        with open(f"data/{name}.ds.pkl", "wb") as fout:
            pickle.dump(dataset, fout)
        return dataset
    else:
        with open(f"data/{name}.ds.pkl", "rb") as fin:
            return pickle.load(fin)
