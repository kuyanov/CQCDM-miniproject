import random
import sys
import os
import pickle
from discopy.frobenius import Ty as FTy, Box as FBox, Diagram as FDiagram

from actor import Actor, actor_names


class Scenario:
    events = ["same_dir", "opp_dir", "turns", "waves", "follows", "unfollows"]

    def __init__(self, actors, n_sentences, binary=False, enable_following=True):
        self.enable_following = enable_following
        self.turn_directions = ["left", "right", "around"] if not binary else ["around"]
        self.actors = actors
        self.idx_by_name = {}
        for i, actor in enumerate(self.actors):
            self.idx_by_name[actor.name] = i
        self.story = [("walks", actor.name, actor.direction) for actor in self.actors]
        for _ in range(n_sentences):
            self.random_event()
        self.question = random.sample(self.actors, 2)
        self.answer = self.question[0].get_direction() == self.question[1].get_direction()

    def random_event(self):
        ev = random.choice(self.events)
        if ev == "same_dir":
            actor1, actor2 = random.sample(self.actors, 2)
            if actor1.following is not None:
                self.random_event()
                return
            actor1.same_direction(actor2)
            self.story.append((ev, actor1.name, actor2.name))
        elif ev == "opp_dir":
            actor1, actor2 = random.sample(self.actors, 2)
            if actor1.following is not None:
                self.random_event()
                return
            actor1.opposite_direction(actor2)
            self.story.append((ev, actor1.name, actor2.name))
        elif ev == "turns":
            actor = random.choice(self.actors)
            if actor.following is not None:
                self.random_event()
                return
            turn_direction = random.choice(self.turn_directions)
            actor.turns(turn_direction)
            self.story.append((ev, actor.name, turn_direction))
        elif ev == "waves":
            actor1, actor2 = random.sample(self.actors, 2)
            self.story.append((ev, actor1.name, actor2.name))
        elif ev == "follows":
            if not self.enable_following:
                self.random_event()
                return
            actor1, actor2 = random.sample(self.actors, 2)
            if actor1.name in actor2.list_following():
                self.random_event()
                return
            actor1.follows(actor2)
            self.story.append((ev, actor1.name, actor2.name))
        elif ev == "unfollows":
            if not self.enable_following:
                self.random_event()
                return
            actor1 = random.choice(self.actors)
            if actor1.following is None:
                self.random_event()
                return
            actor2 = actor1.following
            actor1.unfollows()
            self.story.append((ev, actor1.name, actor2.name))
        else:
            raise NotImplementedError(ev)

    def story_diagram(self):
        dom = FTy(*[actor.name for actor in self.actors])
        cod = dom

        @FDiagram.from_callable(dom, cod)
        def diagram(*args):
            actors = list(args)
            for event in self.story:
                if event[0] == "walks":
                    name, direction = event[1:]
                    walks_box = FBox(f"walks_{direction}", FTy(name), FTy(name))
                    i = self.idx_by_name[name]
                    actors[i] = walks_box(actors[i])
                elif event[0] == "same_dir":
                    name1, name2 = event[1:]
                    same_dir_box = FBox("same_dir", FTy(name1, name2), FTy(name1, name2))
                    i1, i2 = self.idx_by_name[name1], self.idx_by_name[name2]
                    actors[i1], actors[i2] = same_dir_box(actors[i1], actors[i2])
                elif event[0] == "opp_dir":
                    name1, name2 = event[1:]
                    opp_dir_box = FBox("opp_dir", FTy(name1, name2), FTy(name1, name2))
                    i1, i2 = self.idx_by_name[name1], self.idx_by_name[name2]
                    actors[i1], actors[i2] = opp_dir_box(actors[i1], actors[i2])
                elif event[0] == "turns":
                    name, direction = event[1:]
                    turns_box = FBox(f"turns_{direction}", FTy(name), FTy(name))
                    i = self.idx_by_name[name]
                    actors[i] = turns_box(actors[i])
                elif event[0] == "waves":
                    name1, name2 = event[1:]
                    waves_box = FBox("waves", FTy(name1, name2), FTy(name1, name2))
                    i1, i2 = self.idx_by_name[name1], self.idx_by_name[name2]
                    actors[i1], actors[i2] = waves_box(actors[i1], actors[i2])
                elif event[0] == "follows":
                    name1, name2 = event[1:]
                    follows_box = FBox("follows", FTy(name1, name2), FTy(name1, name2))
                    i1, i2 = self.idx_by_name[name1], self.idx_by_name[name2]
                    actors[i1], actors[i2] = follows_box(actors[i1], actors[i2])
                elif event[0] == "unfollows":
                    name1, name2 = event[1:]
                    unfollows_box = FBox("unfollows", FTy(name1, name2), FTy(name1, name2))
                    i1, i2 = self.idx_by_name[name1], self.idx_by_name[name2]
                    actors[i1], actors[i2] = unfollows_box(actors[i1], actors[i2])
                else:
                    raise NotImplementedError(event[0])
            return tuple(actors)

        return diagram

    def question_diagram(self):
        dom = FTy(*[actor.name for actor in self.actors])
        cod = FTy("answer")

        @FDiagram.from_callable(dom, cod)
        def diagram(*args):
            actors = list(args)
            name1, name2 = self.question[0].name, self.question[1].name
            question_box = FBox("question", FTy(name1, name2), FTy("answer"))
            i1, i2 = self.idx_by_name[name1], self.idx_by_name[name2]
            return question_box(actors[i1], actors[i2])

        return diagram

    def build_diagram(self):
        return self.story_diagram() >> self.question_diagram()


def gen_scenarios(n_scenarios, min_actors, max_actors, min_sentences, max_sentences, binary=False,
                  enable_following=True):
    cnt_yes = n_scenarios // 2
    cnt_no = n_scenarios - cnt_yes
    scenarios = []
    n_iterations = 0
    while cnt_yes > 0 or cnt_no > 0:
        n_iterations += 1
        n_actors = random.randint(min_actors, max_actors)
        n_sentences = random.randint(min_sentences, max_sentences)
        actors = [Actor(name, binary=binary) for name in random.sample(actor_names, n_actors)]
        scenario = Scenario(actors, n_sentences, binary=binary, enable_following=enable_following)
        if scenario.answer is True and cnt_yes == 0 or scenario.answer is False and cnt_no == 0:
            continue
        scenarios.append(scenario)
        if scenario.answer:
            cnt_yes -= 1
        else:
            cnt_no -= 1
    print(f"Generated {n_scenarios} scenarios in {n_iterations} iterations", file=sys.stderr)
    return scenarios


def get_scenarios(name, **kwargs):
    if not os.path.exists(f"data/{name}.sc.pkl"):
        os.makedirs("data", exist_ok=True)
        scenarios = gen_scenarios(**kwargs)
        with open(f"data/{name}.sc.pkl", "wb") as fout:
            pickle.dump(scenarios, fout)
        return scenarios
    else:
        with open(f"data/{name}.sc.pkl", "rb") as fin:
            return pickle.load(fin)
