import random


class Actor:
    turn_dict = {
        "right": {
            "north": "east",
            "east": "south",
            "south": "west",
            "west": "north",
        },
        "around": {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
        },
        "left": {
            "north": "west",
            "west": "south",
            "south": "east",
            "east": "north",
        },
    }

    def __init__(self, name, binary=False, start_direction=None):
        self.name = name
        self.directions = ["north", "east", "south", "west"] if not binary else ["north", "south"]
        self.direction = start_direction or random.choice(self.directions)
        self.following = None

    def same_direction(self, actor):
        self.direction = actor.get_direction()

    def opposite_direction(self, actor):
        self.direction = self.turn_dict["around"][actor.get_direction()]

    def turns(self, turn_direction):
        self.direction = self.turn_dict[turn_direction][self.direction]

    def follows(self, actor):
        self.following = actor

    def unfollows(self):
        self.following = None

    def get_direction(self):
        return self.direction if self.following is None else self.following.get_direction()

    def list_following(self):
        return [] if self.following is None else [self.following.name] + self.following.list_following()


actor_names = [
    "Bob",
    "Alice",
    "Daniel",
    "Dorothy",
    "Paul",
    "Helen",
    "Jason",
    "Ruth",
    "Michael",
    "Linda",
    "Brian",
    "Donna",
    "Matthew",
    "Betty",
    "Charles",
    "Patricia",
    "James",
    "Susan",
    "George",
    "Sarah",
    "Richard",
    "Karen",
    "Christopher",
    "Nancy",
    "Steven",
    "Carol",
    "Kevin",
    "Anna",
    "Edward",
    "Lisa",
    "Eric",
    "Michelle",
    "Timothy",
    "Jennifer",
    "Robert",
    "Kimberly",
    "Mark",
    "Jessica",
    "David",
    "Laura",
    "Joseph",
    "Maria",
    "John",
    "Sharon",
    "William",
    "Elizabeth",
    "Andrew",
    "Emily",
    "Thomas",
    "Sandra",
    "Kenneth",
    "Mary",
    "Ben",
    "Margaret",
    "Jack",
    "Paula",
    "Ethan",
    "Natalie",
    "Peter",
    "Victoria",
    "Charlie",
]
