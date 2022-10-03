#!/usr/bin/python3
# -*- coding: utf-8 -*-

# TODO unit tests (copy doc)
# TODO std tests
# TODO arg tests (with json output)

# %%
import sys
import string
import random
import json

from typing import TypedDict
from math import sqrt
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm

KEYBOARD_CHARS = "qazwsxedcrfvtgbyhnujmik,ol.p;/"  # Qwerty top to bottom column-wise


class Individual(TypedDict):
    genome: str
    fitness: int


Population = list[Individual]


def initialize_individual(genome: str, fitness: int) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as string, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[str, int]
    Modifies:       Nothing
    """
    return Individual(genome=genome, fitness=fitness)


def initialize_pop(objective: str, pop_size: int) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     Goal string, population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    """
    return Population(
        [
            initialize_individual(
                genome="".join(
                    random.sample(
                        list(objective), len(objective)
                    )  # Shuffle doesn't work on strings python 3.10 :(
                ),
                fitness=0,
            )
            for _ in range(pop_size)
        ]
    )


def recombine_pair(parent1: Individual, parent2: Individual) -> Population:
    """
    Purpose:        Recombine two parents to produce one child (pythonic video implementation)
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    """

    def rotate_str(s: str, n: int):
        """Positive n rotates s left; negative n rotates s right; no effect if n > len(s)"""
        return s[n:] + s[:n]

    # Crossover with wrap around
    crossover_point = random.randint(
        0, len(parent1["genome"]) - 1
    )  # Index where parent1's genes start in child genome
    contrib_length = random.randint(
        0, len(parent1["genome"]) - 1
    )  # How much of the genome to carry over from parent 1

    # Rotating simplifies wrap around crossovers
    rotated_p1 = rotate_str(parent1["genome"], crossover_point)
    rotated_p2 = rotate_str(
        parent2["genome"], (crossover_point + contrib_length) % len(parent1["genome"])
    )

    parent1_contrib = rotated_p1[:contrib_length]

    # Complete genome with unused letters, preserving ordering, from parent2
    parent2_contrib = "".join(
        [c if c not in parent1_contrib else "" for c in rotated_p2]
    )[: len(parent1["genome"]) - len(parent1_contrib)]

    # Rotate genome back so parent1's contribution starts at position crossover_point
    child_genome = rotate_str(parent1_contrib + parent2_contrib, -crossover_point)

    return Individual(genome=child_genome, fitness=0)


def recombine_group(parents: Population, num_children: int) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population of size parents
    Parameters:     list of parents to breed, amount of children to have
    User Input:     no
    Prints:         no
    Returns:        New population of children
    Modifies:       Nothing
    """
    # Randomly select, with replacement (as in the video), which parents to breed with which parents
    return Population(
        [
            recombine_pair(parent, random.choice(parents))
            for parent in random.choices(parents, k=num_children)
        ]
    )


def mutate_individual(parent: Individual, mutate_rate: float) -> Individual:
    """
    Purpose:        Mutate one individual by swapping two positions
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    """
    if random.random() > mutate_rate:
        return parent

    swap_pos1 = random.randint(0, len(parent["genome"]) - 1)
    swap_val = parent["genome"][swap_pos1]
    swap_pos2 = random.randint(0, len(parent["genome"]) - 1)

    new_genome = list(parent["genome"])
    new_genome[swap_pos1] = new_genome[swap_pos2]
    new_genome[swap_pos2] = swap_val
    new_genome = "".join(new_genome)

    return Individual(genome=new_genome, fitness=0)


def mutate_group(children: Population, mutate_rate: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    """
    return Population([mutate_individual(child, mutate_rate) for child in children])


def evaluate_individual(objective: str, individual: Individual) -> Individual:
    """
    Purpose:        Computes and modifies the objective fitness for one individual (ten finger evaluation)
    Parameters:     Objective string, One Individual
    User Input:     no
    Prints:         no
    Returns:        Evaluated individual (for multiprocessing purposes)
    Modifies:       None
    """
    def get_homekey(idx):
        col = idx // 3

        if col == 4:  # Left index finger
            idx -= 3
        elif col == 5:  # Right index finger
            idx += 3

        row = idx % 3

        if row == 0:
            return idx + 1
        elif row == 1:
            return idx
        return idx - 1

    # Distances per key location, comments are for examples for QWERTY
    distances = {
        0: {0: 0,       1: 1.031,   2: 2.136},  # Q
        1: {0: 1.031,   1: 0,       2: 1.118},  # A
        2: {0: 2.136,   1: 1.118,   2: 0},      # Z
        3: {3: 0,       4: 1.031,   5: 2.136},  # W
        4: {3: 1.031,   4: 0,       5: 1.118},  # S
        5: {3: 2.136,   4: 1.118,   5: 0},      # X
        6: {6: 0,       7: 1.031,   8: 2.136},  # E
        7: {6: 1.031,   7: 0,       8: 1.118},  # D
        8: {6: 2.136,   7: 1.118,   8: 0},      # C
        9: {9: 0,       10: 1.031,  11: 2.136,  12: 1,      13: 1.601,  14: 2.658}, # R
        10: {9: 1.031,  10: 0,      11: 1.118,  12: 1.25,   13: 1,      14: 1.803}, # F
        11: {9: 2.136,  10: 1.118,  11: 0,      12: 2.016,  13: 1.118,  14: 1},     # V
        12: {9: 1,      10: 1.25,   11: 2.016,  12: 0,      13: 1.031,  14: 2.136}, # T
        13: {9: 1.601,  10: 1,      11: 1.118,  12: 1.031,  13: 0,      14: 1.118}, # G
        14: {9: 2.568,  10: 1.803,  11: 1,      12: 2.136,  13: 1.118,  14: 0},     # B
        15: {15: 0,     16: 1.031,  17: 2.136,  18: 1,      19: 1.601,  20: 2.658}, # Y
        16: {15: 1.031, 16: 0,      17: 1.118,  18: 1.25,   19: 1,      20: 1.803}, # H
        17: {15: 2.136, 16: 1.118,  17: 0,      18: 2.016,  19: 1.118,  20: 1},     # N
        18: {15: 1,     16: 1.25,   17: 2.016,  18: 0,      19: 1.031,  20: 2.136}, # U
        19: {15: 1.601, 16: 1,      17: 1.118,  18: 1.031,  19: 0,      20: 1.118}, # J
        20: {15: 2.568, 16: 1.803,  17: 1,      18: 2.136,  19: 1.118,  20: 0},     # M
        21: {21: 0,     22: 1.031,  23: 2.136}, # I
        22: {21: 1.031, 22: 0,      23: 1.118}, # K
        23: {21: 2.136, 22: 1.118,  23: 0},     # ,
        24: {24: 0,     25: 1.031,  26: 2.136}, # O
        25: {24: 1.031, 25: 0,      26: 1.118}, # L
        26: {24: 2.136, 25: 1.118,  26: 0},     # .
        27: {27: 0,     28: 1.031,  29: 2.136}, # P
        28: {27: 1.031, 28: 0,      29: 1.118}, # ;
        29: {27: 2.136, 28: 1.118,  29: 0}      # /
    }

    char_index = {c: i for i, c in enumerate(individual["genome"])}

    # Remove invalid characters
    valid_obj = objective
    for c in set(objective) - set(individual["genome"]):
        valid_obj = valid_obj.replace(c, "")

    # Calculate distances
    total_dist = 0
    # Zip to loop through previous character and current character, starting from homekey of first char
    for prev, curr in zip(
        individual["genome"][get_homekey(char_index[valid_obj[0]])] + valid_obj[:-1],
        valid_obj,
    ):
        curr_index = char_index[curr]
        prev_index = char_index[prev]
        total_dist += distances[curr_index].get(prev_index, distances[curr_index][get_homekey(curr_index)])

    return Individual(
        genome=individual["genome"], fitness=individual["fitness"] + total_dist
    )


def evaluate_group(objectives: list[str], individuals: Population) -> Population:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     List of objective strings, Population
    User Input:     no
    Prints:         no
    Returns:        The individuals
    Modifies:       None
    """
    amt_objectives = len(objectives)
    with Pool() as p:
        for objective in tqdm(objectives, desc="Evaluatation Progress"):
            individuals = p.starmap(
                evaluate_individual, zip([objective] * len(individuals), individuals)
            )
    print()
    return individuals


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    """
    individuals.sort(key=lambda x: x["fitness"], reverse=True)


def evolve(objectives: list[str], pop_size: int, epochs: int) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The evolved population of solutions
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    """
    print("Initializing population...")
    population = initialize_pop(objective=KEYBOARD_CHARS, pop_size=pop_size)
    print("done.")

    population = evaluate_group(objectives=objectives, individuals=population)

    rank_group(individuals=population)

    for epoch in range(epochs):
        print(f"Generating generation {epoch + 1}...")
        # Top 10% get copied to new generation
        survived = Population(
            [
                Individual(genome=i["genome"], fitness=0)
                for i in population[-len(population) // 10 :]
            ]
        )
        new_gen = survived + recombine_group(
            parents=population[-len(population) // 2 :],
            num_children=len(population) - len(survived),
        )  # Top 50% get to breed
        mut_new_gen = mutate_group(children=new_gen, mutate_rate=0.10)
        print("done.")

        assert len(new_gen) == len(population), "Old gen and new gen sizes do not match"

        mut_new_gen = evaluate_group(objectives=objectives, individuals=mut_new_gen)

        rank_group(individuals=mut_new_gen)
        print(
            f"Epoch {epoch} with best individual (Fitness: {mut_new_gen[-1]['fitness']:.3f}):"
        )
        print_keyboard(mut_new_gen[-1]["genome"])
        # print(population)  # Uncomment to make cool patterns!
        print("\n\n")

        population = mut_new_gen

    return population


def print_keyboard(genome: str):
    """
    Purpose:        Prints genome in an ASCII keyboard
    Parameters:     The genome string
    User Input:     No
    Prints:         ASCII keyboard
    Returns:        None
    Modifies:       None
    """
    print("+---" * 10 + "+")

    row_wise = genome[0::3] + genome[1::3] + genome[2::3]
    for i, key in enumerate(row_wise):
        if not i % 10 and i != 0:
            print("|")
            print("|" * (i // 10) + "----" * 10 + "-")
            print("|" * (i // 10), end="")
        print(f"| {key} ", end="")
    print("|\n" + "+-" + "+---" * 10 + "+")

def distance1(prev, curr):
        if prev == curr:
            return 0

        idx_diff = abs(prev - curr)
        prev_col = prev // 3
        curr_col = curr // 3
        prev_row = prev % 3
        curr_row = curr % 3

        # Prob worst thing I've written since 1570...
        if prev_col == curr_col:  # Keys responsible for the same finger
            if idx_diff == 1:  # Up or down a row
                if min(prev_row, curr_row):  # Mid -> Bottom
                    stagger = 0.5
                else:  # Top -> Mid
                    stagger = 0.25
            else:  # idx_diff == 2; Up or down two rows
                stagger = 0.75
        elif (min(prev_col, curr_col) == 3 and max(prev_col, curr_col) == 4) or (
            min(prev_col, curr_col) == 5 and max(prev_col, curr_col) == 6
        ):  # Index fingers
            if idx_diff == 1:  # V -> T on QWERTY
                stagger = 0.25
            elif idx_diff == 2:  # F -> T or V -> G on QWERTY
                if min(prev_row, curr_row):  # V -> G
                    stagger = 0.5
                else:  # F -> T
                    stagger = 0.75
            elif idx_diff == 3:  # Key is directly to the right
                stagger = 1
            elif idx_diff == 4:  # R -> G or F -> B on QWERTY
                if min(prev_row, curr_row):  # F-> B
                    stagger = 1.5
                else:  # R -> G
                    stagger = 1.25
            else:  # idx_diff == 5; R -> B on QWERTY
                stagger = 1.75
        else:  # Unrelated keys
            idx_diff = abs(get_homekey(curr) - curr)
            if idx_diff == 1:  # Up or down a row
                if min(prev_row, curr_row):  # Mid -> Bottom
                    stagger = 0.5
                else:  # Top -> Mid
                    stagger = 0.25
            else:  # idx_diff == 2; Up or down two rows
                stagger = 0.75

        return sqrt(stagger**2 + abs(prev_row - curr_row) ** 2)


if __name__ == "__main__":
    POPULATION_SIZE = 150  # Be aware that the lower the population size, the higher the chance for homogeneity
    GENERATIONS = 20  # Number of Epochs; Video used 1000 to generate meaningful results
    TRAINING_SIZE = (
        0.001  # Set to 1 to use all data; 0.001 makes for reasonable demonstration time
    )

    print("Reading training dataset...")
    with open(Path("data/arxiv-metadata-oai-snapshot.json"), "r") as fin:
        data = (
            fin.readlines()
        )  # Some sort of json formatting issue, so load it the slow way
        objectives = [
            json.loads(d)["abstract"].lower()
            for d in random.sample(data, k=int(len(data) * TRAINING_SIZE))
        ]
    print("done.")

    population = evolve(objectives, POPULATION_SIZE, GENERATIONS)

    print(f"Best layout (Fitness: {population[-1]['fitness']}):")
    print_keyboard(population[-1]["genome"])

    Path("out").mkdir(exist_ok=True)
    with open(
        Path("out") / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"), "w"
    ) as fout:
        json.dump(population, fout)
