from typing import Any
from collections.abc import Callable
import numpy as np
import ast
from copy import deepcopy


def powerlaw_generator(N, d, idx):
    """Generates a combination of rate laws in terms of power given a stoichiometric list.

    Args:
        N: The stoichiometric list (all elements should be greater than or equal to zero).
        d: An empty dictionary.
        idx: Refers to the index of the rate constant to be used.
             0 for irreversible reactions, 1 for reversible reactions.

    Returns:
        None. Modifies the dictionary in place.

    Uses recursion and hashmap (dynamic programming) for efficiency.
    """
    s = str(N)
    if s in d:
        return d[s]

    elif sum(N) == 0:
        d[s] = lambda y, K: K[idx] * np.ones((y[0].shape))
        return d[s]

    else:
        for i in range(len(N)):
            if N[i] != 0:
                Nnew = deepcopy(N)
                Nnew[i] = Nnew[i] - 1
                powerlaw_generator(Nnew, d, idx)
                if s not in d:
                    fun = d[str(Nnew)]
                    # binding i is important. Else will return the same function
                    d[s] = lambda y, K, saved_i=i: y[saved_i] * fun(y, K)
        return d[s]


class RateLaw:
    def __init__(self, type: str, expression: str, function: Callable[[list], Any]):
        self.type = type
        self.expression = expression
        self.function = function

    def __repr__(self):
        return f'RateLaw(type = "{self.type}", Expression = "{self.expression}")'

    # have to cross check error handling capabilities #
    def __add__(self, other):
        """Addition compatibility"""
        func = lambda y, K: self.function(y, K) + other.function(y, K)  # noqa: E731
        return RateLaw(
            type="<unk>",
            expression=self.expression + "+ " + other.expression,
            function=func,
        )

    def __sub__(self, other):
        """Subtraction compatibility"""
        func = lambda y, K: self.function(y, K) - other.function(y, K)  # noqa: E731
        return RateLaw(
            type="<unk>",
            expression=self.expression + "- " + other.expression,
            function=func,
        )

    def __mul__(self, other):
        """Product compatibility"""
        func = lambda y, K: self.function(y, K) * other.function(y, K)  # noqa: E731
        return RateLaw(
            type="<unk>",
            expression=self.expression + "*" + other.expression,
            function=func,
        )

    def __truediv__(self, other):
        """Division compatibility"""
        func = lambda y, K: self.function(y, K) / other.function(y, K)  # noqa: E731
        return RateLaw(
            type="<unk>",
            expression=self.expression + "/ " + other.expression,
            function=func,
        )


class CandidateRateLaws:
    """
    Represents a collection of candidate rate laws based on stoichiometric coefficients.

    Args:
        N (list): The stoichiometric list.
        type (list): The types of rate laws to consider (e.g., ["Reversible", "Irreversible"]).

    Attributes:
        _ratelaws (list): The list of RateLaw objects representing the candidate rate laws.

    """

    def __init__(self, N, type=["Reversible", "Irreversible"]):
        self.N = N

        if (
            "Irreversible" in type or "Reversible" in type
        ):  # This might misbehave if user types IrReversible. Can Improvise
            # we have to calculate irreversible rate laws even if it is just reversible
            N_irr = [abs(num) if num < 0 else 0 for num in N]
            d_irr = {}
            powerlaw_generator(N_irr, d_irr, idx=0)

            # irreversible
            self._ratelaws = []
            for key in d_irr.keys():
                temp = ast.literal_eval(
                    key
                )  # converts string repr of list to list back
                exp_list = [f"C[{i}]^{num}" for i, num in enumerate(temp) if num > 0]
                exp_list.insert(0, "K[0]")
                exp = "*".join(exp_list)

                if exp == "":
                    exp == "<unk>"  # unknown expression if it's unable to generate

                if "Irreversible" in type:
                    self._ratelaws.append(
                        RateLaw(
                            type="Irreversible", expression=exp, function=d_irr[key]
                        )
                    )

        if "Reversible" in type:
            N_rev_temp = [num if num > 0 else 0 for num in N]
            d_rev_temp = {}
            powerlaw_generator(N_rev_temp, d_rev_temp, idx=1)

            for key1 in d_irr.keys():
                for key2 in d_rev_temp.keys():
                    temp1 = ast.literal_eval(key1)
                    temp2 = ast.literal_eval(key2)
                    exp_list1 = [
                        f"C[{i}]^{num}" for i, num in enumerate(temp1) if num > 0
                    ]
                    exp_list1.insert(0, "K[0]")
                    exp_list2 = [
                        f"C[{i}]^{num}" for i, num in enumerate(temp2) if num > 0
                    ]
                    exp_list2.insert(0, "K[1]")
                    exp_1 = "*".join(exp_list1)
                    exp_2 = "*".join(exp_list2)

                    if exp_1 == "" and exp_2 == "":
                        exp = "<unk>"
                    elif exp_2 == "":
                        exp = exp_1

                    if exp_1 == "K[0]" or exp_2 == "K[1]":
                        continue

                    r_fun = lambda y, K, saved_key1=key1, saved_key2=key2: d_irr[  # noqa: E731
                        saved_key1
                    ](y, K) - d_rev_temp[saved_key2](y, K)
                    self._ratelaws.append(
                        RateLaw(
                            type="Reversible",
                            expression=exp_1 + "- " + exp_2,
                            function=r_fun,
                        )
                    )

    def __len__(self):
        """
        Returns the number of candidate rate laws.

        Returns:
            int: The number of candidate rate laws.

        """
        return len(self._ratelaws)

    def __getitem__(self, position):
        """
        Returns the candidate rate law at the specified position.

        Args:
            position (int): The position of the candidate rate law.

        Returns:
            RateLaw: The candidate rate law at the specified position.

        """

        return self._ratelaws[position]

    def __repr__(self) -> str:
        """
        Returns a string representation of the CandidateRateLaws object.

        Returns:
            str: The string representation of the CandidateRateLaws object.

        """
        return "[" + "\n".join(list(map(str, self._ratelaws))) + "]"

    def append(self, rl: RateLaw) -> None:
        """
        Appends a RateLaw object to the list of candidate rate laws.

        Args:
            rl (RateLaw): The RateLaw object to append.

        Returns:
            None

        """
        self._ratelaws.append(rl)

    @property
    def ratelaws(self):
        """
        Returns a list of rate law functions.

        Returns:
            list: The list of rate law functions.

        """
        rl = [self._ratelaws[i].function for i in range(len(self._ratelaws))]
        return rl
