import numpy as np

from pymoo.model.survival import Survival
from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
from pymoo.model.duplicate import DefaultDuplicateElimination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.model.population import Population
from pymoo.model.selection import Selection
from pymoo.operators.crossover.biased_crossover import BiasedCrossover
from pymoo.operators.mutation.no_mutation import NoMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import SingleObjectiveDisplay, MultiObjectiveDisplay
from pymoo.util.termination.default import (
    SingleObjectiveDefaultTermination,
    MultiObjectiveDefaultTermination,
)
from pymoo.algorithms.so_brkga import EliteBiasedSelection


class CEliteSurvival(Survival):
    def __init__(self, n_elites, eliminate_duplicates=None):
        super().__init__(False)
        self.n_elites = n_elites
        self.eliminate_duplicates = eliminate_duplicates

    def _do(self, problem, pop, n_survive, algorithm=None, **kwargs):

        pop = DefaultDuplicateElimination(func=lambda p: p.get("F")).do(pop)

        if problem.n_obj == 1:
            pop = FitnessSurvival().do(problem, pop, len(pop))
            elites = pop[: self.n_elites]
            non_elites = pop[self.n_elites :]
        else:
            # Only use feasible solutions for NDS and getting the elites
            _feas = pop.get("feasible")[:, 0]
            if _feas.any():
                F = pop.get("F")[_feas]
                I = NonDominatedSorting(method="tree_based_non_dominated_sort").do(
                    F, only_non_dominated_front=True
                )
                elites = pop[_feas][I]
                _I = np.arange(len(pop))
                assert len(_I[_feas][I]) == len(I)
                assert len(_I[_feas][I]) <= len(_feas)
                I = _I[_feas][I]
            else:
                I = NonDominatedSorting(method="tree_based_non_dominated_sort").do(
                    pop.get("F"), only_non_dominated_front=True
                )
                elites = pop[I]

            non_elites = pop[[k for k in range(len(pop)) if k not in I]]

        assert len(elites) + len(non_elites) == len(pop), (len(elites), len(non_elites))
        elites.set("type", ["elite"] * len(elites))
        non_elites.set("type", ["non_elite"] * len(non_elites))

        return pop


class CSCF(GeneticAlgorithm):
    def __init__(
        self,
        n_elites=200,
        n_offsprings=700,
        n_mutants=100,
        bias=0.7,
        sampling=FloatRandomSampling(),
        survival=None,
        display=SingleObjectiveDisplay(),
        eliminate_duplicates=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_elites : int
            Number of elite individuals
        n_offsprings : int
            Number of offsprings to be generated through mating of an elite and a non-elite individual
        n_mutants : int
            Number of mutations to be introduced each generation
        bias : float
            Bias of an offspring inheriting the allele of its elite parent
        eliminate_duplicates : bool or class
            The duplicate elimination is more important if a decoding is used. The duplicate check has to be
            performed on the decoded variable and not on the real values. Therefore, we recommend passing
            a DuplicateElimination object.
            If eliminate_duplicates is simply set to `True`, then duplicates are filtered out whenever the
            objective values are equal.
        """

        super().__init__(
            pop_size=n_elites + n_offsprings + n_mutants,
            n_offsprings=n_offsprings,
            sampling=sampling,
            selection=EliteBiasedSelection(),
            crossover=BiasedCrossover(bias, prob=1.0),
            mutation=NoMutation(),
            survival=CEliteSurvival(n_elites),
            display=display,
            eliminate_duplicates=True,
            **kwargs,
        )

        self.n_elites = n_elites
        self.n_mutants = n_mutants
        self.bias = bias
        # This is overwritten later anyway, so don't mind
        self.default_termination = SingleObjectiveDefaultTermination()

    def _next(self):
        pop = self.pop
        elites = np.where(pop.get("type") == "elite")[0]

        # actually do the mating given the elite selection and biased crossover
        off = self.mating.do(
            self.problem, pop, n_offsprings=self.n_offsprings, algorithm=self
        )

        # create the mutants randomly to fill the population with
        mutants = FloatRandomSampling().do(self.problem, self.n_mutants, algorithm=self)

        # evaluate all the new solutions
        to_evaluate = Population.merge(off, mutants)
        self.evaluator.eval(self.problem, to_evaluate, algorithm=self)

        # finally merge everything together and sort by fitness
        pop = Population.merge(pop[elites], to_evaluate)

        # the do survival selection - set the elites for the next round
        self.pop = self.survival.do(self.problem, pop, len(pop), algorithm=self)
