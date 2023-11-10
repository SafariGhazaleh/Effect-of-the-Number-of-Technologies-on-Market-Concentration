#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:26:18 2023

@author: Ghazaleh Safari

International Master- and PhD program in Mathematics 
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(123)
class Agent:
    def __init__(self, S, id_number, choice_function_exponent):
        """
        Constructor method.

        Parameters
        ----------
        S : Simulation object
            The simulation the agent belongs to.
        id_number : int
            Unique ID number of the agent.
        choice_function_exponent : numeric, optional
            Exponent of the Generalized Eggenberger-Polya process choice 
            function. Values >1 will lead to winner-take-all dynamics, Values 
            <1 lead to equalization dynamics. The default is 2.

        Returns
        -------
        None.

        """
        self.id_number = id_number
        self.Simulation = S
        self.technology = None
        self.choice_function_exponent = choice_function_exponent

    def choose(self):
        """
        Method for choosing a technology to adopt.

        Returns
        -------
        int or None
            Previous technology.
        int or None
            New technology.

        """
        """ Obtain distribution of technologies used by direct neighbors"""
        neighbors = self.get_neighbors()
        tech_list = self.Simulation.get_technologies_list()
        tech_frequency = {tech: 0 for tech in tech_list}
        for A in neighbors:
            tech = A.get_technology()
            if tech is not None:
                tech_frequency[tech] += 1

        """ Compute choice probabilities based on the distribution in the 
            immediate neighborhood. The form of the transformation may tend to 
            the technology used by the majority (if self.choice_function_exponent > 1)
            or overrepresent to those used by the minority (if 
            self.choice_function_exponent < 1)"""
        tech_probability = [tech_frequency[tech] ** self.choice_function_exponent for tech in tech_list]
        if np.sum(tech_probability) > 0:
            """ Select and adopt a technology"""
            tech_probability = np.asarray(tech_probability) / np.sum(tech_probability)
            old_tech = self.technology
            self.technology = np.random.choice(tech_list, p=tech_probability)
            """ Report the change back"""
            return old_tech, self.technology
        else:
            """ Report that no change was possible"""
            return None, None

    def get_technology(self):
        """
        Getter method for the technology the agent uses.

        Returns
        -------
        int
            Current technology. The technologies are characterized as ints.

        """
        return self.technology

    def set_technology(self, tech):
        """
        Setter method for the technology the agent uses.

        Parameters
        ----------
        tech : int
            New technology the agent should adopt. The technologies are 
            characterized as ints.

        Returns
        -------
        None.

        """
        self.technology = tech

    def get_neighbors(self):
        """
        Method for returning a list of neighbor agents

        Returns
        -------
        List of Agent objects:
            List of Agents that are direct neighbors
        """
        return [self.Simulation.G.nodes[N]["agent"] for N in nx.neighbors(self.Simulation.G, self.id_number)]


class Simulation():
    def __init__(self,
                 n_agents=1000,
                 n_technologies=3,
                 n_initial_adopters=2,
                 reconsideration_probability=0.2,
                 choice_function_exponent=2,
                 network_type="Erdos-Renyi"):
        """
        Constructor method.

        Parameters
        ----------
        n_agents : int, optional
            Number of agents. The default is 1000.
        n_technologies : int, optional
            Number of technologies. The default is 3.
        n_initial_adopters : int, optional
            Number of initial adopters of each technology. The default is 2.
        reconsideration_probability : float, optional
            Probability for agents that have already chosen to reconsider their 
            choice when given the chance. The default is 0.2.
        choice_function_exponent : numeric, optional
            Exponent of the Generalized Eggenberger-Polya process choice 
            function. Values >1 will lead to winner-take-all dynamics, Values 
            <1 lead to equalization dynamics. The default is 2.
        network_type : str, optional
            Network type. Can be Erdos-Renyi, Barabasi-Albert, or Watts-Strogatz. 
            The default is "Erdos-Renyi".

        Returns
        -------
        None.

        """
        """ Define parameters"""
        self.n_agents = n_agents
        self.t_max = 200
        self.n_technologies = n_technologies
        self.n_initial_adopters = n_initial_adopters
        self.reconsideration_probability = reconsideration_probability
        self.choice_function_exponent = choice_function_exponent

        """ Prepare technology list"""
        self.technologies_list = list(range(self.n_technologies))
        """ Prepare technology frequency dict. Each technology initialized with
            number zero."""
        self.tech_frequency = {tech: 0 for tech in self.technologies_list}

        """ Generate network"""
        if network_type == "Erdos-Renyi":
            self.G = nx.erdos_renyi_graph(n=self.n_agents, p=0.1)
        elif network_type == "Barabasi-Albert":
            self.G = nx.barabasi_albert_graph(n=self.n_agents, m=40)
        elif network_type == "Watts-Strogatz":
            self.G = nx.connected_watts_strogatz_graph(n=self.n_agents, k=40, p=0.15)
        else:
            assert False, "Unknown network type {:s}".format(network_type)

        """ Create agents and place them on the network"""
        self.agents_list = []

        for i in range(self.G.order()):
            A = Agent(self, i, self.choice_function_exponent)
            self.agents_list.append(A)
            self.G.nodes[i]["agent"] = A

        """ Seed technologies in random agents"""
        for tech in self.technologies_list:
            n_early_adopters = self.n_technologies * self.n_initial_adopters
            early_adopters = list(np.random.choice(self.agents_list,
                                                   replace=False,
                                                   size=n_early_adopters))
            for i in range(self.n_technologies):
                for j in range(self.n_initial_adopters):
                    A = early_adopters.pop()
                    A.set_technology(self.technologies_list[i])
                self.tech_frequency[i] += self.n_initial_adopters

        """ Prepare history variables and record initial values"""
        self.history_tech_frequency = {tech: [self.tech_frequency[tech] / self.n_agents]
                                       for tech in self.technologies_list}
        self.history_t = [0]

    def run(self):
        """
        Run method. Governs the course of the simulation.

        Returns
        -------
        None.

        """
        """ Time iteration"""
        for t in range(0, self.t_max + 1):
            """ Select one agent in each time step"""
            A = np.random.choice(self.agents_list)
            """ The agent will choose a technology if they have none, otherwise
                they may reconsider depending on self.reconsideration_probability"""
            tech = A.get_technology()
            if (tech is None) or (np.random.random() < self.reconsideration_probability):
                old, new = A.choose()
                if old is not None:
                    self.tech_frequency[old] -= 1
                if new is not None:
                    self.tech_frequency[new] += 1
            """ Record current state"""
            for i in range(self.n_technologies):
                self.history_tech_frequency[i].append(self.tech_frequency[i] / self.n_agents)
            self.history_t.append(t)

    def get_technologies_list(self):
        """
        Getter method for technologies list

        Returns
        -------
        list of int
            List of technologies. Each technology is identified as an int.

        """
        return self.technologies_list

    def calculate_hhi(self):
        """
        Calculate the Herfindahl-Hirschman Index (HHI) to measure concentration.

        Returns
        -------
        float
            HHI value.

        """
        market_shares = np.array(list(self.tech_frequency.values())) / self.n_agents
        hhi = np.sum(market_shares ** 2)
        return hhi

    def return_results(self):
        """
        Method for returning and visualizing results

        Returns
        -------
        simulation_history : dict
            Recorded data on the simulation run.

        """

        """ Prepare return dict"""
        simulation_history = {"history_t": self.history_t,
                              "history_tech_frequency": self.history_tech_frequency}

        """ Create figure showing the development of usage shares of the 
            technologies"""
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
        for tech in self.history_tech_frequency.keys():
            ax[0][0].plot(self.history_t, self.history_tech_frequency[tech], label="Technology " + str(tech))
        ax[0][0].set_ylim(0, 1)
        ax[0][0].set_xlim(0, self.t_max + 1)
        ax[0][0].set_ylabel("Frequency")
        ax[0][0].set_xlabel("Time")
        ax[0][0].legend()

        """ Save (as pdf) and show figure"""
        plt.tight_layout()
        plt.savefig("technology_choice_simulation.pdf")
        plt.show()

        return simulation_history


class Experiment:
    def __init__(self, num_simulations=5):
        """
        Experiment class initialization.

        Parameters
        ----------
        num_simulations : int, optional
            Number of simulations to run. The default is 5.

        Returns
        -------
        None.

        """
        self.num_simulations = num_simulations

    def run(self):
        num_technologies_range = [2, 3, 4, 5]
        hhi_values = []

        for num_technologies in num_technologies_range:
            hhi_values_per_num_tech = []

            for _ in range(self.num_simulations):
                simulation = Simulation(n_technologies=num_technologies)
                simulation.run()
                hhi = simulation.calculate_hhi()
                hhi_values_per_num_tech.append(hhi)

            hhi_values.append(hhi_values_per_num_tech)

        return hhi_values

    def analyze_results(self, hhi_values):
        num_technologies_range = [2, 3, 4, 5]

        for i, num_technologies in enumerate(num_technologies_range):
            hhi_mean = np.mean(hhi_values[i])
            hhi_std = np.std(hhi_values[i])

            print("For {:d} technologies: Mean HHI = {:.3f}, Standard Deviation HHI = {:.3f}"
                  .format(num_technologies, hhi_mean, hhi_std))

if __name__ == '__main__':
    experiment = Experiment(num_simulations=10)
    hhi_values = experiment.run()
    experiment.analyze_results(hhi_values)
