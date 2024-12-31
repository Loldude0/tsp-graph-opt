#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

using namespace std;

struct Tour {
    vector<int> path;
    double fitness;
};

double cost(const vector<int> &tour, const vector<double> &graph, int n) {

    double len = 0;

    for (int i = 0; i < n; ++i) {

        int j = tour[i];
        int k = tour[(i + 1) % n];

        len += graph[j * n + k];
    }

    return len;
}

vector<Tour> init(int pop_size, int n) {

    vector<Tour> population(pop_size);

    for (int i = 0; i < pop_size; ++i) {

        population[i].path.resize(n);

        for (int j = 0; j < n; ++j) {
            population[i].path[j] = j;
        }

        shuffle(population[i].path.begin(), population[i].path.end(), mt19937{random_device{}()});
    }

    return population;
}

Tour tournament_selection(const vector<Tour> &population, int size) {

    static mt19937 gen{random_device{}()};
    uniform_int_distribution<> dis(0, population.size() - 1);

    Tour best = population[dis(gen)];

    for (int i = 1; i < size; ++i) {

        Tour contestant = population[dis(gen)];

        if (contestant.fitness < best.fitness) {
            best = contestant;
        }

    }
    return best;
}

pair<Tour, Tour> pmx(const Tour &parent1, const Tour &parent2) {

    int n = parent1.path.size();

    static mt19937 gen{random_device{}()};
    uniform_int_distribution<> dis(0, n - 1);

    Tour child1, child2;

    child1.path.resize(n, -1);
    child2.path.resize(n, -1);

    int start = dis(gen);
    int end = dis(gen);
    if (start > end) swap(start, end);

    vector<int> mapping1(n, -1), mapping2(n, -1);

    for (int i = start; i <= end; ++i) {

        child1.path[i] = parent2.path[i];
        child2.path[i] = parent1.path[i];

        mapping1[parent2.path[i]] = parent1.path[i];
        mapping2[parent1.path[i]] = parent2.path[i];
    }

    for (int i = 0; i < n; ++i) {

        if (i < start || i > end) {

            int val1 = parent1.path[i];
            int val2 = parent2.path[i];

            while (mapping1[val1] != -1) val1 = mapping1[val1];
            while (mapping2[val2] != -1) val2 = mapping2[val2];

            child1.path[i] = val1;
            child2.path[i] = val2;
        }
    }

    return {child1, child2};
}

void displacement_mutation(Tour &tour) {

    int n = tour.path.size();
    if (n <= 1) return;

    static mt19937 gen{random_device{}()};
    uniform_int_distribution<> dis(0, n - 1);

    int start = dis(gen);
    int length = 1 + dis(gen) % (n - start); 
    int insert = dis(gen);

    vector<int> subtour(tour.path.begin() + start, tour.path.begin() + start + length);

    tour.path.erase(tour.path.begin() + start, tour.path.begin() + start + length); //remove sub

    if (insert > tour.path.size()) insert = tour.path.size();

    tour.path.insert(tour.path.begin() + insert, subtour.begin(), subtour.end()); //add sub
}

void exchange_mutation(Tour &tour) {

    int n = tour.path.size();
    if (n <= 1) return;

    static mt19937 gen{random_device{}()};
    uniform_int_distribution<> dis(0, n - 1);

    int i = dis(gen);
    int j = dis(gen);

    if (i != j) swap(tour.path[i], tour.path[j]);
}

pair<vector<int>, double> genetic_algorithm(const vector<double> &graph, int n, int pop_size, int generations,
                                            int tournament_size, double crossover_prob,
                                            double mutation_rate, int &nodes_expanded) {

    static mt19937 gen{random_device{}()};
    uniform_real_distribution<> dis(0.0, 1.0);
    nodes_expanded = 0;

    vector<Tour> population = init(pop_size, n);

    for (auto &tour : population) tour.fitness = cost(tour.path, graph, n);

    Tour best_tour = *min_element(population.begin(), population.end(), [](const Tour &a, const Tour &b) { return a.fitness < b.fitness; });

    for (int generation = 0; generation < generations; ++generation) {
        
        vector<Tour> new_population;

        while (new_population.size() < pop_size) {

            Tour child1, child2;
            Tour parent1 = tournament_selection(population, tournament_size);
            Tour parent2 = tournament_selection(population, tournament_size);

            if (dis(gen) < crossover_prob) {
                tie(child1, child2) = pmx(parent1, parent2);
            } else {
                child1 = parent1;
                child2 = parent2;
            }

            if (dis(gen) < mutation_rate) displacement_mutation(child1);
            if (dis(gen) < mutation_rate) exchange_mutation(child2);

            child1.fitness = cost(child1.path, graph, n);
            child2.fitness = cost(child2.path, graph, n);

            new_population.push_back(child1);

            if (new_population.size() < pop_size) new_population.push_back(child2);

            nodes_expanded += 2;
        }

        population = move(new_population);

        Tour current_best_tour = *min_element(population.begin(), population.end(), [](const Tour &a, const Tour &b) { return a.fitness < b.fitness; });

        if (current_best_tour.fitness < best_tour.fitness) best_tour = current_best_tour;

    }

    return {best_tour.path, best_tour.fitness};
}

pair<vector<double>, int> read_graph(const string &filename) {
    ifstream file(filename);
    string line;
    getline(file, line);
    int n = stoi(line);

    vector<double> graph(n * n);
    for (int i = 0; i < n; ++i) {
        getline(file, line);
        istringstream iss(line);
        for (int j = 0; j < n; ++j) {
            iss >> graph[i * n + j];
        }
    }
    return {graph, n};
}

int main() {
    vector<int> sizes = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
    ofstream outfile("q3_genetic.csv");
    outfile << "size,index,cost,nodes,nodes_expanded,real_time,cpu_time" << endl;

    int siz = 0;
    auto total = chrono::high_resolution_clock::now();

    int pop = 500;
    int gen = 700;
    int tournament = 3;
    double crossover_prob = 0.95;
    double mut = 0.4;

    int total_cost = 0;

    for (int index = 0; index < 300; ++index) {
        int size = sizes[siz];
        string filename = "square_graph/graph_size_" + to_string(size) + "_index_" + to_string(index) + ".txt";
        auto [graph, n] = read_graph(filename);

        auto start_real = chrono::high_resolution_clock::now();
        clock_t start_cpu = clock();

        int nodes_expanded;

        auto [path, cost] = genetic_algorithm(graph, n, pop, gen, tournament, crossover_prob, mut, nodes_expanded);

        total_cost += cost;

        clock_t end_cpu = clock();
        auto end_real = chrono::high_resolution_clock::now();
        double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC;
        double real_time = chrono::duration<double>(end_real - start_real).count();

        outfile << size << "," << index << "," << cost << "," << path.size() << ","
                << nodes_expanded << "," << real_time << "," << cpu_time << endl;

        printf("Size: %d, Index: %d, Cost: %f, Nodes: %d, Nodes Expanded: %d, Real Time: %f, CPU Time: %f\n",
               size, index, cost, (int)path.size(), nodes_expanded, real_time, cpu_time);

        if (index % 30 == 29)
            ++siz;
    }

    auto end_total = chrono::high_resolution_clock::now();

    double total_time = chrono::duration<double>(end_total - total).count();

    printf("Total time: %f\n", total_time);
    printf("Total cost: %d\n", total_cost);
    outfile.close();

    return 0;
}