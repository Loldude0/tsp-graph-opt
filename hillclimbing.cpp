#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <chrono>
#include <fstream>
#include <tuple>
#include <iostream>
#include <random>
#include <vector>
#include <sstream>
#include <cstring>

using namespace std;

random_device rd;
mt19937 gen(rd());

tuple<vector<int>, double, int> hill_climbing(const vector<double>& graph, int max_iter, int r, int n) {

    vector<int> best_path;
    double best_cost = numeric_limits<double>::max();

    int total_nodes_expanded = 0;

    vector<int> current_path(n);

    for (int restart = 0; restart < r; ++restart) {

        iota(current_path.begin(), current_path.end(), 0);
        shuffle(current_path.begin(), current_path.end(), gen);

        double current_cost = 0;

        for (int i = 0; i < n; ++i) current_cost += graph[current_path[i] * n + current_path[(i + 1) % n]];

        int nodes_expanded = 0;

        for (int iteration = 0; iteration < max_iter; ++iteration) {

            bool improved = false;

            for (int i = 0; i < n - 1; ++i) {
                for (int j = i + 1; j < n; ++j) {

                    nodes_expanded++;

                    swap(current_path[i], current_path[j]);

                    double new_cost = 0;

                    for (int k = 0; k < n; ++k) new_cost += graph[current_path[k] * n + current_path[(k + 1) % n]];

                    if (new_cost < current_cost) {

                        current_cost = new_cost;
                        improved = true;
                        break;

                    } else swap(current_path[i], current_path[j]);

                }
                if (improved) break;
            }
            if (!improved) break;
        }

        if (current_cost < best_cost) {
            best_path = current_path;
            best_cost = current_cost;
        }

        total_nodes_expanded += nodes_expanded;
    }

    return make_tuple(best_path, best_cost, total_nodes_expanded);
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
    ofstream outfile("q3_hillclimb.csv");
    outfile << "size, index, cost, nodes, nodes_expanded, real_time, cpu_time" << endl;
    int siz = 0;

    int max_iter = 1000000;
    int r = 500;

    auto total = chrono::high_resolution_clock::now();

    for (int index = 0; index < 300; ++index) {
        int size = sizes[siz];
        string filename = "square_graph/graph_size_" + to_string(size) + "_index_" + to_string(index) + ".txt";
        auto [graph, n] = read_graph(filename);
        auto start_real = chrono::high_resolution_clock::now();
        clock_t start_cpu = clock();
        
        auto [path, cost, nodes_expanded] = hill_climbing(graph, max_iter, r, n);
        
        clock_t end_cpu = clock();
        auto end_real = chrono::high_resolution_clock::now();

        double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC;
        double real_time = chrono::duration<double>(end_real - start_real).count();

        outfile << size << ", " << index << ", " << cost << ", " << path.size() << ", " << nodes_expanded << ", " << real_time << ", " << cpu_time << endl;

        printf("Size: %d, Index: %d, Cost: %f, Nodes: %d, Nodes Expanded: %d, Real Time: %f, CPU Time: %f\n", size, index, cost, path.size(), nodes_expanded, real_time, cpu_time);

        if (index % 30 == 29) ++siz;
    }

    auto end_total = chrono::high_resolution_clock::now();

    double total_time = chrono::duration<double>(end_total - total).count();

    printf("Total time: %f\n", total_time);
    outfile.close();
    return 0;
}