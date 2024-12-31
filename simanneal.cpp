#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;

tuple<vector<int>, double, int> simanneal(const vector<double>& graph, int n, double temp, double alpha, int max_iter, int r) {

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0); //for calcing accept

    vector<int> best_path;
    double best_cost = numeric_limits<double>::infinity();

    int total_nodes_expanded = 0;

    for (int re = 0; re < r; ++re) {

        vector<int> curr_path(n);

        for (int i = 0; i < n; ++i) curr_path[i] = i;

        shuffle(curr_path.begin(), curr_path.end(), gen);

        double curr_cost = 0;
        for (int i = 0; i < n; ++i) curr_cost += graph[curr_path[i] * n + curr_path[(i + 1) % n]];

        double temp2 = temp;

        for (int i = 0; i < max_iter; ++i) {

            vector<int> neighbor = curr_path;
            int j = rand() % curr_path.size();
            int k = rand() % curr_path.size();
            swap(neighbor[j], neighbor[k]);

            double neighbor_cost = 0;
            for (int j = 0; j < n; ++j) neighbor_cost += graph[neighbor[j] * n + neighbor[(j + 1) % n]];

            total_nodes_expanded++;

            double delta = neighbor_cost - curr_cost;

            if (delta < 0 || dis(gen) < exp(-delta / temp2)) { //if random less
                curr_path = neighbor;
                curr_cost = neighbor_cost;
            }

            if (curr_cost < best_cost) { //if better
                best_path = curr_path;
                best_cost = curr_cost;
            }

            temp2 *= alpha;
        }
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
    ofstream outfile("q3_simanneal.csv");
    outfile << "size,index,cost,nodes,nodes_expanded,real_time,cpu_time" << endl;
    int siz = 0;

    double temp = 15000;
    double alpha = 0.999;
    int max_iter = 100000;
    int r = 18;
    int total_cost = 0;

    auto total = chrono::high_resolution_clock::now();

    for (int index = 0; index < 300; ++index) {
        int size = sizes[siz];
        string filename = "square_graph/graph_size_" + to_string(size) + "_index_" + to_string(index) + ".txt";
        auto [graph, n] = read_graph(filename);

        auto start_real = chrono::high_resolution_clock::now();
        clock_t start_cpu = clock();
        
        auto [path, cost, nodes_expanded] = simanneal(graph, n, temp, alpha, max_iter, r);
        
        total_cost += cost;

        clock_t end_cpu = clock();
        auto end_real = chrono::high_resolution_clock::now();
        double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC;
        double real_time = chrono::duration<double>(end_real - start_real).count();

        outfile << size << ", " << index << ", " << cost << ", " << path.size() << ", " << nodes_expanded << ", " << real_time << ", " << cpu_time << endl;
        
        printf("Size: %d, Index: %d, Cost: %f, Nodes: %d, Nodes Expanded: %d, Real Time: %f, CPU Time: %f\n", 
               size, index, cost, static_cast<int>(path.size()), nodes_expanded, real_time, cpu_time);

        if (index % 30 == 29) ++siz;
    }

    auto end_total = chrono::high_resolution_clock::now();

    double total_time = chrono::duration<double>(end_total - total).count();

    printf("Total time: %f\n", total_time);
    printf("Total cost: %d\n", total_cost);
    outfile.close();
    return 0;
}