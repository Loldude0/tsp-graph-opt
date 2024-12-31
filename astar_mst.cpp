#include <algorithm>
#include <bitset>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <queue>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

const size_t MAX_CACHE_SIZE = 500000000;
const double OPTIMAL_PATH_LENGTH = 4.1;

struct State {
    vector<int> path;
    double g, h;
    bool operator<(const State &other) const {
        return g + h > other.g + other.h;
    }
};

struct StateHash {
    size_t operator()(const State &state) const {
        size_t hash = 0;
        for (int city : state.path) {
            hash = hash * 31 + city;
        }
        return hash;
    }
};

struct StateEqual {
    bool operator()(const State &lhs, const State &rhs) const {
        return lhs.path == rhs.path;
    }
};

void add_to_cache(unordered_map<size_t, double> &cache, size_t key, double value) {
    if (cache.size() >= MAX_CACHE_SIZE) cache.erase(cache.begin());
    cache[key] = value;
}

double prims(const vector<double> &graph, int n, const vector<bool> &global_visited, unordered_map<size_t, double> &heuristic_cache) {
    size_t visited_hash = 0;

    for (bool b : global_visited) visited_hash = (visited_hash << 1) | b;

    if (heuristic_cache.find(visited_hash) != heuristic_cache.end()) return heuristic_cache[visited_hash];

    bitset<50> visited_bits;
    for (int i = 0; i < n; ++i) visited_bits[i] = global_visited[i];

    int unvisited_count = n - visited_bits.count();
    if (unvisited_count <= 1) {
        add_to_cache(heuristic_cache, visited_hash, 0);
        return 0;
    }

    vector<double> key(n, INFINITY);
    vector<int> parent(n, -1);
    int start = 0;
    while (visited_bits[start]) ++start;
    key[start] = 0;

    double mst_cost = 0;

    for (int i = 0; i < unvisited_count; ++i) {
        int u = -1;
        for (int v = 0; v < n; ++v) {
            if (!visited_bits[v] && (u == -1 || key[v] < key[u])) u = v;
        }

        if (u == -1) break;

        visited_bits[u] = true;
        mst_cost += key[u];

        //TODo: AVX
        __m256d key_vec = _mm256_loadu_pd(&key[0]); //load avx reg
        __m256d graph_vec = _mm256_loadu_pd(&graph[u * n]);
        __m256d mask = _mm256_cmp_pd(graph_vec, key_vec, _CMP_LT_OQ); //avx cmp
        key_vec = _mm256_blendv_pd(key_vec, graph_vec, mask); //avx blend
        _mm256_storeu_pd(&key[0], key_vec); //back to memory

        for (int v = 0; v < n; ++v) {
            if (!visited_bits[v] && graph[u * n + v] < key[v]) {
                key[v] = graph[u * n + v];
            }
        }
    }

    add_to_cache(heuristic_cache, visited_hash, mst_cost);
    return mst_cost;
}

std::tuple<std::vector<int>, double, int> astar_mst(const vector<double> &graph, int n) {
    priority_queue<State> pq;
    unordered_map<size_t, double> heuristic_cache;
    unordered_map<State, double, StateHash, StateEqual> visited_states;
    pq.push({{0}, 0, prims(graph, n, vector<bool>(n, false), heuristic_cache)});

    int nodes_expanded = 0;
    double best_cost = INFINITY;

    while (!pq.empty()) {
        State current = pq.top();
        pq.pop();

        nodes_expanded++;

        if (current.g + current.h >= best_cost) continue;

        if (current.g >= OPTIMAL_PATH_LENGTH) continue;

        if (current.path.size() == n) {
            current.path.push_back(0);
            current.g += graph[current.path[n - 1] * n + 0];
            best_cost = current.g;
            return {current.path, current.g, nodes_expanded};
        }

        vector<bool> visited(n, false);

        for (int city : current.path) visited[city] = true;

        for (int next = 0; next < n; ++next) {

            if (visited[next]) continue;

            vector<int> new_path = current.path;

            new_path.push_back(next);

            double new_g = current.g + graph[current.path.back() * n + next];

            if (new_g + current.h >= OPTIMAL_PATH_LENGTH) continue;

            visited[next] = true;

            double new_h = prims(graph, n, visited, heuristic_cache);

            State new_state = {move(new_path), new_g, new_h};

            if (visited_states.find(new_state) != visited_states.end() && visited_states[new_state] <= new_g) continue;


            visited_states[new_state] = new_g;
            pq.push(new_state);
        }
    }

    return {{}, -1, 0};
}

vector<double> read_graph(const string &filename) {
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
    return graph;
}

int main() {
    vector<int> sizes = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
    ofstream outfile("q2_results_optimized_2.txt");
    auto total = chrono::high_resolution_clock::now();
    int siz = 0;
    for (int index = 0; index < 300; ++index) {
        int size = sizes[siz];

        string filename = "square_graph/graph_size_" + to_string(size) + "_index_" + to_string(index) + ".txt";
        auto graph = read_graph(filename);

        auto start_real = chrono::high_resolution_clock::now();
        clock_t start_cpu = clock();

        auto [path, cost, nodes_expanded] = astar_mst(graph, size);

        clock_t end_cpu = clock();
        auto end_real = chrono::high_resolution_clock::now();

        double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC;
        double real_time = chrono::duration<double>(end_real - start_real).count();

        outfile << "size: " << size << ", index: " << index << ", cost: " << cost
                << ", nodes: " << path.size() << ", nodes_expanded: " << nodes_expanded
                << ", real_time: " << real_time << "s, cpu_time: " << cpu_time << "s" << endl;

        if (index % 30 == 29)
            ++siz;
    }

    auto end_total = chrono::high_resolution_clock::now();

    double total_time = chrono::duration<double>(end_total - total).count();

    printf("Total time: %f\n", total_time);
    outfile.close();
    return 0;
}