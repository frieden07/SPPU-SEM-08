#include <bits/stdc++.h>
#include <omp.h>

using namespace std;


//  class for DFS and BFS implementation
class Graph {
    int V;
    //  declare a graph adjacnecy list
    vector<vector<int>> adj;

    public:
    // create a constructor
    Graph(int V) : V(V), adj(V) {}

    //  add data to adjacnecy list
    void addEdge(int v, int w) {
        adj[v].push_back(w);
    }

    // function for sequnetial BFS implementation
    void sequentialBFS(int startVertex) {
        // create a visited array and queue
        vector<bool> visited(V, false);
        queue<int> q;

        //  make start node as visited and push into queue
        visited[startVertex] = true;
        q.push(startVertex);

        //  run until queue is empty
        while (!q.empty()) {
            //  pop first element from queue and store it in variable
            int v = q.front();
            q.pop();
            cout << v << " ";

            // enqueue unvisted neighbour of v in queue
            for (int i = 0; i < adj[v].size(); ++i) {
                int n = adj[v][i];
                if (!visited[n]) {
                    visited[n] = true;
                    q.push(n);
                }
            }
        }
    }

    // function for sequnetial DFS implementation
    void sequentialDFS(int startVertex) {
        // create a visited array and stack
        vector<bool> visited(V, false);
        stack<int> s;

        //  push start node into stack
        s.push(startVertex);

        //  run until stack is empty
        while (!s.empty()) {
            //  pop first element from stack and store it in variable
            int v = s.top();
            s.pop();

            if (!visited[v]){
                cout << v << " ";
                visited[v] = true;
            }

            // enqueue unvisted neighbour of v in stack
            for (int i = 0; i < adj[v].size(); ++i) {
                int n = adj[v][i];
                if (!visited[n]) {
                    s.push(n);
                }
            }
        }
    }

    // function for parallel BFS implementation
    void parallelBFS(int startVertex) {
        // create a visited array and queue
        vector<bool> visited(V, false);
        queue<int> q;

        //  make start node as visited and push into queue
        visited[startVertex] = true;
        q.push(startVertex);

        //  run until queue is empty
        while (!q.empty()) {
            //  pop first element from queue and store it in variable
            int v = q.front();
            q.pop();
            cout << v << " ";

            // create a parallel for loop
            #pragma omp parallel for
            // enqueue unvisted neighbour of v in queue
            for (int i = 0; i < adj[v].size(); ++i) {
                int n = adj[v][i];
                #pragma omp critical
                if (!visited[n]) {
                    visited[n] = true;
                    q.push(n);
                }
            }
        }
    }

    // function for parallel DFS implementation
    void parallelDFS(int startVertex) {
         // create a visited array and stack
        vector<bool> visited(V, false);
        stack<int> s;

        //  push start node into stack
        s.push(startVertex);

         //  run until stack is empty
        while (!s.empty()) {
            //  pop first element from stack and store it in variable
            int v = s.top();
            s.pop();

            if (!visited[v]){
                cout << v << " ";
                visited[v] = true;
            }

            // create a parallel for loop
            #pragma omp parallel for
            // enqueue unvisted neighbour of v in stack
            for (int i = 0; i < adj[v].size(); ++i) {
                int n = adj[v][i];
                #pragma omp critical
                if (!visited[n]) {
                    s.push(n);
                }
            }
        }
    }
};


int main() {
    // take number of vertex and edge input from user
    cout << "Create the graph first : \n\nEnter the no. of vertex in graph : ";
    int vertex_count;
    cin>>vertex_count;
    cout << "Enter the no. of edges in graph : ";
    int edge_count;
    cin>>edge_count;

    // create a object of class Graph
    Graph g(vertex_count);

    // create a graph adajancy list
    cout << "Enter the edges connections in graph (v1, v2) : \n";
    for (int i = 0; i < edge_count; ++i) {
        int v1, v2;
        v1 = rand() % vertex_count;
        v2 = rand() % vertex_count;
        g.addEdge(v1, v2);
    }

    double start_time, end_time;

    //  create a loop that ask user for choice to break
    while(true){
        //  show choices
        cout<<"\n\nUser Manual:\n1. Sequential BFS\n2. Sequential DFS\n3. Parallel BFS\n4. Parallel DFS\n5. Exit\nEnter you choice : ";
        int choice;
        //  input choices
        cin>>choice;

        //  condition for those choices, these print call respestive function from respective classes amnd also note the start and end time of execution and show time taken at the end.
        if (choice==1){
            cout << "Depth-First Search (DFS): ";
            start_time = omp_get_wtime();
            g.sequentialDFS(0);
            end_time = omp_get_wtime(); 
            cout << "\nSequential DFS took : " << end_time - start_time << " seconds.\n";
        }
        else if (choice==2){
            cout << "Breadth-First Search (BFS): ";
            start_time = omp_get_wtime();
            g.sequentialBFS(0);
            end_time = omp_get_wtime(); 
            cout << "\nSequential BFS took : " << end_time - start_time << " seconds.\n";
        }
        else if (choice==3){
            cout << "Breadth-First Search (BFS): ";
            start_time = omp_get_wtime();
            g.parallelBFS(0);
            end_time = omp_get_wtime(); 
            cout << "\nParallel BFS took : " << end_time - start_time << " seconds.\n";
        }
        else if (choice==4){
            cout << "Depth-First Search (DFS): ";
            start_time = omp_get_wtime();
            g.parallelDFS(0);
            end_time = omp_get_wtime(); 
            cout << "\nParallel BFS took took : " << end_time - start_time << " seconds.\n";
        }
        else if (choice==5){
            break;
        }
        else{
            cout<<"Invalid Choice"<<endl;
        }
    }

    //  stop the program
    return 0;
}

