#include <bits/stdc++.h>
#include<omp.h>

using namespace std;


//  class for bubble sort implementation
class BubbleSort{
    public: 

    //  sequetial bubble sort implementation function
    void bubblesort(int array[], int n){
        for (int i = 0; i < n - 1; i++){
            for (int j = 0; j < n - i - 1; j++){
                if (array[j] > array[j + 1]) swap(array[j], array[j + 1]);
            }
        }
    }
    
    // parallel bubble sort implementation function
    void parallel_bubblesort(int array[], int n) {
        for (int i = 0; i < n; i++){
            int first = i % 2;
            #pragma omp parallel for shared (array, first) num_threads(16)

            for (int j = first; j < n - 1; j += 2)
                if (array[j] > array[j + 1])
                    swap(array[j], array[j + 1]);
        }

        return;
    }
};

//  class for merge sort implementation
class MergeSort{
    public:

    //  code to merge two arrays in sorted order
    void merge(int arr[], int low, int mid, int high) {
        int n1 = mid - low + 1;
        int n2 = high - mid;
        int left[n1];
        int right[n2];
        
        for (int i = 0; i < n1; i++) left[i] = arr[low + i];
        for (int j = 0; j < n2; j++) right[j] = arr[mid + 1 + j];
        
        int i = 0, j = 0, k = low;

        while (i < n1 && j < n2) {
            if (left[i] <= right[j]){
                arr[k] = left[i];
                i++;
            } 
            else{
                arr[k] = right[j];
                j++;
            }
            k++;
        }

        while (i < n1) {
            arr[k] = left[i];
            i++;
            k++;
        }

        while (j < n2) {
            arr[k] = right[j];
            j++;
            k++;
        }
    }

    //  sequetial merge sort implementation function
    void mergeSort(int arr[], int low, int high) {
        if (low < high) {
            int mid = (low + high) / 2;
            mergeSort(arr, low, mid);
            mergeSort(arr, mid + 1, high);
            merge(arr, low, mid, high);
        }
    }

    //  parallel merge sort implementation function
    void parallelMergeSort(int arr[], int low, int high) {
        if (low < high) {
            int mid = (low + high) / 2;

            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    parallelMergeSort(arr, low, mid);
                }

                #pragma omp section
                {
                    parallelMergeSort(arr, mid + 1, high);
                }
            }
            merge(arr, low, mid, high);
        }
    }
};

// function to print the array of size n
void printArray(int arr[], int n){
    for(int i = 0; i < n; i++) cout << arr[i] << " ";
    cout << "\n";
}


int main(){
    //  take size of array from user
    int n;
    cout<<"\n\nEnter the size of the random number array : ";
    cin>>n;

    //  create four array for four differnt operation
    int arr[n];
    int brr[n];
    int crr[n];
    int drr[n];
    double start_time, end_time;

    //  generate the random number for above arrays
    for(int i = 0; i<n; i++){
    		arr[i] = 1 + (rand() % 1000);
    		brr[i]=arr[i];
            crr[i]=arr[i];
            drr[i]=arr[i];
    }
    

    //  create object of both BubbleSort and MergeSort classes
    BubbleSort bub;
    MergeSort mer;


    //  create a loop that ask user for choice to break
    while(true){
        //  show choices
        cout<<"\n\nUser Manual:\n1. Sequential Bubble sort\n2. Sequential Merge sort\n3. Parallel Bubble sort\n4. Parallel Merge sort\n5. Exit\nEnter you choice : ";
        int choice;

        //  input choices
        cin>>choice;

        //  condition for those choices, these print call respestive function from respective classes amnd also note the start and end time of execution and show time taken at the end.
        if (choice==1){;
            cout<<"Unsorted array starting 50 values : ";
            printArray(arr, 50);
            start_time = omp_get_wtime();
            bub.bubblesort(arr, n);
            end_time = omp_get_wtime();     
            cout << "Sequential Bubble Sort took : " << end_time - start_time << " seconds.\n";
            cout<<"Sorted array starting 50 values : ";
            printArray(arr, 50);
        }
        else if (choice==2){
            cout<<"Unsorted array starting 50 values : ";
            printArray(crr, 50);
            start_time = omp_get_wtime(); 
            mer.mergeSort(crr, 0, n - 1);
            end_time = omp_get_wtime(); 
            cout << "Sequential Merge Sort took: " << end_time - start_time << " seconds\n";
            cout<<"Sorted array starting 50 values : ";
            printArray(crr, 50);
        }
        else if (choice==3){
            cout<<"Unsorted array starting 50 values : ";
            printArray(brr, 50);
            start_time = omp_get_wtime();
            bub.parallel_bubblesort(brr, n);
            end_time = omp_get_wtime();     
            cout << "Parallel Bubble Sort took : " << end_time - start_time << " seconds.\n";
            cout<<"Sorted array starting 50 values : ";
            printArray(brr, 50);
        }
        else if (choice==4){
            cout<<"Unsorted array starting 50 values : ";
            printArray(drr, 50);
            start_time = omp_get_wtime(); 
            mer.parallelMergeSort(drr, 0, n - 1);
            end_time = omp_get_wtime(); 
            cout << "Parallel Merge Sort took : " << end_time - start_time << " seconds";
            cout<<"Sorted array starting 50 values : ";
            printArray(drr, 50);
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