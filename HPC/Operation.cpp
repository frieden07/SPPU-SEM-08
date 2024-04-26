#include <bits/stdc++.h>
#include<omp.h>

using namespace std;

//  class for sequential operations implementation
class SequentialOperation{
    public:

    //  function for minimum value from array
    int minval(int arr[], int n){
      int minval = arr[0];
        for(int i = 0; i < n; i++){
          if(arr[i] < minval) minval = arr[i];
        }
      return minval;
    }

    //  function for maximum value from array
    int maxval(int arr[], int n){
      int maxval = arr[0];
        for(int i = 0; i < n; i++){
          if(arr[i] > maxval) maxval = arr[i];
        }
      return maxval;
    }

    //  function for sum of array
    int sum(int arr[], int n){
      int sum = 0;
        for(int i = 0; i < n; i++){
          sum += arr[i];
        }
      return sum;
    }

    //  function for average of array
    int average(int arr[], int n){
      return (double)sum(arr, n) / n;
    }
};


//  class for parallel operations implementation
class ParallelOperation{
    public:

    //  function for minimum value from array
    int minval(int arr[], int n){
      int minval = arr[0];
      #pragma omp parallel for reduction(min : minval)
        for(int i = 0; i < n; i++){
          if(arr[i] < minval) minval = arr[i];
        }
      return minval;
    }

    //  function for maximum value from array
    int maxval(int arr[], int n){
      int maxval = arr[0];
      #pragma omp parallel for reduction(max : maxval)
        for(int i = 0; i < n; i++){
          if(arr[i] > maxval) maxval = arr[i];
        }
      return maxval;
    }

    //  function for sum of array
    int sum(int arr[], int n){
      int sum = 0;
      #pragma omp parallel for reduction(+ : sum)
        for(int i = 0; i < n; i++){
          sum += arr[i];
        }
      return sum;
    }

    //  function for average of array
    int average(int arr[], int n){
      return (double)sum(arr, n) / n;
    }
};

int main(){
  //  take size of array from user
    int n;
    cout<<"\n\nEnter the size of the random number array : ";
    cin>>n;

    //  create four array for four differnt operation
    int arr[n];
    double start_time, end_time;

    //  generate the random number for above arrays
    for(int i = 0; i < n; i++){
        arr[i] = 1 + (rand() % 1000);
    }

     //  show the array starting 50 values
    cout<<"Starting 50 values of generated array : ";
    for(int i = 0; i < 50; i++){
        cout<<arr[i]<<" ";
    }

    //  create object of both SequentialOperation and ParallelOperation classes
    SequentialOperation seqop;
    ParallelOperation paops;

    //  create a loop that ask user for choice to break
    while(true){
          //  show choices
          cout<<"\n\nUser Manual:\n1. Sequential Max\n2. Sequential Min\n3. Sequential Sum\n4. Sequential Average\n5. Parallel Max\n6. Parallel Min\n7. Parallel Sum\n8. Parallel Average\n9. Exit\nEnter you choice : ";
          int choice;

          //  input choices
          cin>>choice;

          //  condition for those choices, these print call respestive function from respective classes amnd also note the start and end time of execution and show time taken at the end.
          if (choice==1){;
              start_time = omp_get_wtime();
              cout << "The Maximum value is: " << seqop.maxval(arr, n) << '\n';
              end_time = omp_get_wtime();     
              cout << "Sequential max operation : " << end_time - start_time << " seconds.\n";
          }
          else if (choice==2){
              start_time = omp_get_wtime();
              cout << "The minimum value is: " << seqop.minval(arr, n) << '\n';
              end_time = omp_get_wtime();     
              cout << "Sequential min operation : " << end_time - start_time << " seconds.\n";
          }
          else if (choice==3){
              start_time = omp_get_wtime();
              cout << "The summation is: " << seqop.sum(arr, n) << '\n';
              end_time = omp_get_wtime();     
              cout << "Sequential sum operation : " << end_time - start_time << " seconds.\n";
          }
          else if (choice==4){
              start_time = omp_get_wtime();
              cout << "The average is: " << seqop.average(arr, n) << '\n';
              end_time = omp_get_wtime();     
              cout << "Sequential avg operation : " << end_time - start_time << " seconds.\n";
          }
          if (choice==5){;
              start_time = omp_get_wtime();
              cout << "The Maximum value is: " << seqop.maxval(arr, n) << '\n';
              end_time = omp_get_wtime();     
              cout << "Parallel max operation : " << end_time - start_time << " seconds.\n";
          }
          else if (choice==6){
              start_time = omp_get_wtime();
              cout << "The minimum value is: " << seqop.minval(arr, n) << '\n';
              end_time = omp_get_wtime();     
              cout << "Parallel min operation : " << end_time - start_time << " seconds.\n";
          }
          else if (choice==7){
              start_time = omp_get_wtime();
              cout << "The summation is: " << seqop.sum(arr, n) << '\n';
              end_time = omp_get_wtime();     
              cout << "Parallel sum operation : " << end_time - start_time << " seconds.\n";
          }
          else if (choice==8){
              start_time = omp_get_wtime();
              cout << "The average is: " << seqop.average(arr, n) << '\n';
              end_time = omp_get_wtime();     
              cout << "Parallel avg operation : " << end_time - start_time << " seconds.\n";
          }
          else if (choice==9){
              break;
          }
          else{
              cout<<"Invalid Choice"<<endl;
          }
      }

    //  stop the program
    return 0;
}