#include <omp.h>
#include <iostream>
using namespace std;

int main() {
	const int size = 100000;
	int *a = new int[size];
	#pragma omp parallel for
	for(int i = 0; i < size; i++) {
		a[i] = i;
	}
	int sum = 0;
	#pragma omp parallel for reduction(+:sum)
	for(int i = 0; i < size; i++) {
		sum += a[i];
	}
	cout << "Sum = " << sum << endl;
	delete[] a;
	return 0;
}
