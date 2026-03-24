#include <stdio.h>

// Simple function
double add(double a, double b) {
    return a + b;
}

// Example with pointer (more realistic for you)
void scale_array(double* arr, int n, double factor) {
    for (int i = 0; i < n; i++) {
        arr[i] *= factor;
    }
}

