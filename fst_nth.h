#ifndef ASR_DECODER_NTH_H_
#define ASR_DECODER_NTH_H_

#include <stdint.h>

int partition(int32_t* a, int low, int high);
int32_t nth(int32_t *a, int low, int high, int n);

inline 
int partition(int32_t* a, int low, int high) {
  int i = low;
  int j = high;
  int32_t key = a[low];
  int32_t temp;
  while (i < j) {
    while(i < j && a[j] >= key) j--;
    while(i < j && a[i] <= key) i++;
    if (i < j) {
      temp = a[i];
      a[i] = a[j];
      a[j] = temp;
    }
  }
  a[low] = a[i];
  a[i] = key;
  return i;
}

inline
int32_t nth(int32_t *a, int low, int high, int n) {
  int mid = partition(a, low, high);
  if (mid == n) return a[mid];
  return
    (mid < n) ?
    nth(a, mid+1, high, n) :
    nth(a, low, mid-1, n);
}
#endif
