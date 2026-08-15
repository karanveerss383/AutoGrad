#include "stdio.h"
#include "weight_init.h"
#include "utils.h"


int main(){
  
  Layer L1;
  createLayer(&L1, 1, 1, 6, 9);
  for(int i = 0; i < L1.fan_out; i++){
    printf("Neuron %d\n : ", (i+1));
    for (int j = 0; j < L1.fan_in; j++){
      
      printf("w%d - %f\n", (j+1), L1.n[i].weights[j].data);
    
    }
    printf("b - %f\n", L1.n[i].bias.data);
  }
}
