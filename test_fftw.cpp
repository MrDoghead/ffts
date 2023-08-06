#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "fftw3.h"
#define N 16

using namespace std::chrono;

void print_complex(fftw_complex* arr)
{
    for (int i = 0; i<N; i++)
    {
        if (arr[i][1] >= 0)
            printf("%f+%fi,\n", arr[i][0], arr[i][1]);
        else
            printf("%f%fi,\n", arr[i][0], arr[i][1]);
    }
}

int main()
{
	fftw_complex *input, *fft_out, *ifft_out;
	fftw_plan fft_plan, ifft_plan;

    // allocate mem
	input = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	fft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    ifft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    // initialize inputs
	if ((input == NULL) || (fft_out == NULL) || (ifft_out == NULL))
	{
		printf("Error:insufficient available memory\n");
	}
	else
	{
		for (int i = 0; i<N; i++)
		{
			input[i][0] = i + 1;
			input[i][1] = i;
		}
	}
    printf("Input:\n");
    print_complex(input);
    printf("\n");

    // fft
    fft_plan = fftw_plan_dft_1d(N, input, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    auto fft_start = system_clock::now();
	fftw_execute(fft_plan); 
    auto fft_end   = system_clock::now();
    auto fft_duration = duration_cast<microseconds>(fft_end - fft_start);
    printf("After FFT:\n");
    print_complex(fft_out);
    printf("time cost: %f us\n", double(fft_duration.count()) * microseconds::period::num / microseconds::period::den);
    printf("\n");

    // ifft
    ifft_plan = fftw_plan_dft_1d(N, fft_out, ifft_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    auto ifft_start = system_clock::now();
    fftw_execute(ifft_plan);
    auto ifft_end   = system_clock::now();
    auto ifft_duration = duration_cast<microseconds>(ifft_end - ifft_start);
    printf("After IFFT:\n");
    print_complex(ifft_out);
    printf("time cost: %f us\n", double(ifft_duration.count()) * microseconds::period::num / microseconds::period::den);

    // clean up
	fftw_destroy_plan(fft_plan);
    fftw_destroy_plan(ifft_plan);
	fftw_cleanup();
	if (input != NULL) fftw_free(input);
	if (fft_out != NULL) fftw_free(fft_out);
    if (ifft_out != NULL) fftw_free(ifft_out);

	return 0;
}

