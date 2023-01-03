#include <iostream>

int main()
{
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            std::cout << std::to_string(i) + ' ' + std::to_string(j) + '\n';

    return 0;
}
