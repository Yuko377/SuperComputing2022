#include <cmath>
#include <mpi.h>
#include <iostream>
#include <vector>


double GetScalarProduct(const std::vector<double> & v1, const std::vector<double> & v2)
{
    const auto size = v1.size();

    double result = 0;
    for (int i = 0; i < size; i++)
    {
        result += v1[i] * v2[i];
    }

    return result;
}

void SetTest(
    std::vector<double> & a, 
    std::vector<double> & b, 
    std::vector<double> & x, 
    int displ, int localSize, int taskSize
)
{
    for (int i = 0; i < taskSize; i++)
    {
        for (int j = 0; j < localSize; j++)
        {
            if (displ + j == i)
            {
                a[i * localSize + j] = 2.0;
            } else
            {
                a[i * localSize + j ] = 1.0;
            }
        }
    }

    for (int i = 0; i < localSize; ++i)
    {
        b[i] = taskSize + 1;
        x[i] = rand() % 15;
    }
}

void Gemv(
    const std::vector<double> & mat,
    const std::vector<double> & vec,
    std::vector<double> & result
)
{
    int m = vec.size();
    int n = mat.size() / m;

    for (int i = 0; i < n; i++)
    {
        result[i] = 0;

        for (int j = 0; j < m; j++)
        {
            result[i] += mat[i * m + j] * vec[j];
        }
    }
}

void PrintVector(const std::vector<double> & v)
{
    const int size = v.size();

    for (int i = 0; i < size; ++i)
    {
        std::cout << v[i] << " ";
    }

    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    const int taskSize = 100;
    const double epsilon = 1e-9;
    const int maxIterations = 1000000000;

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start = MPI_Wtime();

    std::vector<int> localSizes(size, 0.0);
    std::vector<int> displs(size, 0.0);

    for (int i = 0; i < size; i++)
    {
        localSizes[i] = taskSize / size + ((i < taskSize % size) ? 1 : 0);
    }

    for (int i = 1; i < size; i++)
    {
        displs[i] = displs[i - 1] + localSizes[i - 1];
    }

    int localSize = localSizes[rank];

    std::vector<double> bLocal(localSize, 0.0);
    std::vector<double> xLocal(localSize, 0.0);
    std::vector<double> aLocal(taskSize * localSize);
    std::vector<double> yLocal(localSize, 0.0);
    std::vector<double> axLocal(taskSize, 0.0);
    std::vector<double> ax(taskSize, 0.0);
    std::vector<double> ay(taskSize, 0.0);
    std::vector<double> ayLocal(taskSize, 0.0);

    SetTest(aLocal, bLocal, xLocal, displs[rank], localSize, taskSize);

    double bNormLocal = GetScalarProduct(bLocal, bLocal);
    double normB = 0;
    MPI_Allreduce(&bNormLocal, &normB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double cond = 100;
    int iterations = 0;

    while (cond > epsilon && iterations++ < maxIterations)
    {
        Gemv(aLocal, xLocal, axLocal);
        MPI_Allreduce(axLocal.data(), ax.data(), taskSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int i = 0; i < localSize; i++)
        {
            yLocal[i] = ax[i + displs[rank]] - bLocal[i];
        }

        Gemv(aLocal, yLocal, ayLocal);
        MPI_Allreduce(ayLocal.data(), ay.data(), taskSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double localNumerator = 0.0;
        double localDenominator = 0.0;

        for (int i = 0; i < localSize; ++i)
        {
            localNumerator += yLocal[i ] * ay[i + displs[rank]];
            localDenominator += ay[i + displs[rank]] * ay[i + displs[rank]];
        }

        double numerator = 0;
        double denominator = 0;

        MPI_Allreduce(&localNumerator, &numerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localDenominator, &denominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double tetta = numerator / denominator;

        for (int i = 0; i < localSize; i++)
        {
            xLocal[i] -= tetta * yLocal[i];
        }

        double yNormLocal = GetScalarProduct(yLocal, yLocal);

        double yNorm = 0;
        MPI_Allreduce(&yNormLocal, &yNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        cond = yNorm / normB;
    }

    std::vector<double> x(taskSize, 0.0);
    MPI_Allgatherv(xLocal.data(), localSize, MPI_DOUBLE, x.data(), localSizes.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Elapsed time: " << MPI_Wtime() - start << std::endl;
        PrintVector(x);
    }

    MPI_Finalize();
}
