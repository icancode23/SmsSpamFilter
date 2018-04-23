#include<stdio.h>
#include<stdbool.h>
#define N 4
void printSol(int bd[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf(" %d ", bd[i][j]);
        printf("\n");
    }
}
bool isPlaceable(int bd[N][N], int r, int c)
{
    int i, j;
    for (i = 0; i < c; i++)
        if (bd[r][i])
            return false;
    for (i=r, j=c; i>=0 && j>=0; i--, j--)
        if (bd[i][j])
            return false;
    for (i=r, j=c; j>=0 && i<N; i++, j--)
        if (bd[i][j])
            return false;
    return true;
}
bool solveNQueen(int bd[N][N], int c)
{
    if (c >= N)
        return true;
    for (int i = 0; i < N; i++)
    {
        if ( isPlaceable(bd, i, c) )
        {
            bd[i][c] = 1;
            if ( solveNQueen(bd, c + 1) )
                return true;
            bd[i][c] = 0;
        }
    }
    return false;
}
bool solveNQ()
{
    int bd[N][N] = { {0, 0, 0, 0},
                        {0, 0, 0, 0},
                        {0, 0, 0, 0},
                        {0, 0, 0, 0}};
    if ( solveNQueen(bd, 0) == false )
    {
        printf("Solution does not exist");
        return false;
    }
    printSol(bd);
    return true;
}
int main()
{
    printf("The Solution is:");
    solveNQ();
    return 0;
}
