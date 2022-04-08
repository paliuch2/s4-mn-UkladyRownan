// Kamil Paluszewski 180194

#include <math.h>
#include <chrono>
#include <iostream>

using namespace std;

#define f 0 // 3. cyfra indeksu (f)
#define e 1 // 4. cyfra indeksu (e)

#define N 994 

typedef struct Mat {
	double** vals;
	int size;
} Mat;

typedef struct Params {
	// struktura przechowujaca informacje wykorzystywane w metodach (wektor residuum i jego norma
	// czas trwania obliczen i liczba iteracji)
	double* res;
	double norm_res;
	double time;
	int iterations;
} Params;

void InitMatrix(Mat* mat, int size)
{
	mat->size = size;
	mat->vals = (double**)malloc(size * sizeof(double*));

	for (int i = 0; i < size; i++)
	{
		mat->vals[i] = (double*)malloc(size * sizeof(double));
	}
}

void FillMatrix(Mat* mat, double a1, double a2, double a3)
{
	int size = mat->size;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (i == j) { mat->vals[i][j] = a1; }
			else if (i == j - 1) { mat->vals[i][j] = a2; }
			else if (i == j + 1) { mat->vals[i][j] = a2; }
			else if (i == j - 2) { mat->vals[i][j] = a3; }
			else if (i == j + 2) { mat->vals[i][j] = a3; }
			else mat->vals[i][j] = 0;

		}
	}
}

double* CrossProduct(Mat* mat, double* vector)
{
	int size = mat->size;
	double* result = (double*)malloc(size * sizeof(double));

	for (int i = 0; i < size; i++)
	{
		double value = 0.0;

		for (int j = 0; j < size; j++)
		{
			value += mat->vals[i][j] * vector[j];
		}
		result[i] = value;
	}

	return result;
}

void CalculateRes(Mat* mat, double* b, double* x,  Params* p)
{
	free(p->res); // zwolnienie pamieci przed nadpisaniem wektora residuum
	p->res = CrossProduct(mat, x);

	for (int i = 0; i < mat->size; i++)
	{
		p->res[i] -= b[i];
	}
}

double CalculateNormRes(double* res, int length)
{
	double result = 0;

	for (int i = 0; i < length; i++)
	{
		result += pow(res[i], 2);
	}

	result = sqrt(result);
	return result;
}

void Jacobi(Mat* mat, double* b, double* x, Params* params, double epsilon)
{
	params->iterations = 0;
	int size = mat->size;

	double* previous_x = (double*)malloc(size * sizeof(double));
	params->res = (double*)malloc(size * sizeof(double));
	double sum1 = 0.0;
	double sum2 = 0.0;

	clock_t start, end;

	for (int i = 0; i < size; i++)
	{
		previous_x[i] = 1; // ustawiam wektor poprzednich rozwiazan przed 1. iteracja jako wektor jedynek
	}
	start = clock();

	do
	{
		params->iterations++;

		for (int i = 0; i < size; i++)
		{
			sum1 = 0.0;
			sum2 = 0.0;
			for (int j = 0; j <= i - 1; j++)
			{
				sum1 += mat->vals[i][j] * previous_x[j];
			}
			for (int j = i + 1; j < size; j++)
			{
				sum2 += mat->vals[i][j] * previous_x[j];
			}

			x[i] = (b[i] - sum1 - sum2) / mat->vals[i][i];
		}

		for (int i = 0; i < size; i++)
		{
			previous_x[i] = x[i];
		}

		CalculateRes(mat, b,x, params);
		params->norm_res = CalculateNormRes(params->res, size);

	} while (params->norm_res > epsilon);

	end = clock();
	params->time = double(end - start) / double(CLOCKS_PER_SEC);

	free(previous_x);
}

void GaussSeidel(Mat* mat, double* b, double* x, Params* params, double epsilon)
{
	params->iterations = 0;
	int size = mat->size;

	double* previous_x = (double*)malloc(size * sizeof(double));
	double sum1 = 0.0;
	double sum2 = 0.0;

	clock_t start, end;

	for (int i = 0; i < size; i++)
	{
		previous_x[i] = 1;// ustawiam wektor poprzednich rozwiazan przed 1. iteracja jako wektor jedynek
	}
	start = clock();

	do
	{
		params->iterations++;

		for (int i = 0; i < size; i++)
		{
			sum1 = 0.0;
			sum2 = 0.0;
			for (int j = 0; j <= i - 1; j++)
			{
				sum1 += mat->vals[i][j] * x[j];
			}
			for (int j = i + 1; j < size; j++)
			{
				sum2 += mat->vals[i][j] * previous_x[j];
			}

			x[i] = (b[i] - sum1 - sum2) / mat->vals[i][i];
		}

		for (int i = 0; i < size; i++)
		{
			previous_x[i] = x[i];
		}

		CalculateRes(mat, b, x,params);
		params->norm_res = CalculateNormRes(params->res, size);

	} while (params->norm_res > epsilon);

	end = clock();
	params->time = double(end - start) / double(CLOCKS_PER_SEC);

	free(previous_x);
}

void CopyMatrix(Mat* from, Mat* to) // deep copy macierzy
{
	to->size = from->size;

	for (int i = 0; i < from->size; i++)
	{
		for (int j = 0; j < from->size; j++)
		{
			to->vals[i][j] = from->vals[i][j];
		}
	}
}

void ForwardSubstitution( Mat* mat, double* y, double* b, int size)
{
	for (int i = 0; i < size; i++)
	{
		double sum = 0.0;
		for (int j = 0; j <= i - 1; j++)
		{
			sum += mat->vals[i][j] * y[j];
		}

		y[i] = (b[i] - sum) / mat->vals[i][i];
	}
}

void BackwardSubstitution(Mat* mat, double* x, double* y, int size)
{
	for (int i = size-1; i >=0; i--)
	{
		double sum = 0.0;
		for (int j = i+1; j < size ; j++)
		{
			sum += mat->vals[i][j] * x[j];
		}

		x[i] = (y[i] - sum) / mat->vals[i][i];
	}
}

void LU(Mat* mat, double* b, double* x,  Params* params)
{
	int size = mat->size;
	
	Mat L, U;
	InitMatrix(&L, size);
	FillMatrix(&L, 1.0, 0.0, 0.0); // Macierz jednostkowa

	InitMatrix(&U, size);
	CopyMatrix(mat, &U);

	double* y = (double*)malloc(size * sizeof(double));

	clock_t start, end;

	start = clock();

	for (int i = 0; i < size - 1; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			L.vals[j][i] = U.vals[j][i] / U.vals[i][i];

			for (int k = i; k < size; k++)
			{
				U.vals[j][k] = U.vals[j][k] - L.vals[j][i] * U.vals[i][k];
			}
		}
	}

	//Ly = B
	ForwardSubstitution(&L, y, b,size);

	//Ux = y
	BackwardSubstitution(&U, x, y, size);

	CalculateRes(mat, b, x, params);
	params->norm_res = CalculateNormRes(params->res, size);

	end = clock();
	params->time = double(end - start) / double(CLOCKS_PER_SEC);
	
	free(y);
	for (int j = 0; j < size; j++)
	{
		free(L.vals[j]);
		free(U.vals[j]);
	}
	free(L.vals);
	free(U.vals);
}

int main()
{
	Mat A;
	Params par;

	InitMatrix(&A, N);
	FillMatrix(&A, 5.0 + e, -1.0, -1.0);

	double* b = (double*)malloc(N * sizeof(double));
	double* x = (double*)malloc(N * sizeof(double));
	par.res = (double*)malloc(N * sizeof(double));

	for (int i = 0; i < N; i++)
	{
		int n = i + 1; // poniewa¿ n=1,2,3...994, to do indeksu z tablicy dodajê 1.
		b[i] = sin(n * (f + 1.0)); 
	}

	double epsilon = 1e-9;

	// ZADANIE B

	Jacobi(&A, b,x, &par, epsilon);
	cout << "Zadanie B - Metoda Jacobiego" << endl;
	cout << "Wartosc normy wektora residuum\t: " << par.norm_res << endl;
	cout << "Liczba iteracji\t\t: " << par.iterations << endl;
	cout << "Czas trwania operacji\t: " << par.time << "s" << endl << endl << endl;


	GaussSeidel(&A, b,x, &par, epsilon);
	cout << "Zadanie B - Metoda Gaussa-Seidla" << endl;
	cout << "Wartosc normy wektora residuum\t: " << par.norm_res << endl;
	cout << "Liczba iteracji\t\t: " << par.iterations << endl;
	cout << "Czas trwania operacji\t: " << par.time << "s" << endl << endl << endl;

	// ZADANIE C

	FillMatrix(&A, 3.0, -1.0, -1.0);

	Jacobi(&A, b,x, &par, epsilon);
	cout << "Zadanie C - Metoda Jacobiego" << endl;
	cout << "Wartosc normy wektora residuum\t: " << par.norm_res << endl;
	cout << "Liczba iteracji\t\t: " << par.iterations << endl;
	cout << "Czas trwania operacji\t: " << par.time << "s" << endl << endl << endl;

	GaussSeidel(&A, b,x, &par, epsilon);
	cout << "Zadanie C - Metoda Gaussa-Seidla" << endl;
	cout << "Wartosc normy wektora residuum\t: " << par.norm_res << endl;
	cout << "Liczba iteracji\t\t: " << par.iterations << endl;
	cout << "Czas trwania operacji\t: " << par.time << "s" << endl << endl << endl;
	
	// ZADANIE D

	LU(&A, b, x, &par);
	cout << "Zadanie D - faktoryzacja LU" << endl;
	cout << "Wartosc normy wektora residuum\t: " << par.norm_res << endl;
	cout << "Czas trwania operacji\t: " << par.time << "s" << endl << endl << endl;


for (int j = 0; j < N; j++)
	{
		free(A.vals[j]);
	}
	free(x);
	free(b);
	free(A.vals);

	// ZADANIE E

	int Ntable[8] = { 100,500,1000,2000,3000,4000,5000,6000 };

	for (int i = 0; i < 8; i++)
	{
		Mat A;
		Params p;
		double* b = (double*)malloc(Ntable[i] * sizeof(double));
		x = (double*)malloc(Ntable[i] * sizeof(double));

		InitMatrix(&A, Ntable[i]);
		FillMatrix(&A, 5.0 + e, -1.0, -1.0);

		for (int j = 0; j < Ntable[i]; j++)
		{
			int n = j + 1; // poniewa¿ n=1,2,3...994, to do indeksu z tablicy dodajê 1.
			b[j] = sin(n * (f + 1.0));
		}

		Jacobi(&A, b,x, &p, epsilon);
		cout << "Zadanie E - Metoda Jacobiego dla N = " << Ntable[i] << endl;
		cout << "Wartosc normy wektora residuum\t: " << p.norm_res << endl;
		cout << "Liczba iteracji\t\t: " << p.iterations << endl;
		cout << "Czas trwania operacji\t: " << p.time << "s" << endl << endl << endl;

		GaussSeidel(&A, b,x, &p, epsilon);
		cout << "Zadanie E - Metoda Gaussa-Seidla dla N = " << Ntable[i] << endl;
		cout << "Wartosc normy wektora residuum\t: " << p.norm_res << endl;
		cout << "Liczba iteracji\t\t: " << p.iterations << endl;
		cout << "Czas trwania operacji\t: " << p.time << "s" << endl << endl << endl;

		LU(&A, b,x, &p);
		cout << "Zadanie E - faktoryzacja LU dla N = " << Ntable[i] << endl;
		cout << "Wartosc normy wektora residuum\t: " << p.norm_res << endl;
		cout << "Czas trwania operacji\t: " << p.time << "s" << endl << endl << endl;

		for (int j = 0; j < Ntable[i]; j++)
		{
			free(A.vals[j]);
		}
		free(A.vals);
		free(b);
		free(x);
	}
	return 0;
}