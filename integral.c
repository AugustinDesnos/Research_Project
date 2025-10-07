#include <math.h>
#include <stdio.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// Gaussian PDF in 2D
double gaussian_pdf(double x, double y,
                    double mu_x, double mu_y,
                    double sigma_x, double sigma_y,
                    double rho)
{
    double dx = (x - mu_x) / sigma_x;
    double dy = (y - mu_y) / sigma_y;
    double norm = 1.0 / (2.0 * M_PI * sigma_x * sigma_y * sqrt(1 - rho*rho));
    double expo = -1.0 / (2.0 * (1 - rho*rho)) * (dx*dx - 2*rho*dx*dy + dy*dy);
    return norm * exp(expo);
}

// Integrand with alpha/lambda transformation
EXPORT double integrand(double lat, double lon,
                 double inc, int N,
                 double mu_x, double mu_y,
                 double sigma_x, double sigma_y, double rho)
{   
    double alpha1 = asin(sin(lat) / sin(inc));
    if (isnan(alpha1)) ;
    double alpha2 = M_PI - alpha1;
    double pdf_sum = 0.0;

    for (int n = -N; n <= N; n++) {
        for (int m = -N; m <= N; m++) {
            double alpha_vals[2] = { alpha1 + 2.0*M_PI*n, alpha2 + 2.0*M_PI*n };
            for (int b = 0; b < 2; b++) {
                double alpha = alpha_vals[b];
                double num = cos(alpha) * sin(lon) - cos(inc) * sin(alpha) * cos(lon);
                double den = cos(alpha) * cos(lon) + cos(inc) * sin(alpha) * sin(lon);
                double lambda = atan2(num, den) + 2.0*M_PI*m;

                pdf_sum += gaussian_pdf(lambda, alpha, mu_x, mu_y, sigma_x, sigma_y, rho);
            }
        }

    }
    return pdf_sum;
}

