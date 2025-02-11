#include <stdio.h>
#include <limits.h>
#include <float.h>

int main() {
    printf("char: %d to %d\n", CHAR_MIN, CHAR_MAX);
    printf("short: %d to %d\n", SHRT_MIN, SHRT_MAX);
    printf("int: %d to %d\n", INT_MIN, INT_MAX);
    printf("long: %ld to %ld\n", LONG_MIN, LONG_MAX);
    printf("long long: %lld to %lld\n", LLONG_MIN, LLONG_MAX);

    printf("float: min %.10e, max %.10e\n", FLT_MIN, FLT_MAX);
    printf("double: min %.10e, max %.10e\n", DBL_MIN, DBL_MAX);
    printf("long double: min %.10Le, max %.10Le\n", LDBL_MIN, LDBL_MAX);

    return 0;
}
