#
# Copyright (C) Advanced Micro Devices, Inc. 2016 - 2022. ALL RIGHTS RESERVED.
# Copyright (c) 2001-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

# ROCM_PARSE_FLAGS(ARG, VAR_LIBS, VAR_LDFLAGS, VAR_CPPFLAGS)
# ----------------------------------------------------------
# Parse whitespace-separated ARG into appropriate LIBS, LDFLAGS, and
# CPPFLAGS variables.
AC_DEFUN([ROCM_PARSE_FLAGS],
[for arg in $$1 ; do
    AS_CASE([$arg],
        [yes],               [],
        [no],                [],
        [-l*|*.a|*.so],      [$2="$$2 $arg"],
        [-L*|-WL*|-Wl*],     [$3="$$3 $arg"],
        [-I*],               [$4="$$4 $arg"],
        [*lib|*lib/|*lib64|*lib64/],[AS_IF([test -d $arg], [$3="$$3 -L$arg"],
                                 [AC_MSG_WARN([$arg of $1 not parsed])])],
        [*include|*include/],[AS_IF([test -d $arg], [$4="$$4 -I$arg"],
                                 [AC_MSG_WARN([$arg of $1 not parsed])])],
        [AC_MSG_WARN([$arg of $1 not parsed])])
done])

# ROCM_BUILD_FLAGS(ARG, VAR_LIBS, VAR_LDFLAGS, VAR_CPPFLAGS, VAR_ROOT)
# ----------------------------------------------------------
# Parse value of ARG into appropriate LIBS, LDFLAGS, and
# CPPFLAGS variables.
AC_DEFUN([ROCM_BUILD_FLAGS],
    $4="-I$1/include/hsa -I$1/include"
    $3="-L$1/lib -L$1/lib64 -L$1/hsa/lib"
    $2="-lhsa-runtime64 -lhsakmt"
    $5="$1"
)

# HIP_BUILD_FLAGS(ARG, VAR_LIBS, VAR_LDFLAGS, VAR_CPPFLAGS)
# ----------------------------------------------------------
# Parse value of ARG into appropriate LIBS, LDFLAGS, and
# CPPFLAGS variables.
AC_DEFUN([HIP_BUILD_FLAGS],
    $4="-D__HIP_PLATFORM_AMD__ -I$1/include/hip -I$1/include"
    $3="-L$1/lib"
    $2="-lamdhip64"
)

#
# Check for ROCm  support
#
AC_DEFUN([CHECK_ROCM],[

AS_IF([test "x$rocm_checked" != "xyes"],[

AC_ARG_WITH([rocm],
    [AS_HELP_STRING([--with-rocm=(DIR)],
        [Enable the use of ROCm (default is autodetect).])],
    [],
    [with_rocm=guess])

rocm_happy=no
hip_happy=no
AS_IF([test "x$with_rocm" != "xno"],
    [AS_CASE(["x$with_rocm"],
        [x|xguess|xyes],
            [AC_MSG_NOTICE([ROCm path was not specified. Guessing ...])
             with_rocm="/opt/rocm"
             ROCM_BUILD_FLAGS([$with_rocm],
                          [ROCM_LIBS], [ROCM_LDFLAGS], [ROCM_CPPFLAGS], [ROCM_ROOT])],
        [x/*],
            [AC_MSG_NOTICE([ROCm path given as $with_rocm ...])
             ROCM_BUILD_FLAGS([$with_rocm],
                          [ROCM_LIBS], [ROCM_LDFLAGS], [ROCM_CPPFLAGS], [ROCM_ROOT])],
        [AC_MSG_NOTICE([ROCm flags given ...])
         ROCM_PARSE_FLAGS([$with_rocm],
                          [ROCM_LIBS], [ROCM_LDFLAGS], [ROCM_CPPFLAGS])])

    SAVE_CPPFLAGS="$CPPFLAGS"
    SAVE_LDFLAGS="$LDFLAGS"
    SAVE_LIBS="$LIBS"

    CPPFLAGS="$ROCM_CPPFLAGS $CPPFLAGS"
    LDFLAGS="$ROCM_LDFLAGS $LDFLAGS"
    LIBS="$ROCM_LIBS $LIBS"

    rocm_happy=yes
    AS_IF([test "x$rocm_happy" = xyes],
          [AC_CHECK_HEADERS([hsa.h], [rocm_happy=yes], [rocm_happy=no])])
    AS_IF([test "x$rocm_happy" = xyes],
          [AC_CHECK_HEADERS([hsa_ext_amd.h], [rocm_happy=yes], [rocm_happy=no])])
    AS_IF([test "x$rocm_happy" = xyes],
          [AC_CHECK_LIB([hsa-runtime64], [hsa_init], [rocm_happy=yes], [rocm_happy=no])])

    AS_IF([test "x$rocm_happy" = "xyes"],
          [AC_DEFINE([HAVE_ROCM], 1, [Enable ROCM support])
           AC_SUBST([ROCM_CPPFLAGS])
           AC_SUBST([ROCM_LDFLAGS])
           AC_SUBST([ROCM_LIBS])
           AC_SUBST([ROCM_ROOT])],
          [AC_MSG_WARN([ROCm not found])])

    CPPFLAGS="$SAVE_CPPFLAGS"
    LDFLAGS="$SAVE_LDFLAGS"
    LIBS="$SAVE_LIBS"

    #Check whether we run on ROCm 5.0 or higher
    AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[#include <${with_rocm}/include/rocm_version.h>
        ]], [[
#if ROCM_VERSION_MAJOR >= 5
int main() {return 0;}
#else
intr make+compilation_fail()
#endif
        ]])],
        [ROCM_VERSION_50_OR_GREATER=1],
        [ROCM_VERSION_50_OR_GREATER=0])


    HIP_BUILD_FLAGS([$with_rocm], [HIP_LIBS], [HIP_LDFLAGS], [HIP_CPPFLAGS])
    AC_MSG_CHECKING([if ROCm version is 5.0 or above])
    if test "$ROCM_VERSION_50_OR_GREATER" = "1" ; then
        AC_MSG_RESULT([yes])
    else
        AC_MSG_RESULT([no])
        HIP_CPPFLAGS="${HIP_CPPFLAGS} -I${with_rocm}/hip/include"
        HIP_LDFLAGS="${HIP_LDFLAGS} -L${with_rocm}/hip/lib"
    fi

    CPPFLAGS="$HIP_CPPFLAGS $CPPFLAGS"
    LDFLAGS="$HIP_LDFLAGS $LDFLAGS"
    LIBS="$HIP_LIBS $LIBS"

    # hip_runtime.h contains a C++17 features
    # in some ROCm releases. Only releveant for
    # compiling gtests and ucc_test_mpi with UCC
    #
    AC_MSG_CHECKING([c++17 support])
    AC_LANG_PUSH([C++])
    SAVE_CXXFLAGS="$CXXFLAGS"
    CXX17FLAGS="-std=c++17"
    CXXFLAGS="$CXXFLAGS $CXX17FLAGS"
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([[#include <iostream>
					int main(int argc, char** argv) {
						const int x=3;
						if constexpr (x != 3) return 1;
						return 0;
					} ]])],
                  [AC_MSG_RESULT([yes])
                   cxx17_happy=yes],
                  [AC_MSG_RESULT([no])
                   cxx17_happy=no])
    CXXFLAGS="$SAVE_CXXFLAGS"
    AC_LANG_POP

    hip_happy=no
    AC_CHECK_LIB([hip_hcc], [hipFree], [AC_MSG_WARN([Please install ROCm-3.7.0 or above])], [hip_happy=yes])
    AS_IF([test "x$hip_happy" = xyes],
          [AC_CHECK_HEADERS([hip/hip_runtime.h], [hip_happy=yes], [hip_happy=no])])
    AS_IF([test "x$hip_happy" = xyes],
          [AC_CHECK_LIB([amdhip64], [hipFree], [hip_happy=yes], [hip_happy=no])])
    AS_IF([test "x$hip_happy" = xyes],
          [AS_IF([test "x$cxx17_happy" = xyes], [HIP_CXXFLAGS="--std=gnu++17"], [HIP_CXXFLAGS=--std=gnu++11])],
          [])

    CPPFLAGS="$SAVE_CPPFLAGS"
    LDFLAGS="$SAVE_LDFLAGS"
    LIBS="$SAVE_LIBS"


    AS_IF([test "x$hip_happy" = "xyes"],
          [AC_PATH_PROG([HIPCC], [hipcc], [notfound], [$PATH:$with_rocm/bin])])
         AS_IF([test "$HIPCC" = "notfound"], [hip_happy="no"])

    AS_IF([test "x$hip_happy" = "xyes"],
          [AC_DEFINE([HAVE_HIP], 1, [Enable HIP support])
           AC_SUBST([HIPCC])
           AC_SUBST([HIP_CPPFLAGS])
           AC_SUBST([HIP_CXXFLAGS])
           AC_SUBST([HIP_LDFLAGS])
           AC_SUBST([HIP_LIBS])],
          [AC_MSG_WARN([HIP Runtime not found])])

    ],
    [AC_MSG_WARN([ROCm was explicitly disabled])]
)



rocm_checked=yes
AM_CONDITIONAL([HAVE_ROCM], [test "x$rocm_happy" != xno])
AM_CONDITIONAL([HAVE_HIP], [test "x$hip_happy" != xno])

])

])
