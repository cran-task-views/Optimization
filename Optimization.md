---
name: Optimization
topic: Optimization and Mathematical Programming
maintainer: Hans W. Borchers, Florian Schwendinger
email: R-optimization at mailbox.org
version: 2022-01-09
source: https://github.com/cran-task-views/Optimization/
---

This CRAN Task View contains a list of packages which offer facilities
for solving optimization problems. Although every regression model in
statistics solves an optimization problem, they are not part of this
view. If you are looking for regression methods, the following views
will contain useful starting points: 
`r view("Multivariate")`, 
`r view("SocialSciences")`, 
`r view("Robust")`
The focus of this task view is on **Optimization
Infrastructure Packages**, **General Purpose Continuous
Solvers**, **Mathematical Programming Solvers**, **Specific
Applications in Optimization**, or **Multi Objective Optimization**.

Packages are categorized according to these sections.
See also the "Related Links" and "Other Resources" sections at the end.

Many packages provide functionality for more than one of the subjects
listed at the end of this task view. E.g., mixed integer linear
programming solvers typically offer standard linear programming routines
like the simplex algorithm. Therefore following each package description
a list of abbreviations describes the typical features of the optimizer
(i.e., the problems which can be solved). The full names of the
abbreviations given in square brackets can be found at the end of this
task view under **Classification According to Subject**.

If you think that some package is missing from the list, please contact
the maintainer via e-mail or submit an issue or pull request in the GitHub
repository linked above.


### Optimization Infrastructure Packages

-   The `r pkg("optimx")` package provides a replacement and
    extension of the `optim()` function in Base R with a call to several
    function minimization codes in R in a single statement. These
    methods handle smooth, possibly box constrained functions of several
    or many parameters. Function `optimr()` in this package extends the
    `optim()` function with the same syntax but more 'method' choices.
    Function `opm()` applies several solvers to a selected optimization
    task and returns a dataframe of results for easy comparison.

-   The R Optimization Infrastructure (`r pkg("ROI")`)
    package provides a framework for handling optimization problems in
    R. It uses an object-oriented approach to define and solve various
    optimization tasks from different problem classes (e.g., linear,
    quadratic, non-linear programming problems). This makes optimization
    transparent for the user as the corresponding workflow is abstracted
    from the underlying solver. The approach allows for easy switching
    between solvers and thus enhances comparability. For more
    information see the [ROI home
    page](http://roi.r-forge.r-project.org/) .

-   The package `r pkg("CVXR")` provides an object-oriented
    modeling language for Disciplined Convex Programming (DCP). It
    allows the user to formulate convex optimization problems in a
    natural way following mathematical convention and DCP rules. The
    system analyzes the problem, verifies its convexity, converts it
    into a canonical form, and hands it off to an appropriate solver
    such as ECOS or SCS to obtain the solution. (CVXR is derived from
    the MATLAB toolbox CVX, developed at Stanford University, cf. [CVXR
    home page](https://cvxr.rbind.io) .)

### General Purpose Continuous Solvers

Package stats offers several general purpose optimization routines. For
one-dimensional unconstrained function optimization there is
`optimize()` which searches an interval for a minimum or maximum.
Function `optim()` provides an implementation of the
Broyden-Fletcher-Goldfarb-Shanno (BFGS) method, bounded BFGS, conjugate
gradient (CG), Nelder-Mead, and simulated annealing (SANN) optimization
methods. It utilizes gradients, if provided, for faster convergence.
Typically it is used for unconstrained optimization but includes an
option for box-constrained optimization.

Additionally, for minimizing a function subject to linear inequality
constraints, stats contains the routine `constrOptim()`. Then there is
`nlm` which is used for solving nonlinear unconstrained minimization
problems. `nlminb()` offers box-constrained optimization using the PORT
routines. \[RGA, QN\]

-   Package `r pkg("lbfgs")` wraps the libBFGS C library by
    Okazaki and Morales (converted from Nocedal's L-BFGS-B 3.0 Fortran
    code), interfacing both the L-BFGS and the OWL-QN algorithm, the
    latter being particularly suited for higher-dimensional problems.
-   `r pkg("lbfgsb3c")` interfaces J.Nocedal's L-BFGS-B 3.0
    Fortran code, a limited memory BFGS minimizer, allowing bound
    constraints and being applicable to higher-dimensional problems. It
    has an 'optim'-like interface based on 'Rcpp'.
-   Package `r pkg("roptim")` provides a unified wrapper to
    call C++ functions of the algorithms underlying the optim() solver;
    and `r pkg("optimParallel")` provides a parallel version
    of the L-BFGS-B method of optim(); using these packages can
    significantly reduce the optimization time.
-   `r pkg("RcppNumerical")` is a collection of open source
    libraries for numerical computing and their integration with
    'Rcpp'. It provides a wrapper for the L-BFGS algorithm, based on
    the LBFGS++ library (based on code of N. Okazaki).
-   Package `r pkg("ucminf", priority = "core")` implements
    an algorithm of quasi-Newton type for nonlinear unconstrained
    optimization, combining a trust region with line search approaches.
    The interface of `ucminf()` is designed for easy interchange with
    `optim()`.\[QN\]
-   The following packages implement optimization routines in pure R,
    for nonlinear functions with bounds constraints:
    `r pkg("Rcgmin")`: gradient function minimization
    similar to GC; `r pkg("Rvmmin")`: variable metric
    function minimization; `r pkg("Rtnmin")`: truncated
    Newton function minimization.
-   `r pkg("mize")` implements optimization algorithms in
    pure R, including conjugate gradient (CG),
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) and limited memory BFGS
    (L-BFGS) methods. Most internal parameters can be set through the
    calling interface.
-   `r pkg("stochQN")` provides implementations of
    stochastic, limited-memory quasi-Newton optimizers, similar in
    spirit to the LBFGS. It includes an implementation of online LBFGS,
    stochastic quasi-Newton and adaptive quasi-Newton.
-   `r pkg("nonneg.cg")` realizes a conjugate-gradient based
    method to minimize functions subject to all variables being
    non-negative.
-   Package `r pkg("dfoptim", priority = "core")`, for
    derivative-free optimization procedures, contains quite efficient R
    implementations of the Nelder-Mead and Hooke-Jeeves algorithms
    (unconstrained and with bounds constraints). \[DF\]
-   Package `r pkg("nloptr")` provides access to NLopt, an
    LGPL licensed library of various nonlinear optimization algorithms.
    It includes local derivative-free (COBYLA, Nelder-Mead, Subplex) and
    gradient-based (e.g., BFGS) methods, and also the augmented
    Lagrangian approach for nonlinear constraints. \[DF, GO, QN\]
-   Package `r pkg("alabama", priority = "core")` provides an
    implementations of the Augmented Lagrange Barrier minimization
    algorithm for optimizing smooth nonlinear objective functions with
    (nonlinear) equality and inequality constraints.
-   Package `r pkg("Rsolnp")` provides an implementation of
    the Augmented Lagrange Multiplier method for solving nonlinear
    optimization problems with equality and inequality constraints
    (based on code by Y. Ye).
-   `r pkg("NlcOptim")` solves nonlinear optimization
    problems with linear and nonlinear equality and inequality
    constraints, implementing a Sequential Quadratic Programming (SQP)
    method; accepts the input parameters as a constrained matrix.
-   In package Rdonlp2 (see the `r rforge("rmetrics")`
    project) function `donlp2()`, a wrapper for the DONLP2 solver,
    offers the minimization of smooth nonlinear functions and
    constraints. DONLP2 can be used freely for any kind of research
    purposes, otherwise it requires licensing. \[GO, NLP\]
-   `r pkg("psqn")` provides quasi-Newton methods to
    minimize partially separable functions; the methods are largely
    described in "Numerical Optimization" by Nocedal and Wright
    (2006).
-   `r pkg("clue")` contains the function `sumt()` for
    solving constrained optimization problems via the sequential
    unconstrained minimization technique (SUMT).
-   `r pkg("BB")` contains the function `spg()` providing a
    spectral projected gradient method for large scale optimization with
    simple constraints. It takes a nonlinear objective function as an
    argument as well as basic constraints.
-   Package `r pkg("SCOR")` solves optimization problems
    under the constraint that the combined parameters lie on the surface
    of a unit hypersphere.
-   `r pkg("GrassmannOptim")` is a package for Grassmann
    manifold optimization. The implementation uses gradient-based
    algorithms and embeds a stochastic gradient method for global
    search.
-   `r pkg("ManifoldOptim")` is an R interface to the
    'ROPTLIB' optimization library. It optimizes real-valued functions
    over manifolds such as Stiefel, Grassmann, and Symmetric Positive
    Definite matrices.
-   Package `r pkg("gsl")` provides BFGS, conjugate
    gradient, steepest descent, and Nelder-Mead algorithms. It uses a
    "line search" approach via the function `multimin()`. It is based
    on the GNU Scientific Library (GSL). \[RGA, QN\]
-   Several derivative-free optimization algorithms are provided with
    package `r pkg("minqa")`; e.g., the functions
    `bobyqa()`, `newuoa()`, and `uobyqa()` allow to minimize a function
    of many variables by a trust region method that forms quadratic
    models by interpolation. `bobyqa()` additionally permits box
    constraints (bounds) on the parameters. \[DF\]
-   `r pkg("subplex")` provides unconstrained function
    optimization based on a subspace searching simplex method.
-   In package `r pkg("trust")`, a routine with the same
    name offers local optimization based on the "trust region"
    approach.
-   `r pkg("trustOptim")` implements "trust region" for
    unconstrained nonlinear optimization. The algorithm is optimized for
    objective functions with sparse Hessians.
-   Package `r pkg("quantreg")` contains variations of
    simplex and of interior point routines ( `nlrq()`, `crq()`). It
    provides an interface to L1 regression in the R code of function
    `rq()`. \[SPLP, LP, IPM\]

### Quadratic Optimization

-   In package `r pkg("quadprog", priority = "core")`
    `solve.QP()` solves quadratic programming problems with linear
    equality and inequality constraints. (The matrix has to be positive
    definite.) `r pkg("quadprogXT")` extends this with
    absolute value constraints and absolute values in the objective
    function. \[QP\]
-   `r pkg("osqp")` provides bindings to
    [OSQP](https://osqp.org) , the 'Operator Splitting QP' solver from
    the University of Oxford Control Group; it solves sparse convex
    quadratic programming problems with optional equality and inequality
    constraints efficiently. \[QP\]
-   `r pkg("qpmadr")` interfaces the 'qpmad' software and
    solves quadratic programming (QP) problems with linear inequality,
    equality and bound constraints, using the method by Goldfarb and
    Idnani.\[QP\]
-   `r pkg("kernlab")` contains the function `ipop` for
    solving quadratic programming problems using interior point methods.
    (The matrix can be positive semidefinite.) \[IPM, QP\]
-   `r pkg("Dykstra")` solves quadratic programming problems
    using R. L. Dykstra's cyclic projection algorithm for positive
    definite and semidefinite matrices. The routine allows for a
    combination of equality and inequality constraints. \[QP\]
-   `r pkg("coneproj")` contains routines for cone
    projection and quadratic programming, estimation and inference for
    constrained parametric regression, and shape-restricted regression
    problems. \[QP\]
-   `r pkg("LowRankQP")` primal/dual interior point method
    solving quadratic programming problems (especially for semidefinite
    quadratic forms). \[IPM, QP\]
-   The COIN-OR project 'qpOASES' implements a reliable QP solver,
    even when tackling semi-definite or degenerated QP problems; it is
    particularly suited for model predictive control (MPC) applications;
    the ROI plugin `r pkg("ROI.plugin.qpoases")` makes it
    accessible for R users. \[QP\]
-   `r pkg("QPmin")` is active set method solver for the
    solution of indefinite quadratic programs, subject to lower bounds
    on linear functions of the variables. \[QP\]
-   `r pkg("mixsqp")` implements the "mix-SQP" algorithm,
    based on sequential quadratic programming (SQP), for maximum
    likelihood estimations in finite mixture models.
-   `r pkg("limSolve")` offers to solve linear or quadratic
    optimization functions, subject to equality and/or inequality
    constraints. \[LP, QP\]

### Optimization Test Functions

-   Objective functions for benchmarking the performance of global
    optimization algorithms can be found in
    `r pkg("globalOptTests")`.
-   `r pkg("smoof")` has generators for a number of both
    single- and multi-objective test functions that are frequently used
    for benchmarking optimization algorithms; offers a set of convenient
    functions to generate, plot, and work with objective functions.
-   `r pkg("flacco")` contains tools and features used for
    an Exploratory Landscape Analysis (ELA) of continuous optimization
    problems, capable of quantifying rather complex properties, such as
    the global structure, separability, etc., of the optimization
    problems.
-   `r pkg("cec2013")` and 'cec2005benchmark' (archived)
    contain many test functions for global optimization from the 2005
    and 2013 special sessions on real-parameter optimization at the IEEE
    CEC congresses on evolutionary computation.
-   Package `r github("jlmelville/funconstrain")` (on
    Github) implements 35 of the test functions by More, Garbow, and
    Hillstom, useful for testing unconstrained optimization methods.

### Least-Squares Problems

Function `solve.qr()` (resp. `qr.solve()`) handles over- and
under-determined systems of linear equations, returning least-squares
solutions if possible. And package stats provides `nls()` to determine
least-squares estimates of the parameters of a nonlinear model.
`r pkg("nls2")` enhances function `nls()` with brute force
or grid-based searches, to avoid being dependent on starting parameters
or getting stuck in local solutions.

-   Package `r pkg("nlsr")` provides tools for working with
    nonlinear least-squares problems. Functions `nlfb` and `nlxb` are
    intended to eventually supersede the 'nls()' function in Base R,
    by applying a variant of the Marquardt procedure for nonlinear
    least-squares, with bounds constraints and optionally Jacobian
    described as R functions. (It is based on the now-deprecated package
    `r pkg("nlmrt")`.)
-   Package `r pkg("minpack.lm")` provides a function
    `nls.lm()` for solving nonlinear least-squares problems by a
    modification of the Levenberg-Marquardt algorithm, with support for
    lower and upper parameter bounds, as found in MINPACK.
-   Package `r pkg("nnls")` interfaces the Lawson-Hanson
    implementation of an algorithm for non-negative least-squares,
    allowing the combination of non-negative and non-positive
    constraints.
-   Package `r pkg("gslnls")` provides an interface to
    nonlinear least-squares optimization methods from the GNU Scientific
    Library (GSL). The available trust region methods include the
    Levenberg-Marquadt algorithm with and without geodesic acceleration,
    and several variants of Powell's dogleg algorithm.
-   Package `r pkg("bvls")` interfaces the Stark-Parker
    implementation of an algorithm for least-squares with upper and
    lower bounded variables.
-   Package `r pkg("onls")` implements orthogonal nonlinear
    least-squares regression (ONLS, a.k.a. Orthogonal Distance
    Regression, ODR) using a Levenberg-Marquardt-type minimization
    algorithm based on the ODRPACK Fortran library.
-   `r pkg("colf")` performs least squares constrained
    optimization on a linear objective function. It contains a number of
    algorithms to choose from and offers a formula syntax similar to
    `lm()`.

### Semidefinite and Convex Solvers

-   Package `r pkg("ECOSolveR")` provides an interface to
    the Embedded COnic Solver (ECOS), a well-known, efficient, and
    robust C library for convex problems. Conic and equality constraints
    can be specified in addition to integer and boolean variable
    constraints for mixed-integer problems.
-   Package `r pkg("scs")` applies operator splitting to
    solve linear programs (LPs), second-order cone programs (SOCP),
    semidefinite programs, (SDPs), exponential cone programs (ECPs), and
    power cone programs (PCPs), or problems with any combination of
    those cones.
-   `r pkg("sdpt3r")` solves general semidefinite Linear
    Programming problems, using an R implementation of the MATLAB
    toolbox SDPT3. Includes problems such as the nearest correlation
    matrix, D-optimal experimental design, Distance Weighted
    Discrimination, or the maximum cut problem.
-   `r pkg("cccp")` contains routines for solving cone
    constrained convex problems by means of interior-point methods. The
    implemented algorithms are partially ported from CVXOPT, a Python
    module for convex optimization
-   The `r pkg("CLSOCP")` package provides an implementation
    of a one-step smoothing Newton method for the solution of second
    order cone programming (SOCP) problems.
-   CSDP is a library of routines that implements a primal-dual barrier
    method for solving semidefinite programming problems; it is
    interfaced in the `r pkg("Rcsdp")` package. \[SDP\]
-   The DSDP library implements an interior-point method for
    semidefinite programming with primal and dual solutions; it is
    interfaced in package `r pkg("Rdsdp")`. \[SDP\]
-   Package `r pkg("Rmosek")` provides an interface to the
    (commercial) MOSEK optimization library for large-scale LP, QP, and
    MIP problems, with emphasis on (nonlinear) conic, semidefinite, and
    convex tasks; academic licenses are available. (An article on Rmosek
    appeared in the JSS special issue on Optimization with R, see
    below.) \[SDP, CP\]

### Global and Stochastic Optimization

-   Package `r pkg("DEoptim", priority = "core")` provides a
    global optimizer based on the Differential Evolution algorithm.
    `r pkg("RcppDE")` provides a C++ implementation (using
    Rcpp) of the same `DEoptim()` function.
-   `r pkg("DEoptimR")` provides an implementation of the
    jDE variant of the differential evolution stochastic algorithm for
    nonlinear programming problems (It allows to handle constraints in a
    flexible manner.)
-   The `r pkg("CEoptim")` package implements a
    cross-entropy optimization technique that can be applied to
    continuous, discrete, mixed, and constrained optimization problems.
    \[COP\]
-   `r pkg("GenSA")` is a package providing a function for
    generalized Simulated Annealing which can be used to search for the
    global minimum of a quite complex non-linear objective function with
    a large number of optima.
-   `r pkg("GA")` provides functions for optimization using
    Genetic Algorithms in both, the continuous and discrete case. This
    package allows to run corresponding optimization tasks in parallel.
-   Package `r pkg("genalg")` contains `rbga()`, an
    implementation of a genetic algorithm for multi-dimensional function
    optimization.
-   Package `r pkg("rgenoud")` offers `genoud()`, a routine
    which is capable of solving complex function
    minimization/maximization problems by combining evolutionary
    algorithms with a derivative-based (quasi-Newtonian) approach.
-   Machine coded genetic algorithm (MCGA) provided by package
    `r pkg("mcga")` is a tool which solves optimization
    problems based on byte representation of variables.
-   A particle swarm optimizer (PSO) is implemented in package
    `r pkg("pso")`, and also in
    `r pkg("psoptim")`. Another (parallelized)
    implementation of the PSO algorithm can be found in package `ppso`
    available from [rforge.net/ppso](https://www.rforge.net/ppso/) .
-   Package `r pkg("hydroPSO")` implements the Standard
    Particle Swarm Optimization (SPSO) algorithm; it is parallel-capable
    and includes several fine-tuning options and post-processing
    functions.
-   `r github("floybix/hydromad")` (on Github) contains the
    `SCEoptim` function for Shuffled Compex Evolution (SCE)
    optimization, an evolutionary algorithm, combined with a simplex
    method.
-   Package `r pkg("ABCoptim")` implements the Artificial
    Bee Colony (ABC) optimization approach.
-   Package `r pkg("metaheuristicOpt")` contains
    implementations of several evolutionary optimization algorithms,
    such as particle swarm, dragonfly and firefly, sine cosine
    algorithms and many others.
-   Package `r pkg("ecr")` provides a framework for building
    evolutionary algorithms for single- and multi-objective continuous
    or discrete optimization problems. And `r pkg("emoa")`
    has a collection of building blocks for the design and analysis of
    evolutionary multiobjective optimization algorithms.
-   CMA-ES by N. Hansen, global optimization procedure using a
    covariance matrix adapting evolutionary strategy, is implemented in
    several packages: In packages `r pkg("cmaes")` and
    `r pkg("cmaesr")`, in `r pkg("parma")` as
    `cmaes`, in `r pkg("adagio")` as `pureCMAES`, and in
    `r pkg("rCMA")` as `cmaOptimDP`, interfacing Hansen's
    own Java implementation.
-   Package `r pkg("Rmalschains")` implements an algorithm
    family for continuous optimization called memetic algorithms with
    local search chains (MA-LS-Chains).
-   An R implementation of the Self-Organising Migrating Algorithm
    (SOMA) is available in package `r pkg("soma")`. This
    stochastic optimization method is somewhat similar to genetic
    algorithms.
-   `r pkg("nloptr")` supports several global optimization
    routines, such as DIRECT, controlled random search (CRS),
    multi-level single-linkage (MLSL), improved stochastic ranking
    (ISR-ES), or stochastic global optimization (StoGO).
-   The `r pkg("NMOF")` package provides implementations of
    differential evolution, particle swarm optimization, local search
    and threshold accepting (a variant of simulated annealing). The
    latter two methods also work for discrete optimization problems, as
    does the implementation of a genetic algorithm that is included in
    the package.
-   `r pkg("SACOBRA")` is a package for numeric constrained
    optimization of expensive black-box functions under severely limited
    budgets; it implements an extension of the COBRA algorithm with
    initial design generation and self-adjusting random restarts.
-   `r pkg("OOR")` implements optimistic optimization
    methods for global optimization of deterministic or stochastic
    functions.
-   `r pkg("RCEIM")` implements a stochastic heuristic
    method for performing multi-dimensional function optimization.

### Mathematical Programming Solvers

This section provides an overview of open source as well as commercial
optimizers. Which type of mathematical programming problem can be solved
by a certain package or function can be seen from the abbreviations in
square brackets. For a [Classification According to
Subject](#classification-according-to-subject) see the list at the end
of this task view.

-   Package `r pkg("ompr")` is an optimization modeling
    package to model and solve Mixed Integer Linear Programs in an
    algebraic way directly in R. The models are solver-independent and
    thus offer the possibility to solve models with different solvers.
    (Inspired by Julia's JuMP project.)
-   `r pkg("linprog")` solves linear programming problems
    using the function `solveLP()` (the solver is based on
    `r pkg("lpSolve")`) and can read model files in MPS
    format. \[LP\]
-   In the `r pkg("boot")` package there is a routine called
    `simplex()` which realizes the two-phase tableau simplex method for
    (relatively small) linear programming problems. \[LP\]
-   `r pkg("rcdd")` offers the function `lpcdd()` for
    solving linear programs with exact arithmetic using the [GNU
    Multiple Precision (GMP)](https://gmplib.org) library. \[LP\]
-   The [NEOS Server for
    Optimization](https://www.neos-server.org/neos/) provides online
    access to state-of-the-art optimization problem solvers. The
    packages `r pkg("rneos")` and
    `r pkg("ROI.plugin.neos")` enable the user to pass
    optimization problems to NEOS and retrieve results within R.

#### Interfaces to Open Source Optimizers

-   Package `r pkg("lpSolve")` contains the routine `lp()`
    to solve LPs and MILPs by calling the freely available solver
    [lp\_solve](http://lpsolve.sourceforge.net) . This solver is based
    on the revised simplex method and a branch-and-bound (B&B) approach.
    It supports semi-continuous variables and Special Ordered Sets
    (SOS). Furthermore `lp.assign()` and `lp.transport()` are aimed at
    solving assignment problems and transportation problems,
    respectively. Additionally, there is the package
    `r pkg("lpSolveAPI")` which provides an R interface to
    the low level API routines of lp\_solve (see also project
    `r rforge("lpsolve")` on R-Forge).
    `r pkg("lpSolveAPI")` supports reading linear programs
    from files in lp and MPS format. \[BP, IP, LP, MILP, SPLP\]
-   Packages `r pkg("glpkAPI")` as well as package
    `r pkg("Rglpk")` provide an interface to the [GNU Linear
    Programming Kit](https://www.gnu.org/software/glpk/) (GLPK). Whereas
    the former provides high level access to low level routines the
    latter offers a high level routine `Rglpk_solve_LP()` to solve MILPs
    using GLPK. Both packages offer the possibility to use models
    formulated in the MPS format. \[BP, IP, IPM, LP, MILP\]
-   `r pkg("Rsymphony")` has the routine
    `Rsymphony_solve_LP()` that interfaces the SYMPHONY solver for
    mixed-integer linear programs. (SYMPHONY is part of the
    [Computational Infrastructure for Operations
    Research](http://www.coin-or.org/) (COIN-OR) project.) Package
    `lpsymphony` in Bioconductor provides a similar interface to
    SYMPHONY that is easier to install. \[LP, IP, MILP\]
-   The NOMAD solver is implemented in the `r pkg("crs")`
    package for solving mixed integer programming problems. This
    algorithm is accessible via the `snomadr()` function and is
    primarily designed for constrained optimization of blackbox
    functions. \[MILP\]
-   'Clp' and 'Cbc' are open source solvers from the COIN-OR suite.
    'Clp' solves linear programs with continuous objective variables,
    and 'Cbc' is a powerful mixed integer linear programming solver based
    on 'Clp', i.e. applies 'Clp' if no integer variables are set.
    'Cbc' can be installed from r github("dirkschumacher/rcbc"). \[LP, MILP\]

#### Interfaces to Commercial Optimizers

This section surveys interfaces to commercial solvers. Typically, the
corresponding libraries have to be installed separately.

-   Package `r pkg("Rcplex")` provides an interface to the
    IBM [CPLEX
    Optimizer](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/)
    . CPLEX provides dual/primal simplex optimizers as well as a barrier
    optimizer for solving large scale linear and quadratic programs. It
    offers a mixed integer optimizer to solve difficult mixed integer
    programs including (possibly non-convex) MIQCP. Note that CPLEX is
    **not free** and you have to get a license. Academics will receive a
    free licence upon request. \[LP, IP, BP, QP, MILP, MIQP, IPM\]
-   Package `r pkg("Rmosek")` offers an interface to the
    commercial optimizer from [MOSEK](https://www.mosek.com/) . It
    provides dual/primal simplex optimizers as well as a barrier
    optimizer. In addition to solving LP and QP problems this solver can
    handle SOCP and quadratically constrained programming (QPQC) tasks.
    Furthermore, it offers a mixed integer optimizer to solve difficult
    mixed integer programs (MILP, MISOCP, etc.). You have to get a
    license, but Academic licenses are free of charge. \[LP, IP, BP, QP,
    MILP, MIQP, IPM\]
-   The `r pkg("localsolver")` package provides an interface
    to the hybrid mathematical programming software
    [LocalSolver](http://www.localsolver.com/) from Innovation 24,
    combining exact and heuristic techniques. See their website for more
    details. (Academic licenses are available on request.) \[LP, MIP,
    QP, NLP, HEUR\]
-   'Gurobi Optimization' ships an R package with its software that
    allows to call its solvers from R. Gurobi provides powerful solvers
    for LP, MIP, QP, MIQP, SOCP, and MISOCP models. See their website
    for more details. (Academic licenses are available on request.)
    \[LP, QP, MILP, MIQP\]

Some more commercial companies, e.g. 'Artelys Knitro' or 'FICO Xpress
Optimization', have R interfaces that are installed while the software
gets installed. Trial licenses are available, see the corresponding
websites for more information.

### Combinatorial Optimization

-   Package `r pkg("adagio")` provides R functions for
    single and multiple knapsack problems, and solves subset sum and
    assignment tasks.
-   In package `r pkg("clue")` `solve_LSAP()` enables the
    user to solve the linear sum assignment problem (LSAP) using an
    efficient C implementation of the Hungarian algorithm. \[SPLP\]
-   `r pkg("FLSSS")` provides multi-threaded solvers for
    fixed-size single and multi dimensional subset sum problems with
    optional constraints on target sum and element range, fixed-size
    single and multi dimensional knapsack problems, binary knapsack
    problems and generalized assignment problems via exact algorithms or
    metaheuristics.
-   Package `r pkg("qap")` solves Quadratic Assignment
    Problems (QAP) applying a simulated annealing heuristics (other
    approaches will follow).
-   `r pkg("igraph")`, a package for graph and network
    analysis, uses the very fast igraph C library. It can be used to
    calculate shortest paths, maximal network flows, minimum spanning
    trees, etc. \[GRAPH\]
-   `r pkg("mknapsack")` solves multiple knapsack problems,
    based on LP solvers such as 'lpSolve' or 'CBC'; will assign
    items to knapsacks in a way that the value of the top knapsacks is
    as large as possible.
-   Package 'knapsack' (see R-Forge project
    `r rforge("optimist")`) provides routines from the book
    `Knapsack Problems' by Martello and Toth. There are functions for
    (multiple) knapsack, subset sum and binpacking problems. (Use of
    Fortran codes is restricted to personal research and academic
    purposes only.)
-   `r pkg("nilde")` provides routines for enumerating all
    integer solutions of linear Diophantine equations, resp. all
    solutions of knapsack, subset sum, and additive partitioning
    problems (based on a generating functions approach).
-   `r pkg("matchingR")` and
    `r pkg("matchingMarkets")` implement the Gale-Shapley
    algorithm for the stable marriage and the college admissions
    problem, the stable roommates and the house allocation problem.
    \[COP, MM\]
-   Package `r pkg("optmatch")` provides routines for
    solving matching problems by translating them into minimum-cost flow
    problems and then solved optimaly by the RELAX-IV codes of Bertsekas
    and Tseng (free for research). \[SPLP\]
-   Package `r pkg("TSP")` provides basic infrastructure for
    handling and solving the traveling salesperson problem (TSP). The
    main routine `solve_TSP()` solves the TSP through several
    heuristics. In addition, it provides an interface to the [Concorde
    TSP Solver](http://www.tsp.gatech.edu/concorde/index.html) , which
    has to be downloaded separately. \[SPLP\]
-   `r pkg("rminizinc")` provides an interface to the
    open-source constraint modeling language and system (to be
    downloaded separately) [MiniZinc](https://www.minizinc.org/) . R
    users can apply the package to solve combinatorial optimization
    problems by modifying existing 'MiniZinc' models, and also by
    creating their own models.

### Multi Objective Optimization

-   Function `caRamel` in package `r pkg("caRamel")` is a
    multi-objective optimizer, applying a combination of the
    multi-objective evolutionary annealing-simplex (MEAS) method and the
    non-dominated sorting genetic algorithm (NGSA-II); it was initially
    developed for the calibration of hydrological models.
-   Multi-criteria optimization problems can be solved using package
    `r pkg("mco")` which implements genetic algorithms.
    \[MOP\]
-   `r pkg("GPareto")` provides multi-objective optimization
    algorithms for expensive black-box functions and uncertainty
    quantification methods.
-   The `r pkg("rmoo")` package is a framework for multi-
    and many-objective optimization, allowing to work with
    representation of real numbers, permutations and binaries, offering
    a high range of configurations.

### Specific Applications in Optimization

-   The data cloning algorithm is a global optimization approach and a
    variant of simulated annealing which has been implemented in package
    `r pkg("dclone")`. The package provides low level
    functions for implementing maximum likelihood estimating procedures
    for complex models.
-   The `r pkg("irace")` package implements automatic
    configuration procedures for optimizing the parameters of other
    optimization algorithms, that is (offline) tuning their parameters
    by finding the most appropriate settings given a set of optimization
    problems.
-   Package `r pkg("kofnGA")` uses a genetic algorithm to
    choose a subset of a fixed size k from the integers 1:n, such that a
    user- supplied objective function is minimized at that subset.
-   `r pkg("copulaedas")` provides a platform where
    'estimation of distribution algorithms' (EDA) based on copulas can
    be implemented and studied; the package offers various EDAs, and
    newly developed EDAs can be integrated by extending an S4 class.
-   `r pkg("tabuSearch")` implements a tabu search algorithm
    for optimizing binary strings, maximizing a user defined target
    function, and returns the best (i.e. maximizing) binary
    configuration found.
-   Besides functionality for solving general isotone regression
    problems, package `r pkg("isotone")` provides a
    framework of active set methods for isotone optimization problems
    with arbitrary order restrictions.
-   `r pkg("mlrMBO")` is a flexible and comprehensive R
    toolbox for model-based optimization ('MBO'), also known as
    Bayesian optimization. And
    `r pkg("rBayesianOptimization")` is an implementation of
    Bayesian global optimization with Gaussian Processes, for parameter
    tuning and optimization of hyperparameters.
-   The Sequential Parameter Optimization Toolbox
    `r pkg("SPOT")` provides a set of tools for model-based
    optimization and tuning of algorithms. It includes surrogate models
    and design of experiment approaches.
-   The `r pkg("desirability")` package contains S3 classes
    for multivariate optimization using the desirability function
    approach of Harrington (1965).
-   `r pkg("maxLik")` adds a likelihood-specific layer on
    top of a number of maximization routines like
    Brendt-Hall-Hall-Hausman (BHHH) and Newton-Raphson among others. It
    includes summary and print methods which extract the standard errors
    based on the Hessian matrix and allows easy swapping of maximization
    algorithms.

### Classification According to Subject

What follows is an attempt to provide a by-subject overview of packages.
The full name of the subject as well as the corresponding [MSC
2010](http://www.ams.org/mathscinet/msc/msc2010.html?t=90Cxx&btn=Current)
code (if available) are given in brackets.

-   LP (Linear programming, 90C05): `r pkg("boot")`,
    `r pkg("glpkAPI")`, `r pkg("limSolve")`,
    `r pkg("linprog")`, `r pkg("lpSolve")`,
    `r pkg("lpSolveAPI")`, `r pkg("quantreg")`,
    `r pkg("rcdd")`, `r pkg("Rcplex")`,
    `r pkg("Rglpk")`, `r pkg("Rmosek")`,
    `r pkg("Rsymphony")`
-   GO (Global Optimization): `r pkg("DEoptim")`,
    `r pkg("DEoptimR")`, `r pkg("GenSA")`,
    `r pkg("GA")`, `r pkg("pso")`,
    `r pkg("rgenoud")`, `r pkg("cmaes")`,
    `r pkg("nloptr")`, `r pkg("NMOF")`,
    `r pkg("OOR")`
-   SPLP (Special problems of linear programming like transportation,
    multi-index, etc., 90C08): `r pkg("clue")`,
    `r pkg("lpSolve")`, `r pkg("lpSolveAPI")`,
    `r pkg("quantreg")`, `r pkg("TSP")`
-   BP (Boolean programming, 90C09): `r pkg("glpkAPI")`,
    `r pkg("lpSolve")`, `r pkg("lpSolveAPI")`,
    `r pkg("Rcplex")`, `r pkg("Rglpk")`
-   IP (Integer programming, 90C10): `r pkg("glpkAPI")`,
    `r pkg("lpSolve")`, `r pkg("lpSolveAPI")`,
    `r pkg("Rcplex")`, `r pkg("Rglpk")`,
    `r pkg("Rmosek")`, `r pkg("Rsymphony")`
-   MIP (Mixed integer programming and its variants MILP for LP and MIQP
    for QP, 90C11): `r pkg("glpkAPI")`,
    `r pkg("lpSolve")`, `r pkg("lpSolveAPI")`,
    `r pkg("Rcplex")`, `r pkg("Rglpk")`,
    `r pkg("Rmosek")`, `r pkg("Rsymphony")`
-   QP (Quadratic programming, 90C20): `r pkg("kernlab")`,
    `r pkg("limSolve")`, `r pkg("LowRankQP")`,
    `r pkg("quadprog")`, `r pkg("Rcplex")`,
    `r pkg("Rmosek")`
-   SDP (Semidefinite programming, 90C22): `r pkg("Rcsdp")`,
    `r pkg("Rdsdp")`
-   CP (Convex programming, 90C25): `r pkg("cccp")`,
    `r pkg("CLSOCP")`
-   COP (Combinatorial optimization, 90C27):
    `r pkg("adagio")`, `r pkg("CEoptim")`,
    `r pkg("TSP")`, `r pkg("matchingR")`
-   MOP (Multi-objective and goal programming, 90C29):
    `r pkg("caRamel")`, `r pkg("GPareto")`,
    `r pkg("mco")`, `r pkg("emoa")`,
    `r pkg("rmoo")`
-   NLP (Nonlinear programming, 90C30): `r pkg("nloptr")`,
    `r pkg("alabama")`, `r pkg("Rsolnp")`,
    Rdonlp2
-   GRAPH (Programming involving graphs or networks, 90C35):
    `r pkg("igraph")`
-   IPM (Interior-point methods, 90C51): `r pkg("kernlab")`,
    `r pkg("glpkAPI")`, `r pkg("LowRankQP")`,
    `r pkg("quantreg")`, `r pkg("Rcplex")`
-   RGA (Methods of reduced gradient type, 90C52): stats ( `optim()`),
    `r pkg("gsl")`
-   QN (Methods of quasi-Newton type, 90C53): stats ( `optim()`),
    `r pkg("gsl")`, `r pkg("lbfgs")`,
    `r pkg("lbfgsb3c")`, `r pkg("nloptr")`,
    `r pkg("optimParallel")`, `r pkg("ucminf")`
-   DF (Derivative-free methods, 90C56): `r pkg("dfoptim")`,
    `r pkg("minqa")`, `r pkg("nloptr")`



### Links

-   JSS Article: [ROI: An Extensible R Optimization Infrastructure (Theußl, Schwendinger, Hornik)](https://www.jstatsoft.org/article/view/v094i15)
-   JSS Article: [CVXR: An R Package for Disciplined Convex Optimization (Fu, Narasimhan, Boyd)](https://www.jstatsoft.org/article/view/v094i14)
-   JSS Special Issue: [Special Volume on Optimization (Ed. R. Varadhan)](https://www.jstatsoft.org/v60)
-   Textbook: [Nonlinear Parameter Optimization Using R Tools (J.C. Nash)](https://www.wiley.com/en-us/Nonlinear+Parameter+Optimization+Using+R+Tools-p-9781118569283)
-   Textbook: [Modern Optimization With R (P. Cortez)](https://link.springer.com/book/10.1007/978-3-030-72819-9)
-   Textbook: [Numerical Optimization (Nocedal, Wright)](https://link.springer.com/book/10.1007/978-0-387-40065-5)
-   Cheatsheet: [Base R Optim Cheatsheet](https://github.com/hwborchers/CheatSheets/blob/main/Base%20R%20Optim%20Cheatsheet.pdf)
-   Tutorial: [CVXR Tutorial](https://github.com/bnaras/cvxr_tutorial) and [Examples](https://cvxr.rbind.io/examples/)
-   Manual: [NLopt Manual (S. Johnson)](https://nlopt.readthedocs.io/en/latest/NLopt_manual/)
-   [Yet Another Math Programming Consultant](http://yetanothermathprogrammingconsultant.blogspot.com)
-   [COIN-OR Project](http://www.coin-or.org/)
-   [NEOS Optimization Guide](http://www.neos-guide.org/Optimization-Guide)
-   [Decision Tree for Optimization Software](http://plato.asu.edu/sub/pns.html)
-   [Mathematics Subject Classification - Mathematical programming](http://www.ams.org/mathscinet/msc/msc2010.html?t=90Cxx&btn=Current)
