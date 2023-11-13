#include "../include/tracker.h"
#include "../include/utils.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::Matrix2Xi;
using Eigen::Vector3d;
using cv::Mat;

void signal_callback_handler(int signum) {
   // Terminate program
   exit(signum);
}

double pt2pt_dis_sq (MatrixXd pt1, MatrixXd pt2) {
    return (pt1 - pt2).rowwise().squaredNorm().sum();
}

double pt2pt_dis (MatrixXd pt1, MatrixXd pt2) {
    return (pt1 - pt2).rowwise().norm().sum();
}

void reg (MatrixXd pts, MatrixXd& Y, double& sigma2, int M, double mu, int max_iter) {
    // initial guess
    MatrixXd X = pts.replicate(1, 1);
    Y = MatrixXd::Zero(M, 3);
    for (int i = 0; i < M; i ++) {
        Y(i, 1) = 0.1 / static_cast<double>(M) * static_cast<double>(i);
        Y(i, 0) = 0;
        Y(i, 2) = 0;
    }
    
    int N = X.rows();
    int D = 3;

    // diff_xy should be a (M * N) matrix
    MatrixXd diff_xy = MatrixXd::Zero(M, N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
        }
    }

    // initialize sigma2
    sigma2 = diff_xy.sum() / static_cast<double>(D * M * N);

    for (int it = 0; it < max_iter; it ++) {
        // update diff_xy
        for (int i = 0; i < M; i ++) {
            for (int j = 0; j < N; j ++) {
                diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
            }
        }

        MatrixXd P = (-0.5 * diff_xy / sigma2).array().exp();
        MatrixXd P_stored = P.replicate(1, 1);
        double c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) * static_cast<double>(M)/N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);

        MatrixXd Pt1 = P.colwise().sum(); 
        MatrixXd P1 = P.rowwise().sum();
        double Np = P1.sum();
        MatrixXd PX = P * X;

        MatrixXd P1_expanded = MatrixXd::Zero(M, D);
        P1_expanded.col(0) = P1;
        P1_expanded.col(1) = P1;
        P1_expanded.col(2) = P1;

        Y = PX.cwiseQuotient(P1_expanded);

        double numerator = 0;
        double denominator = 0;

        for (int m = 0; m < M; m ++) {
            for (int n = 0; n < N; n ++) {
                numerator += P(m, n)*diff_xy(m, n);
                denominator += P(m, n)*D;
            }
        }

        sigma2 = numerator / denominator;
    }
}

// link to original code: https://stackoverflow.com/a/46303314
void remove_row(MatrixXd& matrix, unsigned int rowToRemove) {
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);

    matrix.conservativeResize(numRows,numCols);
}

MatrixXd sort_pts (MatrixXd Y_0) {
    int N = Y_0.rows();
    MatrixXd Y_0_sorted = MatrixXd::Zero(N, 3);
    std::vector<MatrixXd> Y_0_sorted_vec = {};
    std::vector<bool> selected_node(N, false);
    selected_node[0] = true;
    int last_visited_b = 0;

    MatrixXd G = MatrixXd::Zero(N, N);
    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < N; j ++) {
            G(i, j) = (Y_0.row(i) - Y_0.row(j)).squaredNorm();
        }
    }

    int reverse = 0;
    int counter = 0;
    int reverse_on = 0;
    int insertion_counter = 0;

    while (counter < N-1) {
        double minimum = INFINITY;
        int a = 0;
        int b = 0;

        for (int m = 0; m < N; m ++) {
            if (selected_node[m] == true) {
                for (int n = 0; n < N; n ++) {
                    if ((!selected_node[n]) && (G(m, n) != 0.0)) {
                        if (minimum > G(m, n)) {
                            minimum = G(m, n);
                            a = m;
                            b = n;
                        }
                    }
                }
            }
        }

        if (counter == 0) {
            Y_0_sorted_vec.push_back(Y_0.row(a));
            Y_0_sorted_vec.push_back(Y_0.row(b));
        }
        else {
            if (last_visited_b != a) {
                reverse += 1;
                reverse_on = a;
                insertion_counter = 1;
            }
            
            if (reverse % 2 == 1) {
                auto it = find(Y_0_sorted_vec.begin(), Y_0_sorted_vec.end(), Y_0.row(a));
                Y_0_sorted_vec.insert(it, Y_0.row(b));
            }
            else if (reverse != 0) {
                auto it = find(Y_0_sorted_vec.begin(), Y_0_sorted_vec.end(), Y_0.row(reverse_on));
                Y_0_sorted_vec.insert(it + insertion_counter, Y_0.row(b));
                insertion_counter += 1;
            }
            else {
                Y_0_sorted_vec.push_back(Y_0.row(b));
            }
        }

        last_visited_b = b;
        selected_node[b] = true;
        counter += 1;
    }

    // copy to Y_0_sorted
    for (int i = 0; i < N; i ++) {
        Y_0_sorted.row(i) = Y_0_sorted_vec[i];
    }

    return Y_0_sorted;
}

bool isBetween (MatrixXd x, MatrixXd a, MatrixXd b) {
    bool in_bound = true;

    for (int i = 0; i < 3; i ++) {
        if (!(a(0, i)-0.0001 <= x(0, i) && x(0, i) <= b(0, i)+0.0001) && 
            !(b(0, i)-0.0001 <= x(0, i) && x(0, i) <= a(0, i)+0.0001)) {
            in_bound = false;
        }
    }
    
    return in_bound;
}

std::vector<MatrixXd> line_sphere_intersection (MatrixXd point_A, MatrixXd point_B, MatrixXd sphere_center, double radius) {
    std::vector<MatrixXd> intersections = {};
    
    double a = pt2pt_dis_sq(point_A, point_B);
    double b = 2 * ((point_B(0, 0) - point_A(0, 0))*(point_A(0, 0) - sphere_center(0, 0)) + 
                    (point_B(0, 1) - point_A(0, 1))*(point_A(0, 1) - sphere_center(0, 1)) + 
                    (point_B(0, 2) - point_A(0, 2))*(point_A(0, 2) - sphere_center(0, 2)));
    double c = pt2pt_dis_sq(point_A, sphere_center) - pow(radius, 2);
    
    double delta = pow(b, 2) - 4*a*c;

    double d1 = (-b + sqrt(delta)) / (2*a);
    double d2 = (-b - sqrt(delta)) / (2*a);

    if (delta < 0) {
        // no solution
        return {};
    }
    else if (delta > 0) {
        // two solutions
        // the first one
        double x1 = point_A(0, 0) + d1*(point_B(0, 0) - point_A(0, 0));
        double y1 = point_A(0, 1) + d1*(point_B(0, 1) - point_A(0, 1));
        double z1 = point_A(0, 2) + d1*(point_B(0, 2) - point_A(0, 2));
        MatrixXd pt1(1, 3);
        pt1 << x1, y1, z1;

        // the second one
        double x2 = point_A(0, 0) + d2*(point_B(0, 0) - point_A(0, 0));
        double y2 = point_A(0, 1) + d2*(point_B(0, 1) - point_A(0, 1));
        double z2 = point_A(0, 2) + d2*(point_B(0, 2) - point_A(0, 2));
        MatrixXd pt2(1, 3);
        pt2 << x2, y2, z2;

        if (isBetween(pt1, point_A, point_B)) {
            intersections.push_back(pt1);
        }
        if (isBetween(pt2, point_A, point_B)) {
            intersections.push_back(pt2);
        }
    }
    else {
        // one solution
        d1 = -b / (2*a);
        double x1 = point_A(0, 0) + d1*(point_B(0, 0) - point_A(0, 0));
        double y1 = point_A(0, 1) + d1*(point_B(0, 1) - point_A(0, 1));
        double z1 = point_A(0, 2) + d1*(point_B(0, 2) - point_A(0, 2));
        MatrixXd pt1(1, 3);
        pt1 << x1, y1, z1;

        if (isBetween(pt1, point_A, point_B)) {
            intersections.push_back(pt1);
        }
    }
    
    return intersections;
}

std::tuple<MatrixXd, MatrixXd, double> shortest_dist_between_lines (MatrixXd a0, MatrixXd a1, MatrixXd b0, MatrixXd b1, bool clamp) {
    MatrixXd A = a1 - a0;
    MatrixXd B = b1 - b0;
    MatrixXd A_normalized = A / A.norm();
    MatrixXd B_normalized = B / B.norm();

    MatrixXd cross = cross_product(A_normalized, B_normalized);
    double denom = cross.squaredNorm();

    // If lines are parallel (denom=0) test if lines overlap.
    // If they don't overlap then there is a closest point solution.
    // If they do overlap, there are infinite closest positions, but there is a closest distance
    if (denom == 0) {
        double d0 = dot_product(A_normalized, b0-a0);

        // Overlap only possible with clamping
        if (clamp) {
            double d1 = dot_product(A_normalized, b1-a0);

            // is segment B before A?
            if (d0 <= 0 && d1 <= 0) {
                if (abs(d0) < abs(d1)) {
                    return {a0, b0, (a0-b0).norm()};
                }
                else {
                    return {a0, b1, (a0-b1).norm()};
                }
            }

            // is segment B after A?
            else if (d0 >= A.norm() && d1 >= A.norm()) {
                if (abs(d0) < abs(d1)) {
                    return {a1, b0, (a1-b0).norm()};
                }
                else {
                    return {a1, b1, (a1-b1).norm()};
                }
            }
        }

        // Segments overlap, return distance between parallel segments
        return {MatrixXd::Zero(1, 3), MatrixXd::Zero(1, 3), (d0*A_normalized+a0-b0).norm()};
    }

    // Lines criss-cross: Calculate the projected closest points
    MatrixXd t = b0 - a0;
    MatrixXd tempA = MatrixXd::Zero(3, 3);
    tempA.block(0, 0, 1, 3) = t;
    tempA.block(1, 0, 1, 3) = B_normalized;
    tempA.block(2, 0, 1, 3) = cross;

    MatrixXd tempB = MatrixXd::Zero(3, 3);
    tempB.block(0, 0, 1, 3) = t;
    tempB.block(1, 0, 1, 3) = A_normalized;
    tempB.block(2, 0, 1, 3) = cross;

    double t0 = tempA.determinant() / denom;
    double t1 = tempB.determinant() / denom;

    MatrixXd pA = a0 + (A_normalized * t0);  // projected closest point on segment A
    MatrixXd pB = b0 + (B_normalized * t1);  // projected closest point on segment B

    // clamp
    if (clamp) {
        if (t0 < 0) {
            pA = a0.replicate(1, 1);
        }
        else if (t0 > A.norm()) {
            pA = a1.replicate(1, 1);
        }

        if (t1 < 0) {
            pB = b0.replicate(1, 1);
        }
        else if (t1 > B.norm()) {
            pB = b1.replicate(1, 1);
        }

        // clamp projection A
        if (t0 < 0 || t0 > A.norm()) {
            double dot = dot_product(B_normalized, pA-b0);
            if (dot < 0) {
                dot = 0;
            }
            else if (dot > B.norm()) {
                dot = B.norm();
            }
            pB = b0 + (B_normalized * dot);
        }

        // clamp projection B
        if (t1 < 0 || t1 > B.norm()) {
            double dot = dot_product(A_normalized, pB-a0);
            if (dot < 0) {
                dot = 0;
            }
            else if (dot > A.norm()) {
                dot = A.norm();
            }
            pA = a0 + (A_normalized * dot);
        }
    }

    return {pA, pB, (pA-pB).norm()};
}

static GRBEnv& getGRBEnv () {
    static GRBEnv env;
    return env;
}

std::tuple<MatrixXd, MatrixXd> nearest_points_line_segments (MatrixXd last_template, Matrix2Xi E) {
    // find the nearest points on the line segments
    // refer to the website https://math.stackexchange.com/questions/846054/closest-points-on-two-line-segments
    MatrixXd startPts(4, E.cols()*E.cols()); // Matrix: 3 * E^2: startPts.col(E*cols()*i + j) is the nearest point on edge i w.r.t. j
    MatrixXd endPts(4, E.cols()*E.cols()); // Matrix: 3 * E^2: endPts.col(E*cols()*i + j) is the nearest point on edge j w.r.t. i
    for (int i = 0; i < E.cols(); ++i)
    {
        Vector3d P1 = last_template.col(E(0, i));
        Vector3d P2 = last_template.col(E(1, i));
        // cout << "P1:" << endl;
        // cout << P1 << endl << endl;
        // cout << "P2:" << endl;
        // cout << P2 << endl << endl;
        for (int j = 0; j < E.cols(); ++j)
        {
            Vector3d P3 = last_template.col(E(0, j));
            Vector3d P4 = last_template.col(E(1, j));
            
            // cout << "P3:" << endl;
            // cout << P3 << endl << endl;
            // cout << "P4:" << endl;
            // cout << P4 << endl << endl;

            float R21 = (P2-P1).squaredNorm();
            float R22 = (P4-P3).squaredNorm();
            float D4321 = (P4-P3).dot(P2-P1);
            float D3121 = (P3-P1).dot(P2-P1);
            float D4331 = (P4-P3).dot(P3-P1);

            // cout << "original s:" << (-D4321*D4331+D3121*R22)/(R21*R22-D4321*D4321) << endl;
            // cout << "original t:" << (D4321*D3121-D4331*R21)/(R21*R22-D4321*D4321) << endl;

            float s;
            float t;
            
            if (R21*R22-D4321*D4321 != 0)
            {
                s = std::min(std::max((-D4321*D4331+D3121*R22)/(R21*R22-D4321*D4321), 0.0f), 1.0f);
                t = std::min(std::max((D4321*D3121-D4331*R21)/(R21*R22-D4321*D4321), 0.0f), 1.0f);
            } else {
                // means P1 P2 P3 P4 are on the same line
                float P13 = (P3 - P1).squaredNorm();
                s = 0; t = 0;
                float P14 = (P4 - P1).squaredNorm();
                if (P14 < P13) {
                    s = 0; t = 1;
                }
                float P23 = (P3 - P2).squaredNorm();
                if (P23 < P14 && P23 < P13)
                {
                    s = 1; t = 0;
                }
                float P24 = (P4 - P2).squaredNorm();
                if (P24 < P23 && P24 < P14 && P24 < P13) {
                    s = 1; t = 1;
                }
            }
            // cout << "s: " << s << endl;
            // cout << "t: " << t << endl;

            for (int dim = 0; dim < 3; ++dim)
            {
                startPts(dim, E.cols()*i+j) = (1-s)*P1(dim)+s*P2(dim);
                endPts(dim, E.cols()*i+j) = (1-t)*P3(dim)+t*P4(dim);
            }
            startPts(3, E.cols()*i+j) = s;
            endPts(3, E.cols()*i+j) = t;
        }
    }
    return {startPts, endPts};
}

static GRBQuadExpr buildDifferencingQuadraticTerm(GRBVar* point_a, GRBVar* point_b, const size_t num_vars_per_point) {
    GRBQuadExpr expr;

    // Build the main diagonal
    const std::vector<double> main_diag(num_vars_per_point, 1.0);
    expr.addTerms(main_diag.data(), point_a, point_a, (int)num_vars_per_point);
    expr.addTerms(main_diag.data(), point_b, point_b, (int)num_vars_per_point);

    // Build the off diagonal - use -2 instead of -1 because the off diagonal terms are the same
    const std::vector<double> off_diagonal(num_vars_per_point, -2.0);
    expr.addTerms(off_diagonal.data(), point_a, point_b, (int)num_vars_per_point);

    return expr;
}

// Matrix3Xf Optimizer::operator()(const Matrix3Xf& Y, const Matrix2Xi& E, const std::vector<CDCPD::FixedPoint>& fixed_points, const bool self_intersection, const bool interaction_constrain)
MatrixXd cdcpd2_post_processing (MatrixXd Y_0, MatrixXd Y, Matrix2Xi E, MatrixXd initial_template) {
    // Y: Y^t in Eq. (21)
    // E: E in Eq. (21)
    // auto [nearestPts, normalVecs] = nearest_points_and_normal(last_template);
    // auto Y_force = force_pts(nearestPts, normalVecs, Y);
    MatrixXd Y_opt(Y.rows(), Y.cols());
    GRBVar* vars = nullptr;
    try
    {
        const ssize_t num_vectors = Y.cols();
        const ssize_t num_vars = 3 * num_vectors;

        GRBEnv& env = getGRBEnv();

        // Disables logging to file and logging to console (with a 0 as the value of the flag)
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);
        model.set("ScaleFlag", "0");
		// model.set("DualReductions", 0);
        model.set("FeasibilityTol", "0.01");
		// model.set("OutputFlag", "1");

        // Add the vars to the model
        // Note that variable bound is important, without a bound, Gurobi defaults to 0, which is clearly unwanted
        const std::vector<double> lb(num_vars, -GRB_INFINITY);
        const std::vector<double> ub(num_vars, GRB_INFINITY);
        vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int) num_vars);
        model.update();

        // Add the edge constraints
        if (initial_template.rows() != 0) {
            for (ssize_t i = 0; i < E.cols(); ++i) {
                model.addQConstr(
                            buildDifferencingQuadraticTerm(&vars[E(0, i) * 3], &vars[E(1, i) * 3], 3),
                            GRB_LESS_EQUAL,
                            1.05 * 1.05 * (initial_template.col(E(0, i)) - initial_template.col(E(1, i))).squaredNorm(),
                            "upper_edge_" + std::to_string(E(0, i)) + "_to_" + std::to_string(E(1, i)));
            }
            model.update();
        }

        auto [startPts, endPts] = nearest_points_line_segments(Y_0, E);
        for (int row = 0; row < E.cols(); ++row)
        {
            Vector3d P1 = Y_0.col(E(0, row));
            Vector3d P2 = Y_0.col(E(1, row));
            for (int col = 0; col < E.cols(); ++col)
            {
                float s = startPts(3, row*E.cols() + col);
                float t = endPts(3, row*E.cols() + col);
                Vector3d P3 = Y_0.col(E(0, col));
                Vector3d P4 = Y_0.col(E(1, col));
                float l = (endPts.col(row*E.cols() + col).topRows(3) - startPts.col(row*E.cols() + col).topRows(3)).norm();
                if (!P1.isApprox(P3) && !P1.isApprox(P4) && !P2.isApprox(P3) && !P2.isApprox(P4) && l <= 0.02) {
                    // model.addConstr((vars[E(0, col)*3 + 0] - startPts(0, row*E.cols() + col))*(endPts(0, row*E.cols() + col) - startPts(0, row*E.cols() + col)) +
                    //                 (vars[E(0, col)*3 + 1] - startPts(1, row*E.cols() + col))*(endPts(1, row*E.cols() + col) - startPts(1, row*E.cols() + col)) +
                    //                 (vars[E(0, col)*3 + 2] - startPts(2, row*E.cols() + col))*(endPts(2, row*E.cols() + col) - startPts(2, row*E.cols() + col)) >= 0);
                    // model.addConstr((vars[E(1, col)*3 + 0] - startPts(0, row*E.cols() + col))*(endPts(0, row*E.cols() + col) - startPts(0, row*E.cols() + col)) +
                    //                 (vars[E(1, col)*3 + 1] - startPts(1, row*E.cols() + col))*(endPts(1, row*E.cols() + col) - startPts(1, row*E.cols() + col)) +
                    //                 (vars[E(1, col)*3 + 2] - startPts(2, row*E.cols() + col))*(endPts(2, row*E.cols() + col) - startPts(2, row*E.cols() + col)) >= 0);
                    model.addConstr(((vars[E(0, col)*3 + 0]*(1-t) + vars[E(1, col)*3 + 0]*t) - (vars[E(0, row)*3 + 0]*(1-s) + vars[E(1, row)*3 + 0]*s))
                                        *(endPts(0, row*E.cols() + col) - startPts(0, row*E.cols() + col)) +
                                    ((vars[E(0, col)*3 + 1]*(1-t) + vars[E(1, col)*3 + 1]*t) - (vars[E(0, row)*3 + 1]*(1-s) + vars[E(1, row)*3 + 1]*s))
                                        *(endPts(1, row*E.cols() + col) - startPts(1, row*E.cols() + col)) +
                                    ((vars[E(0, col)*3 + 2]*(1-t) + vars[E(1, col)*3 + 2]*t) - (vars[E(0, row)*3 + 2]*(1-s) + vars[E(1, row)*3 + 2]*s))
                                        *(endPts(2, row*E.cols() + col) - startPts(2, row*E.cols() + col)) >= 0.01 * l);
                    std::cout << "0.01 * l = " << 0.01 * l << std::endl;
                }
            }
        }

        // Build the objective function
        GRBQuadExpr objective_fn(0);
        for (ssize_t i = 0; i < num_vectors; ++i)
        {
            const auto expr0 = vars[i * 3 + 0] - Y(0, i);
            const auto expr1 = vars[i * 3 + 1] - Y(1, i);
            const auto expr2 = vars[i * 3 + 2] - Y(2, i);
            objective_fn += expr0 * expr0;
            objective_fn += expr1 * expr1;
            objective_fn += expr2 * expr2;
        }
        model.setObjective(objective_fn, GRB_MINIMIZE);
        model.update();

        // Find the optimal solution, and extract it
        model.optimize();
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL || model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL)  // || model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL || model.get(GRB_IntAttr_Status) == GRB_NUMERIC || modelGRB_INF_OR_UNBD)
        {
            // std::cout << "Y" << std::endl;
            // std::cout << Y << std::endl;
            for (ssize_t i = 0; i < num_vectors; i++)
            {
                Y_opt(0, i) = vars[i * 3 + 0].get(GRB_DoubleAttr_X);
                Y_opt(1, i) = vars[i * 3 + 1].get(GRB_DoubleAttr_X);
                Y_opt(2, i) = vars[i * 3 + 2].get(GRB_DoubleAttr_X);
            }
        }
        else
        {
            std::cout << "Status: " << model.get(GRB_IntAttr_Status) << std::endl;
            exit(-1);
        }
    }
    catch(GRBException& e)
    {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch(...)
    {
        std::cout << "Exception during optimization" << std::endl;
    }

    delete[] vars;
	// auto [nearestPts, normalVecs] = nearest_points_and_normal(last_template);
    // return force_pts(nearestPts, normalVecs, Y_opt);
    
    MatrixXd ret =  Y_opt.transpose();
	return ret;
}

MatrixXd post_processing (MatrixXd Y_0, MatrixXd Y, double check_distance, double dlo_diameter, int nodes_per_dlo, bool clamp) {
    MatrixXd Y_processed = MatrixXd::Zero(Y.rows(), Y.cols());
    int num_of_dlos = Y.rows() / nodes_per_dlo;

    GRBVar* vars = nullptr;
    GRBEnv& env = getGRBEnv();
    env.set(GRB_IntParam_OutputFlag, 0);
    GRBModel model(env);
    // model.set("ScaleFlag", "0");
    // model.set("FeasibilityTol", "0.01");

    // add vars to the model
    const ssize_t num_of_vars = 3 * Y.rows();
    const std::vector<double> lower_bound(num_of_vars, -GRB_INFINITY);
    const std::vector<double> upper_bound(num_of_vars, GRB_INFINITY);
    vars = model.addVars(lower_bound.data(), upper_bound.data(), nullptr, nullptr, nullptr, (int) num_of_vars);

    // add constraints to the model
    for (int i = 0; i < Y.rows()-1; i ++) {
        for (int j = i; j < Y.rows()-1; j ++) {
            // edge 1: y_i, y_{i+1}
            // edge 2: y_j, y_{j+1}
            if (abs(i - j) <= 1) {
                continue;
            }

            // for multiple dlos
            if (num_of_dlos > 1) {
                if ((i+1) % nodes_per_dlo == 0 || (j+1) % nodes_per_dlo == 0) {
                    continue;
                }
            }

            auto[temp1, temp2, cur_shortest_dist] = shortest_dist_between_lines(Y.row(i), Y.row(i+1), Y.row(j), Y.row(j+1), true);
            if (cur_shortest_dist >= check_distance) {
                continue;
            }

            auto[pA, pB, dist] = shortest_dist_between_lines(Y_0.row(i), Y_0.row(i+1), Y_0.row(j), Y_0.row(j+1), clamp);

            std::cout << "Adding self-intersection constraint between E(" << i << ", " << i+1 << ") and E(" << j << ", " << j+1 << ")" << std::endl;

            // pA is the point on edge y_i, y_{i+1}
            // pB is the point on edge y_j, y_{j+1}
            // the below definition should be consistent with CDCPD2's Eq 18-21
            double r_i = ((pA - Y.row(i+1)).array() / (Y.row(i) - Y.row(i+1)).array())(0, 0);
            double r_j = ((pB - Y.row(j+1)).array() / (Y.row(j) - Y.row(j+1)).array())(0, 0);

            std::cout << "r_i, r_j = " << r_i << ", " << r_j << std::endl;

            // === Python ===
            // pA_var = r_i*vars[i] + (1 - r_i)*vars[i+1]
            // pB_var = r_j*vars[j] + (1 - r_j)*vars[j+1]
            // // model.addConstr(operator.ge(np.sum(np.square(pA_var - pB_var)), dlo_diameter**2))
            // model.addConstr(operator.ge(((pA_var[0] - pB_var[0])*(pA[0] - pB[0]) +
            //                              (pA_var[1] - pB_var[1])*(pA[1] - pB[1]) +
            //                              (pA_var[2] - pB_var[2])*(pA[2] - pB[2])) / np.linalg.norm(pA - pB), dlo_diameter))

            // vars can be seen as a flattened array of size len(Y)*3
            model.addConstr((((r_i*vars[3*i] + (1 - r_i)*vars[3*(i+1)]) - (r_j*vars[3*j] + (1 - r_j)*vars[3*(j+1)])) * (pA(0, 0) - pB(0, 0)) +
                             ((r_i*vars[3*i+1] + (1 - r_i)*vars[3*(i+1)+1]) - (r_j*vars[3*j+1] + (1 - r_j)*vars[3*(j+1)+1])) * (pA(0, 1) - pB(0, 1)) +
                             ((r_i*vars[3*i+2] + (1 - r_i)*vars[3*(i+1)+2]) - (r_j*vars[3*j+2] + (1 - r_j)*vars[3*(j+1)+2])) * (pA(0, 2) - pB(0, 2))) / (pA - pB).norm()
                            >= dlo_diameter);
        }
    }

    // objective function (as close to Y as possible)
    GRBQuadExpr objective_fn(0);
    for (ssize_t i = 0; i < Y.rows(); i ++) {
        const auto expr0 = vars[3*i] - Y(i, 0);
        const auto expr1 = vars[3*i+1] - Y(i, 1);
        const auto expr2 = vars[3*i+2] - Y(i, 2);
        objective_fn += expr0 * expr0;
        objective_fn += expr1 * expr1;
        objective_fn += expr2 * expr2;
    }
    model.setObjective(objective_fn, GRB_MINIMIZE);

    model.update();
    model.optimize();
    if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL || model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL) {
        for (ssize_t i = 0; i < Y.rows(); i ++) {
            Y_processed(i, 0) = vars[3*i].get(GRB_DoubleAttr_X);
            Y_processed(i, 1) = vars[3*i+1].get(GRB_DoubleAttr_X);
            Y_processed(i, 2) = vars[3*i+2].get(GRB_DoubleAttr_X);
        }
    }
    else {
        std::cout << "Status: " << model.get(GRB_IntAttr_Status) << std::endl;
        exit(-1);
    }

    return Y_processed;
}

std::vector<GRBLinExpr> build_GW (MatrixXd G, GRBVar* w_vars) {
    std::vector<GRBLinExpr> result = {};

    // let w_vars be [w0_x, w0_y, w0_z, w1_x, w1_y, w1_z, ...]
    for (int row = 0; row < G.rows(); row ++) {
        for (int col = 0; col < 3; col ++) {
            GRBLinExpr temp(0);
            for (int cursor = 0; cursor < G.rows(); cursor ++) {
                if (G(row, cursor) > 1e-4) {
                    temp += G(row, cursor) * w_vars[cursor*3 + col];
                }
            }
            result.push_back(temp);
        }
    }

    return result;
}

GRBQuadExpr build_tr_WTGW (MatrixXd G, GRBVar* w_vars) {
    std::vector<std::vector<GRBVar>> W = {};
    for (int row = 0; row < G.rows(); row ++) {
        std::vector<GRBVar> cur_row = {};
        for (int col = 0; col < 3; col ++) {
            cur_row.push_back(w_vars[row*3 + col]);
        }
        W.push_back(cur_row);
    }

    GRBQuadExpr tr_WTGW(0);

    std::vector<std::vector<GRBLinExpr>> WTG = {};
    for (int row = 0; row < 3; row ++) {
        std::vector<GRBLinExpr> cur_row = {};
        for (int col = 0; col < G.cols(); col ++) {
            GRBLinExpr temp(0);
            for (int cursor = 0; cursor < G.rows(); cursor ++) {
                if (G(cursor, col) > 1e-4) {
                    temp += W[cursor][row] * G(cursor, col);
                }
            }
            cur_row.push_back(temp);
        }
        WTG.push_back(cur_row);
    }

    // WTG is 3xM
    // WTGW is 3x3
    int M = G.rows();
    for (int i = 0; i < 3; i ++) {
        for (int cursor = 0; cursor < M; cursor ++) {
            tr_WTGW += WTG[i][cursor] * W[cursor][i];
        }
    }

    return tr_WTGW;
} 

MatrixXd post_processing_dev_2 (MatrixXd Y_0, MatrixXd Y, Matrix2Xi E, MatrixXd initial_template, MatrixXd G) {
    // Y: Y^t in Eq. (21)
    // E: E in Eq. (21)
    // auto [nearestPts, normalVecs] = nearest_points_and_normal(last_template);
    // auto Y_force = force_pts(nearestPts, normalVecs, Y);
    MatrixXd Y_opt(Y.rows(), Y.cols());
    MatrixXd W_opt(Y.rows(), Y.cols());

    GRBVar* vars = nullptr;
    try
    {
        const ssize_t num_vectors = Y.cols();
        const ssize_t num_vars = 3 * num_vectors;

        GRBEnv& env = getGRBEnv();

        // Disables logging to file and logging to console (with a 0 as the value of the flag)
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);
        model.set("ScaleFlag", "0");
		// model.set("DualReductions", 0);
        model.set("FeasibilityTol", "0.01");
		// model.set("OutputFlag", "1");

        // Add the vars to the model
        // Note that variable bound is important, without a bound, Gurobi defaults to 0, which is clearly unwanted
        const std::vector<double> lb(num_vars, -GRB_INFINITY);
        const std::vector<double> ub(num_vars, GRB_INFINITY);
        vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int) num_vars);
        model.update();

        auto stamp = std::chrono::high_resolution_clock::now();
        std::vector<GRBLinExpr> GW_vars = build_GW(G, vars);
        auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - stamp).count() / 1000.0;
        ROS_INFO_STREAM("Build GW: " + std::to_string(time_diff) + " ms");

        // std::cout << "built GW" << std::endl;

        stamp = std::chrono::high_resolution_clock::now();
        if (initial_template.rows() != 0) {
            for (int i = 0; i < G.rows()-1; i ++) {
                if ((i+1) % 20 == 0) {
                    continue;
                }
                model.addQConstr((Y_0(0, i) + GW_vars[i*3 + 0] - Y_0(0, i+1) - GW_vars[(i+1)*3 + 0])*(Y_0(0, i) + GW_vars[i*3 + 0] - Y_0(0, i+1) - GW_vars[(i+1)*3 + 0]) + 
                                 (Y_0(1, i) + GW_vars[i*3 + 1] - Y_0(1, i+1) - GW_vars[(i+1)*3 + 1])*(Y_0(1, i) + GW_vars[i*3 + 1] - Y_0(1, i+1) - GW_vars[(i+1)*3 + 1]) +
                                 (Y_0(2, i) + GW_vars[i*3 + 2] - Y_0(2, i+1) - GW_vars[(i+1)*3 + 2])*(Y_0(2, i) + GW_vars[i*3 + 2] - Y_0(2, i+1) - GW_vars[(i+1)*3 + 2]), 
                                 GRB_LESS_EQUAL,
                                 1.05 * 1.05 * (initial_template.col(i) - initial_template.col(i+1)).squaredNorm());
            }
            model.update();
        }
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - stamp).count() / 1000.0;
        ROS_INFO_STREAM("Add stretching constraint: " + std::to_string(time_diff) + " ms");

        // std::cout << "added stretching constraint" << std::endl;

        stamp = std::chrono::high_resolution_clock::now();
        auto [startPts, endPts] = nearest_points_line_segments(Y_0, E);
        for (int row = 0; row < E.cols(); ++row)
        {
            Vector3d P1 = Y_0.col(E(0, row));
            Vector3d P2 = Y_0.col(E(1, row));
            for (int col = 0; col < E.cols(); ++col)
            {
                float s = startPts(3, row*E.cols() + col);
                float t = endPts(3, row*E.cols() + col);
                Vector3d P3 = Y_0.col(E(0, col));
                Vector3d P4 = Y_0.col(E(1, col));
                float l = (endPts.col(row*E.cols() + col).topRows(3) - startPts.col(row*E.cols() + col).topRows(3)).norm();
                if (!P1.isApprox(P3) && !P1.isApprox(P4) && !P2.isApprox(P3) && !P2.isApprox(P4) && l <= 0.02) {
                    // model.addConstr(((vars[E(0, col)*3 + 0]*(1-t) + vars[E(1, col)*3 + 0]*t) - (vars[E(0, row)*3 + 0]*(1-s) + vars[E(1, row)*3 + 0]*s))
                    //                     *(endPts(0, row*E.cols() + col) - startPts(0, row*E.cols() + col)) +
                    //                 ((vars[E(0, col)*3 + 1]*(1-t) + vars[E(1, col)*3 + 1]*t) - (vars[E(0, row)*3 + 1]*(1-s) + vars[E(1, row)*3 + 1]*s))
                    //                     *(endPts(1, row*E.cols() + col) - startPts(1, row*E.cols() + col)) +
                    //                 ((vars[E(0, col)*3 + 2]*(1-t) + vars[E(1, col)*3 + 2]*t) - (vars[E(0, row)*3 + 2]*(1-s) + vars[E(1, row)*3 + 2]*s))
                    //                     *(endPts(2, row*E.cols() + col) - startPts(2, row*E.cols() + col)) >= 0.01 * l);
                    model.addConstr((((Y_0(0, E(0, col)) + GW_vars[E(0, col)*3 + 0]) * (1-t) + (Y_0(0, E(1, col)) + GW_vars[E(1, col)*3 + 0]) * t) - ((Y_0(0, E(0, row)) + GW_vars[E(0, row)*3 + 0])*(1-s) + (Y_0(0, E(1, row)) + GW_vars[E(1, row)*3 + 0])*s))
                                        *(endPts(0, row*E.cols() + col) - startPts(0, row*E.cols() + col)) +
                                    (((Y_0(1, E(0, col)) + GW_vars[E(0, col)*3 + 1]) * (1-t) + (Y_0(1, E(1, col)) + GW_vars[E(1, col)*3 + 1]) * t) - ((Y_0(1, E(0, row)) + GW_vars[E(0, row)*3 + 1])*(1-s) + (Y_0(1, E(1, row)) + GW_vars[E(1, row)*3 + 1])*s))
                                        *(endPts(1, row*E.cols() + col) - startPts(1, row*E.cols() + col)) +
                                    (((Y_0(2, E(0, col)) + GW_vars[E(0, col)*3 + 2]) * (1-t) + (Y_0(2, E(1, col)) + GW_vars[E(1, col)*3 + 2]) * t) - ((Y_0(2, E(0, row)) + GW_vars[E(0, row)*3 + 2])*(1-s) + (Y_0(2, E(1, row)) + GW_vars[E(1, row)*3 + 2])*s))
                                        *(endPts(2, row*E.cols() + col) - startPts(2, row*E.cols() + col)) >= 0.01 * l);
                }
            }
        }
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - stamp).count() / 1000.0;
        ROS_INFO_STREAM("Add self-intersection constraint: " + std::to_string(time_diff) + " ms");

        // std::cout << "added self-intersection constraint" << std::endl;

        stamp = std::chrono::high_resolution_clock::now();
        GRBQuadExpr tr_WTGW = build_tr_WTGW(G, vars);
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - stamp).count() / 1000.0;
        ROS_INFO_STREAM("Build tr(W^T*G*W): " + std::to_string(time_diff) + " ms");
        // std::cout << "built tr(W^T*G*W)" << std::endl;

        // Build the objective function
        stamp = std::chrono::high_resolution_clock::now();
        GRBQuadExpr objective_fn(0);
        objective_fn += tr_WTGW;
        for (ssize_t i = 0; i < num_vectors; ++i)
        {
            // const auto expr0 = GW_vars[i * 3 + 0] - (Y(0, i) - Y_0(0, i));
            // const auto expr1 = GW_vars[i * 3 + 1] - (Y(1, i) - Y_0(1, i));
            // const auto expr2 = GW_vars[i * 3 + 2] - (Y(2, i) - Y_0(2, i));
            // objective_fn += expr0 * expr0;
            // objective_fn += expr1 * expr1;
            // objective_fn += expr2 * expr2;
            objective_fn += GW_vars[i * 3 + 0]*GW_vars[i * 3 + 0] - 2*GW_vars[i * 3 + 0]*(Y(0, i) - Y_0(0, i)) + (Y(0, i) - Y_0(0, i))*(Y(0, i) - Y_0(0, i));
            objective_fn += GW_vars[i * 3 + 1]*GW_vars[i * 3 + 1] - 2*GW_vars[i * 3 + 1]*(Y(1, i) - Y_0(1, i)) + (Y(1, i) - Y_0(1, i))*(Y(1, i) - Y_0(1, i));
            objective_fn += GW_vars[i * 3 + 2]*GW_vars[i * 3 + 2] - 2*GW_vars[i * 3 + 2]*(Y(2, i) - Y_0(2, i)) + (Y(2, i) - Y_0(2, i))*(Y(2, i) - Y_0(2, i));
        }
        model.setObjective(objective_fn, GRB_MINIMIZE);
        model.update();
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - stamp).count() / 1000.0;
        ROS_INFO_STREAM("Build objective: " + std::to_string(time_diff) + " ms");

        // std::cout << "built objective function" << std::endl;

        // Find the optimal solution, and extract it
        stamp = std::chrono::high_resolution_clock::now();
        model.optimize();
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - stamp).count() / 1000.0;
        ROS_INFO_STREAM("Optimize model: " + std::to_string(time_diff) + " ms");

        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL || model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL)  // || model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL || model.get(GRB_IntAttr_Status) == GRB_NUMERIC || modelGRB_INF_OR_UNBD)
        {
            // std::cout << "Y" << std::endl;
            // std::cout << Y << std::endl;
            for (ssize_t i = 0; i < num_vectors; i++)
            {
                W_opt(0, i) = vars[i * 3 + 0].get(GRB_DoubleAttr_X);
                W_opt(1, i) = vars[i * 3 + 1].get(GRB_DoubleAttr_X);
                W_opt(2, i) = vars[i * 3 + 2].get(GRB_DoubleAttr_X);
            }
        }
        else
        {
            std::cout << "Status: " << model.get(GRB_IntAttr_Status) << std::endl;
            exit(-1);
        }
    }
    catch(GRBException& e)
    {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch(...)
    {
        std::cout << "Exception during optimization" << std::endl;
    }

    delete[] vars;
	// auto [nearestPts, normalVecs] = nearest_points_and_normal(last_template);
    // return force_pts(nearestPts, normalVecs, Y_opt);

    std::cout << "extracted output" << std::endl;

    MatrixXd ret = Y_0.transpose() + G * W_opt.transpose();
	return ret;
}

// node color and object color are in rgba format and range from 0-1
visualization_msgs::MarkerArray MatrixXd2MarkerArray (MatrixXd Y,
                                                      std::string marker_frame, 
                                                      std::string marker_ns, 
                                                      std::vector<std::vector<int>> node_colors, 
                                                      std::vector<std::vector<int>> line_colors, 
                                                      double node_scale,
                                                      double line_scale,
                                                      int num_of_dlos,
                                                      int nodes_per_dlo,
                                                      std::vector<int> visible_nodes, 
                                                      std::vector<float> occluded_node_color,
                                                      std::vector<float> occluded_line_color) {    // publish the results as a marker array
    
    visualization_msgs::MarkerArray results = visualization_msgs::MarkerArray();
    
    bool last_node_visible = true;
    for (int i = 0; i < Y.rows(); i ++) {
        visualization_msgs::Marker cur_node_result = visualization_msgs::Marker();

        int dlo_index = i / nodes_per_dlo;
        std::vector<int> node_color = node_colors[dlo_index];
        std::vector<int> line_color = line_colors[dlo_index];
    
        // add header
        cur_node_result.header.frame_id = marker_frame;
        // cur_node_result.header.stamp = ros::Time::now();
        cur_node_result.type = visualization_msgs::Marker::SPHERE;
        cur_node_result.action = visualization_msgs::Marker::ADD;
        cur_node_result.ns = marker_ns + "_node_" + std::to_string(i);
        cur_node_result.id = i;

        // add position
        cur_node_result.pose.position.x = Y(i, 0);
        cur_node_result.pose.position.y = Y(i, 1);
        cur_node_result.pose.position.z = Y(i, 2);

        // add orientation
        cur_node_result.pose.orientation.w = 1.0;
        cur_node_result.pose.orientation.x = 0.0;
        cur_node_result.pose.orientation.y = 0.0;
        cur_node_result.pose.orientation.z = 0.0;

        // set scale
        cur_node_result.scale.x = node_scale;
        cur_node_result.scale.y = node_scale;
        cur_node_result.scale.z = node_scale;

        // set color
        bool cur_node_visible;
        if (visible_nodes.size() != 0 && std::find(visible_nodes.begin(), visible_nodes.end(), i) == visible_nodes.end()) {
            cur_node_result.color.r = occluded_node_color[0];
            cur_node_result.color.g = occluded_node_color[1];
            cur_node_result.color.b = occluded_node_color[2];
            cur_node_result.color.a = occluded_node_color[3];
            cur_node_visible = false;
        }
        else {
            cur_node_result.color.r = static_cast<double>(node_color[0]) / 255.0;
            cur_node_result.color.g = static_cast<double>(node_color[1]) / 255.0;
            cur_node_result.color.b = static_cast<double>(node_color[2]) / 255.0;
            cur_node_result.color.a = static_cast<double>(node_color[3]) / 255.0;
            cur_node_visible = true;
        }

        results.markers.push_back(cur_node_result);

        // don't add line if at the first node
        if (i == 0 || i % nodes_per_dlo == 0) {
            continue;
        }

        visualization_msgs::Marker cur_line_result = visualization_msgs::Marker();

        // add header
        cur_line_result.header.frame_id = marker_frame;
        cur_line_result.type = visualization_msgs::Marker::CYLINDER;
        cur_line_result.action = visualization_msgs::Marker::ADD;
        cur_line_result.ns = marker_ns + "_line_" + std::to_string(i);
        cur_line_result.id = i;

        // add position
        cur_line_result.pose.position.x = (Y(i, 0) + Y(i-1, 0)) / 2.0;
        cur_line_result.pose.position.y = (Y(i, 1) + Y(i-1, 1)) / 2.0;
        cur_line_result.pose.position.z = (Y(i, 2) + Y(i-1, 2)) / 2.0;

        // add orientation
        Eigen::Quaternionf q;
        Eigen::Vector3f vec1(0.0, 0.0, 1.0);
        Eigen::Vector3f vec2(Y(i, 0) - Y(i-1, 0), Y(i, 1) - Y(i-1, 1), Y(i, 2) - Y(i-1, 2));
        q.setFromTwoVectors(vec1, vec2);

        cur_line_result.pose.orientation.w = q.w();
        cur_line_result.pose.orientation.x = q.x();
        cur_line_result.pose.orientation.y = q.y();
        cur_line_result.pose.orientation.z = q.z();

        // set scale
        cur_line_result.scale.x = line_scale;
        cur_line_result.scale.y = line_scale;
        cur_line_result.scale.z = pt2pt_dis(Y.row(i), Y.row(i-1));

        // set color
        if (last_node_visible && cur_node_visible) {
            cur_line_result.color.r = static_cast<double>(line_color[0]) / 255.0;
            cur_line_result.color.g = static_cast<double>(line_color[1]) / 255.0;
            cur_line_result.color.b = static_cast<double>(line_color[2]) / 255.0;
            cur_line_result.color.a = static_cast<double>(line_color[3]) / 255.0;
        }
        else {
            cur_line_result.color.r = occluded_line_color[0];
            cur_line_result.color.g = occluded_line_color[1];
            cur_line_result.color.b = occluded_line_color[2];
            cur_line_result.color.a = occluded_line_color[3];
        }

        results.markers.push_back(cur_line_result);
    }

    return results;
}

// overload function
visualization_msgs::MarkerArray MatrixXd2MarkerArray (std::vector<MatrixXd> Y,
                                                      std::string marker_frame, 
                                                      std::string marker_ns, 
                                                      std::vector<float> node_color, 
                                                      std::vector<float> line_color, 
                                                      double node_scale,
                                                      double line_scale,
                                                      std::vector<int> visible_nodes, 
                                                      std::vector<float> occluded_node_color,
                                                      std::vector<float> occluded_line_color) {
    // publish the results as a marker array
    visualization_msgs::MarkerArray results = visualization_msgs::MarkerArray();

    bool last_node_visible = true;
    for (int i = 0; i < Y.size(); i ++) {
        visualization_msgs::Marker cur_node_result = visualization_msgs::Marker();

        int dim = Y[0].cols();
    
        // add header
        cur_node_result.header.frame_id = marker_frame;
        // cur_node_result.header.stamp = ros::Time::now();
        cur_node_result.type = visualization_msgs::Marker::SPHERE;
        cur_node_result.action = visualization_msgs::Marker::ADD;
        cur_node_result.ns = marker_ns + "_node_" + std::to_string(i);
        cur_node_result.id = i;

        // add position
        cur_node_result.pose.position.x = Y[i](0, dim-3);
        cur_node_result.pose.position.y = Y[i](0, dim-2);
        cur_node_result.pose.position.z = Y[i](0, dim-1);

        // add orientation
        cur_node_result.pose.orientation.w = 1.0;
        cur_node_result.pose.orientation.x = 0.0;
        cur_node_result.pose.orientation.y = 0.0;
        cur_node_result.pose.orientation.z = 0.0;

        // set scale
        cur_node_result.scale.x = 0.01;
        cur_node_result.scale.y = 0.01;
        cur_node_result.scale.z = 0.01;

        // set color
        bool cur_node_visible;
        if (visible_nodes.size() != 0 && std::find(visible_nodes.begin(), visible_nodes.end(), i) == visible_nodes.end()) {
            cur_node_result.color.r = occluded_node_color[0];
            cur_node_result.color.g = occluded_node_color[1];
            cur_node_result.color.b = occluded_node_color[2];
            cur_node_result.color.a = occluded_node_color[3];
            cur_node_visible = false;
        }
        else {
            cur_node_result.color.r = node_color[0];
            cur_node_result.color.g = node_color[1];
            cur_node_result.color.b = node_color[2];
            cur_node_result.color.a = node_color[3];
            cur_node_visible = true;
        }

        results.markers.push_back(cur_node_result);

        // don't add line if at the first node
        if (i == 0) {
            continue;
        }

        visualization_msgs::Marker cur_line_result = visualization_msgs::Marker();

        // add header
        cur_line_result.header.frame_id = marker_frame;
        cur_line_result.type = visualization_msgs::Marker::CYLINDER;
        cur_line_result.action = visualization_msgs::Marker::ADD;
        cur_line_result.ns = marker_ns + "_line_" + std::to_string(i);
        cur_line_result.id = i;

        // add position
        cur_line_result.pose.position.x = (Y[i](0, dim-3) + Y[i-1](0, dim-3)) / 2.0;
        cur_line_result.pose.position.y = (Y[i](0, dim-2) + Y[i-1](0, dim-2)) / 2.0;
        cur_line_result.pose.position.z = (Y[i](0, dim-1) + Y[i-1](0, dim-1)) / 2.0;

        // add orientation
        Eigen::Quaternionf q;
        Eigen::Vector3f vec1(0.0, 0.0, 1.0);
        Eigen::Vector3f vec2(Y[i](0, dim-3) - Y[i-1](0, dim-3), Y[i](0, dim-2) - Y[i-1](0, dim-2), Y[i](0, dim-1) - Y[i-1](0, dim-1));
        q.setFromTwoVectors(vec1, vec2);

        cur_line_result.pose.orientation.w = q.w();
        cur_line_result.pose.orientation.x = q.x();
        cur_line_result.pose.orientation.y = q.y();
        cur_line_result.pose.orientation.z = q.z();

        // set scale
        cur_line_result.scale.x = 0.005;
        cur_line_result.scale.y = 0.005;
        cur_line_result.scale.z = sqrt(pow(Y[i](0, dim-3) - Y[i-1](0, dim-3), 2) + pow(Y[i](0, dim-2) - Y[i-1](0, dim-2), 2) + pow(Y[i](0, dim-1) - Y[i-1](0, dim-1), 2));

        // set color
        if (last_node_visible && cur_node_visible) {
            cur_line_result.color.r = line_color[0];
            cur_line_result.color.g = line_color[1];
            cur_line_result.color.b = line_color[2];
            cur_line_result.color.a = line_color[3];
        }
        else {
            cur_line_result.color.r = occluded_line_color[0];
            cur_line_result.color.g = occluded_line_color[1];
            cur_line_result.color.b = occluded_line_color[2];
            cur_line_result.color.a = occluded_line_color[3];
        }

        results.markers.push_back(cur_line_result);
    }

    return results;
}

MatrixXd cross_product (MatrixXd vec1, MatrixXd vec2) {
    MatrixXd ret = MatrixXd::Zero(1, 3);
    
    ret(0, 0) = vec1(0, 1)*vec2(0, 2) - vec1(0, 2)*vec2(0, 1);
    ret(0, 1) = -(vec1(0, 0)*vec2(0, 2) - vec1(0, 2)*vec2(0, 0));
    ret(0, 2) = vec1(0, 0)*vec2(0, 1) - vec1(0, 1)*vec2(0, 0);

    return ret;
}

double dot_product (MatrixXd vec1, MatrixXd vec2) {
    return vec1(0, 0)*vec2(0, 0) + vec1(0, 1)*vec2(0, 1) + vec1(0, 2)*vec2(0, 2);
}