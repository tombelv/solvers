commandwindow
clear

iters = 1;


% INPUT
tol = 1e-8;
x_init = [0.5;1];
lambda_init = 0;
sigma_coeff = 2;
sigma_init = 1;
damping_coeff = 0.5;

hessian_approx = 'GAUSS_NEWTON';
linesearch = 'MERIT';


% optimization variables and constraints dimensions
nx = 2;
ng = 1;



x = sym('x', [nx 1]);
lambda = sym('lambda', [ng 1]);
sigma = sym('sigma');

R_sym = x - ones(nx,1);

% set the cost symbolic expression f_sym as a function of x
f_sym = 0.5*(R_sym.')*R_sym;
% set the equality constraints (
g_sym = 1-(x.')*x;
% set merit function
m1_sym = f_sym + sigma * norm(g_sym, 1);


% compute Lagrangian and gradients
nablaf_sym = gradient(f_sym, x);
nablag_sym = gradient(g_sym, x);

lagrangian_sym = f_sym + lambda.'*g_sym;
nablaLagrangian_sym = gradient(lagrangian_sym, x);

% compute the hessian B according to the chosen hessian approximation
switch hessian_approx
    case 'EXACT'
        B_sym = jacobian(jacobian(lagrangian_sym,x),x);
    case 'GAUSS_NEWTON'
        nablaR = jacobian(R_sym,x);
        B_sym = nablaR*(nablaR.');
    otherwise
        disp("defaulted to EXACT hessian")
        B_sym = jacobian(jacobian(lagrangian_sym,x),x);
end


% generate the matlab functions
matlabFunction(f_sym, 'vars', {x}, 'file', 'f');
matlabFunction(g_sym, 'vars', {x}, 'file', 'g');
matlabFunction(m1_sym, 'vars', {x, sigma}, 'file', 'm1');
matlabFunction(nablaf_sym, 'vars', {x}, 'file', 'nablaf');
matlabFunction(nablag_sym, 'vars', {x}, 'file', 'nablag');
matlabFunction(nablaLagrangian_sym, 'vars', {x, lambda}, 'file', 'nablaLagrangian');
matlabFunction(B_sym, 'vars', {x, lambda}, 'file', 'B');



x_ = x_init;
lambda_ = lambda_init;
sigma_ = sigma_init;

B_ = B(x_,lambda_);
nablaf_ = nablaf(x_);
nablag_ = nablag(x_);
g_ = g(x_);
f_ = f(x_);
m1_ = m1(x_, sigma_);

nablaLagrangian_ = nablaLagrangian(x_,lambda_);

kkt_violation = norm([nablaLagrangian_.', g_], inf);

x_history = [x_];
kkt_violation_history = [kkt_violation];
alpha_history = [];

while kkt_violation > tol
    


    % newton direction
    kkt_matrix = [B_ nablag_; nablag_.' zeros(ng,ng)]; 
    if (det(kkt_matrix) == 0)
        kkt_matrix = kkt_matrix + eye(size(kkt_matrix,1)) * damping_coeff;
    end
    dir = -kkt_matrix\[nablaf_; g_];

    deltax_ = dir(1:nx);
    lambda_plus = dir(nx+1:end);
    disp(lambda_plus)

    switch linesearch
        case 'MERIT'
            % perform linesearch with merit function
            nablam1_ = nablaf_.' * deltax_ - sigma_*norm(g_, 1);
            alpha = linesearch_merit(x_, sigma_, m1_, nablam1_,deltax_);
        case 'ARMIJO'
            % perform linesearch with Armijo condition
            alpha = linesearch_armijo(x_, f_, nablaf_,deltax_);
        otherwise
            % perform linesearch with Armijo condition
            alpha = linesearch_armijo(x_, f_, nablaf_,deltax_);
    end

    

    x_ = x_ + alpha*deltax_;
    lambda_ = (1-alpha)*lambda_ + alpha*lambda_plus;


    
    B_ = B(x_,lambda_);
    nablaf_ = nablaf(x_);
    nablag_ = nablag(x_);
    g_ = g(x_);
    f_ = f(x_);
    nablaLagrangian_ = nablaLagrangian(x_,lambda_);
    if (sigma_coeff*lambda_ > sigma_) 
        sigma_ = sigma_coeff*lambda_;
    end
    m1_ = m1(x_, sigma_);
    kkt_violation = norm([nablaLagrangian_.', g_], inf);
    
    
    disp("-------------------------------------------------------------")
    disp("iteration: " + iters)
    disp("KKT violation: " + kkt_violation)
    
    disp("x: ")
    disp(x_)
    disp("lambda: " + lambda_)
    disp("cost: " + f_)
    disp("alpha: " + alpha)
    
    x_history = [x_history, x_];
    kkt_violation_history = [kkt_violation_history, kkt_violation];
    alpha_history = [alpha_history, alpha];
    
    
    iters = iters + 1;
    
    
end

%%

figure(1)
rectangle('Position',[-1 -1 2 2],'Curvature',[1 1], 'lineWidth', 2), hold on
plot(x_history(1,:),x_history(2,:), 'lineWidth', 2,'Marker', 'o')
axis equal
xlabel("x_1")
ylabel("x_2")
figure(2)


plot(alpha_history, 'lineWidth', 2, 'Marker', 'x')
xlabel("Iteration")
title("alpha history")
figure(3)


semilogy(kkt_violation_history, 'lineWidth', 2), grid on
xlabel("Iteration")
title("KKT violation")
