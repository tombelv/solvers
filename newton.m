commandwindow
clear

iters = 1;
x_history = [];
kkt_violation_history = [];
alpha_history = [];

% INPUT
tol = 1e-6;
x_init = [8;2];
lambda_init = 0;

hessian_approx = 'EXACT';


% optimization variables and constraints dimensions
nx = 2;
ng = 1;

x = sym('x', [nx 1]);
lambda = sym('lambda', [ng 1]);

% set the cost symbolic expression f_sym as a function of x
f_sym = 0.5*(x-[1;0]).'*(x-[1;0]);
% set the equality constraints (
g_sym = 1-(x.')*x;


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
        error('to be implemented')
    otherwise
        disp("defaulted to EXACT hessian")
        B_sym = jacobian(jacobian(lagrangian_sym,x),x);
end


% generate the matlab functions
matlabFunction(f_sym, 'vars', {x}, 'file', 'f');
matlabFunction(g_sym, 'vars', {x}, 'file', 'g');
matlabFunction(nablaf_sym, 'vars', {x}, 'file', 'nablaf');
matlabFunction(nablag_sym, 'vars', {x}, 'file', 'nablag');
matlabFunction(nablaLagrangian_sym, 'vars', {x, lambda}, 'file', 'nablaLagrangian');
matlabFunction(B_sym, 'vars', {x, lambda}, 'file', 'B');



x_ = x_init;
lambda_ = lambda_init;

B_ = B(x_,lambda_);
nablaf_ = nablaf(x_);
nablag_ = nablag(x_);
g_ = g(x_);
f_ = f(x_);

nablaLagrangian_ = nablaLagrangian(x_,lambda_);

kkt_violation = norm([nablaLagrangian_.', g_], inf);

while kkt_violation > tol
    


    % newton direction
    dir = -[B_ nablag_; nablag_.' zeros(ng,ng)]\[nablaf_; g_];

    deltax_ = dir(1:nx);
    lambda_plus = dir(nx+1:end);
    
    % perform linesearch with Armijo condition
    alpha = linesearch_armijo(x_, f_, nablaf_,deltax_);

    x_ = x_ + alpha*deltax_;
    lambda_ = (1-alpha)*lambda_ + alpha*lambda_plus;


    
    B_ = B(x_,lambda_);
    nablaf_ = nablaf(x_);
    nablag_ = nablag(x_);
    g_ = g(x_);
    f_ = f(x_);
    nablaLagrangian_ = nablaLagrangian(x_,lambda_);
    
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









figure(3)
semilogy(kkt_violation_history, 'lineWidth', 2), grid on
xlabel("Iteration")
title("KKT violation")
