commandwindow
clear

iters = 1;


% INPUT
tol = 1e-8;
% x_init = [1;1;0];
% lambda_init = [0;0];
x_init = [0;1];
lambda_init = 0;
mu_init = 0.1;
s_init = 0.1;
tau_init = 0.3;
sigma_coeff = 2;
sigma_init = 1;
damping_coeff = 0.5;

hessian_approx = 'EXACT';
linesearch = 'MERIT';


% optimization variables and constraints dimensions
nx = length(x_init);
ng = length(lambda_init);
nh = length(mu_init);
ns = nh;


x = sym('x', [nx 1]);
lambda = sym('lambda', [ng 1]);
mu = sym('mu', [nh 1]);
sigma = sym('sigma');
s = sym('s', [ns 1]);
tau = sym('tau');

R_sym = x - ones(nx,1);
%R_sym = x - [0;1;0];

% set the cost symbolic expression f_sym as a function of x
f_sym = 0.5*(R_sym.')*R_sym;
% set the equality constraints (
% g_sym = [x(1)^2 - 2*x(2)^3 - x(2) - 10*x(3); x(2) + 10*x(3)];
g_sym = 1 - x.' * x;
h_sym = 0.5-x(1)^2-x(2);
% set merit function
m1_sym = f_sym + sigma * norm(g_sym, 1) + sigma * norm(h_sym + s, 1) - tau*sum(log(s));


% compute Lagrangian and gradients
nablaf_sym = gradient(f_sym, x);
nablag_sym = jacobian(g_sym, x).';
nablah_sym = jacobian(h_sym, x).';
nablam1_sym = jacobian(m1_sym, x).';

lagrangian_sym = f_sym + lambda.'*g_sym + mu.'*h_sym;
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
matlabFunction(h_sym, 'vars', {x}, 'file', 'h');
matlabFunction(m1_sym, 'vars', {x, sigma, s, tau}, 'file', 'm1');
matlabFunction(nablaf_sym, 'vars', {x}, 'file', 'nablaf');
matlabFunction(nablag_sym, 'vars', {x}, 'file', 'nablag');
matlabFunction(nablah_sym, 'vars', {x}, 'file', 'nablah');
matlabFunction(nablam1_sym, 'vars', {x, sigma, s, tau}, 'file', 'nablam1');
matlabFunction(nablaLagrangian_sym, 'vars', {x, lambda, mu}, 'file', 'nablaLagrangian');
matlabFunction(B_sym, 'vars', {x, lambda, mu}, 'file', 'B');



x_ = x_init;
lambda_ = lambda_init;
mu_ = mu_init;
s_ = s_init;
sigma_ = sigma_init;
tau_ = tau_init;

B_ = B(x_,lambda_, mu_);
nablaf_ = nablaf(x_);
nablag_ = nablag(x_);
nablah_ = nablah(x_);
nablam1_ = nablam1(x_, sigma_, s_, tau_);
g_ = g(x_);
f_ = f(x_);
h_ = h(x_);
m1_ = m1(x_, sigma_, s_, tau_);

nablaLagrangian_ = nablaLagrangian(x_,lambda_, mu_);
r = [nablaLagrangian_; g_; h_ + s_; diag(mu_) * s_ - tau_];

kkt_violation = norm([nablaLagrangian_; g_; max(zeros(nh, 1), h_)], inf);

x_history = [x_];
kkt_violation_history = [kkt_violation];
alpha_history = [];
tau_history = [];

while kkt_violation > tol && tau_ > tol
    % regularization of the hessian
    B_ = hessian_regularization(nablag_, B_)  
    
    kkt_matrix = [B_ nablag_ nablah_ zeros(nx, ns); nablag_.' zeros(ng,ng) zeros(ng, nh) zeros(ng, ns);
        nablah_.' zeros(nh,ng) zeros(nh, nh) eye(nh); zeros(nh, nx) zeros(nh, ng) diag(s_) diag(mu_)];
    dir = -kkt_matrix\r;
    delta_x_ = dir(1:nx);
    delta_lambda_ = dir(nx+1);
    delta_mu_ = dir(nx+2);
    delta_s_ = dir(nx+3:end);

    %linesearch merit
    alpha_ = linesearch_mus(s_, mu_, delta_s_, delta_mu_);
    alpha = linesearch_merit_ip(alpha_, x_, sigma_, s_, tau_, m1_, nablam1_, delta_x_);

    x_ = x_ + alpha*delta_x_;
    lambda_ = lambda_ + alpha*delta_lambda_;
    mu_ = mu_ + alpha*delta_mu_;
    s_ = s_ + alpha*delta_s_;
    
    %decrease the barrier parameter
    if (norm(r, inf) <= tau_) 
        tau_ = max(0.1*tau_, tol);
    end

    B_ = B(x_,lambda_, mu_);
    nablaf_ = nablaf(x_);
    nablag_ = nablag(x_);
    g_ = g(x_);
    f_ = f(x_);
    nablaLagrangian_ = nablaLagrangian(x_,lambda_, mu_);
    r = [nablaLagrangian_; g_; h_ + s_; diag(mu_) * s_ - tau_];
    if (sigma_coeff*lambda_ > sigma_) 
        sigma_ = sigma_coeff*lambda_;
    end
    m1_ = m1(x_, sigma_, s_, tau_);
    nablam1_ = nablam1(x_, sigma_, s_, tau_);
    kkt_violation = norm([nablaLagrangian_; g_; max(zeros(nh, 1), h_)], inf);
    
    
    disp("-------------------------------------------------------------")
    disp("iteration: " + iters)
    disp("KKT violation: " + kkt_violation)
    
    disp("x: ")
    disp(x_)
    disp("lambda: " + lambda_)
    disp("cost: " + f_)
    disp("alpha: " + alpha)
    disp("m1: " + m1_)
    
    x_history = [x_history, x_];
    kkt_violation_history = [kkt_violation_history, kkt_violation];
    alpha_history = [alpha_history, alpha];
    tau_history = [tau_history, tau_];
    
    
    iters = iters + 1;
    
    
end

%%

figure(1)
rectangle('Position',[-1 -1 2 2],'Curvature',[1 1], 'lineWidth', 1), hold on
x1 = linspace(-1,1,100);
x2 = 0.5 - x1.^2;
plot(x1, x2)
plot(x_history(1,:),x_history(2,:), 'lineWidth', 1,'Marker', 'o')
axis equal
xlabel("x_1")
ylabel("x_2")
figure(2)
plot(alpha_history, 'lineWidth', 1.5, 'Marker', 'x')
xlabel("Iteration")
title("alpha history")
figure(3)
plot(tau_history, 'lineWidth', 1.5, 'Marker', 'x')
xlabel("Iteration")
title("tau history")
figure(4)
semilogy(kkt_violation_history, 'lineWidth', 1.5), grid on
xlabel("Iteration")
title("KKT violation")