commandwindow
clear

% INPUT
tol = 1e-6;
x_init = [8;2;3];
lambda_init = 0;

hessian_approx = 'EXACT';


% optimization variables and constraints dimensions
nx = 3;
ng = 1;

x = sym('x', [nx 1]);
lambda = sym('lambda', [ng 1]);


f_sym = 0.5*(x-[1;0;0]).'*(x-[1;0;0]);
g_sym = 1-(x.')*x;


nablaf_sym = gradient(f_sym, x);
nablag_sym = gradient(g_sym, x);


lagrangian_sym = f_sym + lambda.'*g_sym;
nablaLagrangian_sym = gradient(lagrangian_sym, x);



switch hessian_approx
    case 'EXACT'
        B_sym = jacobian(jacobian(lagrangian_sym,x),x);
    case 'GAUSS_NEWTON'
        error('to be implemented')
    otherwise
        disp("defaulted to EXACT hessian")
        B_sym = jacobian(jacobian(lagrangian_sym,x),x);
end




matlabFunction(f_sym, 'vars', {x}, 'file', 'f');
matlabFunction(g_sym, 'vars', {x}, 'file', 'g');
matlabFunction(nablaf_sym, 'vars', {x}, 'file', 'nablaf');
matlabFunction(nablag_sym, 'vars', {x}, 'file', 'nablag');
matlabFunction(nablaLagrangian_sym, 'vars', {x, lambda}, 'file', 'nablaLagrangian');
matlabFunction(B_sym, 'vars', {x, lambda}, 'file', 'B');

%%
x_ = x_init;
lambda_ = lambda_init;


iters = 1;

B_ = B(x_,lambda_);
nablaf_ = nablaf(x_);
nablag_ = nablag(x_);
g_ = g(x_);
f_ = f(x_);

nablaLagrangian_ = nablaLagrangian(x_,lambda_);

while norm([nablaLagrangian_.', g_], inf) > tol
    


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
    
    
    disp("-------------------------------------------------------------")
    disp("iteration: " + iters)
    disp("KKT violation: " + norm([nablaLagrangian_.', g_], inf))
    
    disp("x: ")
    disp(x_)
    disp("lambda: " + lambda_)
    disp("cost: " + f_)
    disp("alpha: " + alpha)
    
    

    
    iters = iters + 1;
    
    
end

