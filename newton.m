
commandwindow
clear all

% INPUT
tol = 1e-6;
x_init = [8;2;3];
lambda_init = -1;


nx = 3;
ng = 1;

x = sym('x', [nx 1]);
lambda = sym('lambda', [ng 1]);


f_sym = 0.5*(x-[1;0;0]).'*(x-[1;0;0]);

g_sym = x(1)-1;

nablaf_sym = gradient(f_sym, x);
nablag_sym = gradient(g_sym, x);



lagrangian_sym = f_sym + lambda.'*g_sym;

nablalagrangian_sym = gradient(lagrangian_sym, x);

exactHessian = 1;

if exactHessian == 1
    B_sym = jacobian(jacobian(lagrangian_sym,x),x);
    
else
    disp('ciao')
end


matlabFunction(f_sym, 'vars', {x}, 'file', 'f');
matlabFunction(g_sym, 'vars', {x}, 'file', 'g');
matlabFunction(nablaf_sym, 'vars', {x}, 'file', 'nablaf');
matlabFunction(nablag_sym, 'vars', {x}, 'file', 'nablag');
matlabFunction(nablalagrangian_sym, 'vars', {x, lambda}, 'file', 'nablalagrangian');
matlabFunction(B_sym, 'vars', {x, lambda}, 'file', 'B');

%%
x_ = x_init;
lambda_ = lambda_init;


nablalagrangian_ = nablalagrangian(x_,lambda_);

iters = 1;

B_ = B(x_,lambda_);
nablaf_ = nablaf(x_);
nablag_ = nablag(x_);
g_ = g(x_);
f_ = f(x_);

while norm([nablalagrangian_.', g_], inf) > tol
    



    % newton direction
    dir = -[B_ nablag_; nablag_.' zeros(ng,ng)]\[nablaf_; g_];

    deltax_ = dir(1:nx);
    lambda_plus = dir(nx+1:end);

    alpha = linesearch(x_, f_, nablaf_,deltax_);


    x_ = x_ + alpha*deltax_;
    lambda_ = (1-alpha)*lambda_ + alpha*lambda_plus;


    nablalagrangian_ = nablalagrangian(x_,lambda_);
    
    B_ = B(x_,lambda_);
    nablaf_ = nablaf(x_);
    nablag_ = nablag(x_);
    g_ = g(x_);
    f_ = f(x_);
    
    
    disp("-------------------------------------------------------------")
    disp("iteration: " + iters)
    
    disp("cost: " + f_)
    disp("x: " + x_)
    disp("lambda: " + lambda_)
    disp("alpha: " + alpha)
    
    disp("KKT violation: " + norm([nablalagrangian_.', g_], inf))

    
    iters = iters + 1;
end

