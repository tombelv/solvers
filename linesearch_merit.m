function alpha = linesearch_merit(x_, sigma_, m1_, nablam1_, deltax_)

beta = 0.5;
gamma = 0.1;

alpha_ = 1;

    while m1(x_ + alpha_*deltax_, sigma_) >= m1_ + gamma*alpha_*nablam1_

        alpha_ = beta*alpha_;
    end

alpha = alpha_;
    
end