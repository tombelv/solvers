function alpha = linesearch_merit_ip(alpha_, x_, sigma_, s_, tau_, m1_, nablam1_, deltax_)

beta = 0.5;
gamma = 0.1;

alpha = alpha_;

    while m1(x_ + alpha*deltax_, sigma_, s_, tau_) >= m1_ + gamma*alpha*nablam1_

        alpha = beta*alpha;
    end
    
end