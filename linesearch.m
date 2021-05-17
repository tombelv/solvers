function alpha = linesearch(x_, f_, nablaf_,deltax_)

beta = 0.9;
gamma = 0.00001;

alpha_ = 1;

    while f(x_ + alpha_*deltax_) >= f_ + gamma*alpha_*nablaf_.'*deltax_

        alpha_ = beta*alpha_
    end

alpha = alpha_;
    
end
