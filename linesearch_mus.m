function alpha = linesearch_mus(s_, mu_, delta_s_, delta_mu_)
    epsilon = 0.01;
    beta = 0.8;
    alpha = 1;
    epsilon_s_ = epsilon*s_;
    epsilon_mu_ = epsilon*mu_;
    while not(s_ + alpha*delta_s_ >= epsilon_s_ && mu_ + alpha*delta_mu_ >= epsilon_mu_)
        alpha = alpha * beta;
    end
end