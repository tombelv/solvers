function reg_B = hessian_regularization(nablag_, B_)

epsilon = 1e-6;
Z = null(nablag_.');
[E, L] = eig(Z.'*B_*Z);
L_ = L;
for i = 1:length(L_)
    if L_(i,i) < epsilon
        L_(i,i) = epsilon;
    end
end
reg_B = B_ + Z*E*(L_-L)*E.'*Z.';
