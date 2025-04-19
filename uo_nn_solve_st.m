%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OM / GCED / F.-Javier Heredia https://gnom.upc.edu/heredia
% Function uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Input parameters:
%
% nn:
%          L : loss function.
%         gL : gradient of the loss function.
%        Acc : Accuracy function.
% num_target : set of digits to be identified.
%    tr_freq : frequency of the digits target in the data set.
%    tr_seed : seed for the training set random generation.
%       tr_p : size of the training set.
%    te_seed : seed for the test set random generation.
%       te_q : size of the test set.
%         la : coefficient lambda of the decay factor.
% par:
%       epsG : optimality tolerance.
%    maxiter : maximum number of iterations.
%      c1,c2 : (WC) parameters.
%        isd : optimization algorithm.
%     sg.al0 : \alpha^{SG}_0.
%      sg.be : \beta^{SG}.
%       sg.m : m^{SG}.
%    sg.emax : e^{SGÇ_{max}.
%   sg.eworse: e^{SG}_{worse}.
%    sg.seed : seed for the first random permutation of the SG.
%
% Output parameters:
%
% nnout
%    Xtr : X^{TR}.
%    ytr : y^{TR}.
%     wo : w^*.
%     Lo : {\tilde L}^*.
% tr_acc : Accuracy^{TR}.
%    Xte : X^{TE}.
%    yte : y^{TE}.
% te_acc : Accuracy^{TE}.
%  niter : total number of iterations.
%    tex : total running time (see "tic" "toc" Matlab commands).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [nnout] = uo_nn_solve_st(nn,par)
t1 = clock; %Maybe change for tic/toc


% Training dataset generation
[Xtr, ytr] = uo_nn_dataset(nn.tr_seed, nn.tr_p, nn.num_target, nn.tr_freq);
%

%
% Test dataset generation
[Xte, yte] = uo_nn_dataset(nn.te_seed, nn.te_q, nn.num_target, nn.tr_freq);
%

%
% Optimization
if par.isd == 7
    P.f = @(w, Xtr, ytr) nn.L(w, Xtr, ytr);
    P.g = @(w, Xtr, ytr) nn.gL(w, Xtr, ytr);
    P.h = @(w) eye(35); % hessian not needed (only first order methods)
    P.Xtr = Xtr; P.ytr = ytr;
else
P.f = @(w) nn.L(w, Xtr, ytr);
P.g = @(w) nn.gL(w, Xtr, ytr);
P.h = @(w) eye(35); % hessian not needed (only first order methods)
end

wo = zeros(35, 1);
par.sg.Xtr = Xtr;
par.sg.ytr = ytr;
[sol, ~] = uosol_st(P, wo, par);
w = sol(end).x;

if par.isd == 7
    Lo = P.f(w, Xtr, ytr);
else
    Lo = P.f(w);
end
niter = size(sol, 2); % Numero de interacions fetes


%
% Training accuracy 
tr_acc = nn.Acc(Xtr, ytr, w)
%

%
% Test accuracy
te_acc = nn.Acc(Xte, yte, w)
%


t2 = clock; 
tex = etime(t2,t1);

%
nnout.Xtr    = Xtr;
nnout.ytr    = ytr;
nnout.wo     = w;
nnout.Lo     = Lo;
nnout.niter  = niter;
nnout.tex    = tex;
nnout.tr_acc = tr_acc;
nnout.Xte    = Xte;
nnout.yte    = yte;
nnout.te_acc = te_acc;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
