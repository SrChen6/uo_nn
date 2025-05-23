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
tic %Stopwatch start

% Terminal outputs
fprintf("::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
fprintf("function uo_nn_solve_st called\n")
fprintf("::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")

fprintf("Training data set generation.\n")
fprintf("\tnum_target\t=%d\n", mod(nn.num_target, 10))
fprintf("\ttr_freq\t=%f\n", nn.tr_freq)
fprintf("\ttr_p\t=%d\n", nn.tr_p)
fprintf("\ttr_seed\t=%d\n", nn.tr_seed)

fprintf("Test data set generation\n")
fprintf("\t te_q     \t=%d\n", nn.te_q)
fprintf("\t te_seed\t=%d\n", nn.te_seed)

fprintf("Optimization\n")
fprintf("\tL2 reg. lambda = %f\n", nn.la)
fprintf("\tw1 = [0]")
fprintf("\tCall uosol.\n")


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
par.sg.Xtr = Xtr; par.sg.ytr = ytr;
par.sg.Xte = Xte; par.sg.yte = yte;
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
tr_acc = nn.Acc(Xtr, ytr, w);
%

%
% Test accuracy
te_acc = nn.Acc(Xte, yte, w);
%

nnout.Xtr    = Xtr;
nnout.ytr    = ytr;
nnout.wo     = w;
nnout.Lo     = Lo;
nnout.niter  = niter;
nnout.tex    = toc;
nnout.tr_acc = tr_acc;
nnout.Xte    = Xte;
nnout.yte    = yte;
nnout.te_acc = te_acc;

% Results output on terminal
fprintf("\tOptimization wall time = %f\n", nnout.tex)
w = reshape(1:35, 7, 5);
fprintf("\two = [\n")
for i = 1:size(w, 1)
    fprintf('\t%+5.1e,%+5.1e,%+5.1e,%+5.1e,%+5.1e\n', w(i, :));
end
fprintf("\t     ]\n")
fprintf("Accuracy.\n")
fprintf("\ttr_accuracy = %f\n", tr_acc)
fprintf("\tte_accuracy = %f\n", te_acc)

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
