%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F.-Javier Heredia https://gnom.upc.edu/heredia
% https://creativecommons.org/licenses/by-nc/4.0/
%
% [sol,par] = uosol_st(P,x1,par)
%
% Template for the unconstrained optimization with first and second
% derivative methods.
%
% See uolib.mlx for a description of the input/output arguments,
% standard output and calls.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [sol,par] = uosol_st(P,x,par)

%
% Initializations

if nargin < 3
    par.epsG = 10^-6;
    par.maxiter = 500;        
    par.iAC = 2;
    par.almax = 1;       
    par.almin = 10^-6;                
    par.rho =0.5;
    par.c1 = 0.01;
    par.c2 = 0.9;
    par.isd = 1; %1 for GM, 2 for CGM, 3 for BFGS, 4 for NM, 5 for MNM-SD, 6 for MNM.CMI
    par.delta = 0.001;
end
n=size(x,1);
f = P.f;
g = P.g;
h = P.h;
k = 1;
ldescent = true;
if par.isd == 7 % Initializations for SG
    Xtr = par.sg.Xtr;
    ytr = par.sg.ytr;
    gL = @(x, Xtr, ytr) g(x, Xtr, ytr); %Name change
    g = @(x) g(x, Xtr, ytr);
    p = size(par.sg.Xtr, 2);
    m = par.sg.m;
    k_e = ceil(p/m);
    k_max = par.sg.emax*k_e;
    L_best = inf;
    x_k = x;
    par.maxiter = inf;

    al_SG = 0.01*par.sg.al0;
    k_SG = floor(par.sg.be*k_max);

    e_best = 0; k_best = 0;
end

e = 0;
s = 0;
%
% Algorithm
%
while norm(g(x)) > par.epsG & k < par.maxiter & (ldescent | par.isd == 4 | par.isd == 7) & (e <= par.sg.emax & s < par.sg.eworse)
    if par.isd == 1
        % GM
        d = -g(x);
    elseif par.isd == 3
        % BFGS, Quasi-Newton
        if k == 1
            H = eye(n);
        else
            s = x - (x - al*d); %iteració actual (k) menys la anterior(k-1)
            y = g(x) - g(x - al*d);
            rho = (y'*s)^-1;
            H = (eye(n)-rho*s*y')*H*(eye(n)-rho*y*s')+rho*s*s';
        end
        d = -H*g(x);
        sol(k).H = H;
    elseif par.isd == 4
        % NM
        d = -h(x)^-1*g(x);
        sol(k).H = h(x);
    elseif par.isd == 5
        % MNM-SD
        delta = par.delta;
        [Q, La] = eig(h(x));
        Hat_La = diag(max(delta, diag(La)));
        B = Q*Hat_La*Q';
        d = -B^-1*g(x);
        sol(k).H = B;
    elseif par.isd == 6
        % MNM-CMI
        L_ub = norm(h(x), 'fro');
        tau = 0;
        l = 0;
        [R, PD] = chol(h(x)+tau*eye(n));
        while PD > 0
            l = l + 1;
            tau = (1.01 - 1/2^l)*L_ub;
            [R, PD] = chol(h(x)+tau*eye(n));
        end
        B = R' * R;
        d = -B^-1*g(x);
        sol(k).tau = tau;
        sol(k).H = B;
    elseif par.isd == 7
        % SGM

        Per = randperm(p);
        for i = 0:ceil(p/m-1)
            Set = Per(i*m+1:min((i+1)*m, p));
            X_Str = Xtr(:, Set);
            y_Str = ytr(:, Set);
            d = -gL(x_k, X_Str, y_Str);

            if k <= k_SG
                al = (1 - k/k_SG)*par.sg.al0 + k/k_SG*al_SG;
            else
                al = al_SG;
            end
            % uosolLog update
            sol(k).al = al;
            sol(k).f = f(x_k, Xtr, ytr);
            sol(k).x  = x_k;
            sol(k).d  = d;
            sol(k).AC = "";

            x_k = x_k+al*d;
            k = k + 1;
        end
        x = x_k;
    end
    ldescent = d'*g(x) < 0;
  
    if par.isd == 7 %Stochastic gradient
        e = e + 1;
        L = f(x, par.sg.Xte, par.sg.yte);
        if L < L_best
            L_best = L;
            s = 0;
            e_best = e;
            k_best = k;
        else
            s = s + 1;
        end
    else
        % LS
        if     par.isd == 4     % unit step length for the Newton method.
            al =1;
            ACout="";
        elseif par.iAC == 0     % Exact line search.
            al = -g(x)'*d/(d'*h(x)*d);
            ACout = "ELS";
        elseif par.iAC == 4
            if k == 1
                [al,ACout] = uoBLS_st(x,d,P,par); %In the first iteration alpha is not defined
            else
                par.almax = 2*(f(x)-f(x-al*d))/(g(x)'*d);
            end
            [al, iout] = uoBLSNW32(f, g, x, d, par.almax, par.c1, par.c2);
            if iout == 0
                ACout = "SWC";
            elseif iout == 1
                ACout = "30iter";
            elseif iout == 2
                ACout = "<10^-3";
            end
        elseif par.iAC <= 3      % BLS.
            [al,ACout] = uoBLS_st(x,d,P,par);
        end
    end
    if par.isd ~= 7 %For isd=7 these computations were already made
        sol(k).x  = x;
        sol(k).g  = g(x);
        sol(k).ng = norm(g(x));
        sol(k).d  = d;
        sol(k).al = al;
        sol(k).AC = ACout;
        x = x + al*d; k=k+1;
    end

end %............................................................ main loop
sol(k).x  = x;
sol(k).g  = g(x);
sol(k).ng = norm(g(x));
%
% Iterations log
%

if par.log ~= 0
    if par.isd == 7
        iterinfo.eo = e_best;
        iterinfo.etot = e;
        iterinfo.ko = k_best; % TODO: write a coorect implementation
        iterinfo.ktot = k;
        [sol] = uosolLog(P,par,sol, iterinfo);
    else
        [sol] = uosolLog(P,par,sol);
    end
end

% [end] Function [uosol_st] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

