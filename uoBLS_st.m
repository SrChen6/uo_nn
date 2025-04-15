% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F.-Javier Heredia https://gnom.upc.edu/heredia
% https://creativecommons.org/licenses/by-nc/4.0/
%
% [al,ACout] = uoBLS(x,d,P,par)
%
% Algorithm BLS: Backtracking Line Search with Wolfe conditions
%
% Input arguments:
% x           : x^k
% d           : d^k
% P           : problem.
% par         : BLS parameters
%   par.almin : al^min
%   par.almax : al^max
%   par.c1    : c1
%   par.c2    : c2
%   par.rho   : rho
%   par.iAC   : acceptability conditions
%             :   = 1: WC
%             :   = 2: SWC
%             :   = 3: WC + DC (for CGM-PR+)
%
% Output arguments
% al    : alpha^k
% ACout : acceptability of alpha^k:
%       :   = "WC1": al satisfies (WC1)
%       :   = "WC" : al satisfies (WC)
%       :   = "SWC": al satisfies (SWC)
%       :   = "WCD": al satisfies (WC)+(DC)
%       :   = "***": al satisfies no AC.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [al,ACout] = uoBLS_st(x,d,P,par)
%
f = P.f;
g = P.g;
WC1  = @(al) f(x+al*d) <= f(x)+par.c1*al*g(x)'*d;
WC2  = @(al) g(x+al*d)'*d >= par.c2*g(x)'*d;
SWC2 = @(al) abs(g(x+al*d)'*d) <= par.c2*abs(g(x)'*d);
al   = par.almax;
al_min = par.almin;
%
%  YOUR CODE HERE. Bucle de "until no double worlf conditions
if par.iAC == 2
    cond = @(al) WC1(al) & SWC2(al);
else
    cond = @(al) WC1(al) & WC2(al);
end

while (~(cond(al)) & al > al_min)
    al = al * par.rho;
end

%
ACout = "***";
if WC1(al)                                   ACout = "WC1"; end
if WC1(al) & WC2(al)                         ACout = "WC"; end
if WC1(al) & SWC2(al)                        ACout = "SWC"; end
end
% [end] uoBLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%