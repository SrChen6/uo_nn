clear all; warning off;
fopen('uo_nn_batch.log','w'); diary uo_nn_batch.log
fprintf('[uo_nn_batch]  Starts\n');
%
% Parameters
%
% NN model:''''''''''''''''''''''''''''''''''''''
nn.tr_seed = 7431987; nn.te_seed = 53866934; nn.sg_seed = 74315386; % Seeds.
% tr_seed is Haokang's DNI, te_seed is Laia's and sg_seed is the first 4
% digits of each DNI
nn.tr_p = 25000; nn.te_q = nn.tr_p /10; nn.tr_freq = 0.5;      % Datasets 
% Training
par.epsG = 10^-2; par.maxiter = 100;                           % Stopping cond.
par.iAC = 4; par.c1 = 0.01; par.c2 = 0.9;                      % Linesearch.
par.almax = 1; par.almin = 10^-6; par.rho = 0.5; par.delta = 0.001;
par.sg.seed = nn.sg_seed; par.sg.al0 = 2; par.sg.be = 0.3;     % SGM
par.almax = 1;
par.sg.m = 10; par.sg.emax = 100; par.sg.eworse = 5;
par.log = 1;  % if =0, call to [uosolLog] cancelled.
% Aux. functions
sig    = @(X)   1./(1+exp(-X));
y      = @(X,w) sig(w'*sig(X));
nn.Acc = @(Xds,yds,wo) 100*sum(yds==round(y(Xds,wo)))/size(Xds,2);
%
% Runs
%
global iheader; iheader = 1;
fileID = fopen('uo_nn_batch.csv','w');
t1 = clock;
for num_target = [1:10]
    nn.num_target = num_target;
    for la = [0.0 0.05 0.1]
        % Loss function
        nn.la = la;
        nn.L  = @(w,Xds,yds) (norm(y(Xds,w)-yds)^2)/size(yds,2) + (la*norm(w)^2)/2;
        nn.gL = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w)))')/size(yds,2)+la*w;
        for isd = [1 3 7]
            par.isd = isd;
            [nnout] = uo_nn_solve_st(nn,par); %Funció a implementar. Es pot provar amb uo_nn_solve(nn, par).
            if iheader == 1
                fprintf(fileID,'num_target;      la; isd;  niter;     tex; tr_acc; te_acc;        L*;\n');
            end
            if ~isempty(nnout)
                fprintf(fileID,'         %1i; %7.4f;   %1i; %6i; %7.4f;  %5.1f;  %5.1f;  %8.2e;\n', mod(num_target,10), la, isd, nnout.niter, nnout.tex, nnout.tr_acc, nnout.te_acc, nnout.Lo);
            end
            iheader=0;
        end
    end
end
t2 = clock; total_t = etime(t2,t1);
fprintf('[uo_nn_batch]  Stops, wall time = %6.1f s.\n', total_t); fclose(fileID);
diary off
