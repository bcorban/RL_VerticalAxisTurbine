function [Cp] = Cp_vawt_inst(Ft, Mz, dpitch, param)
%% Compute the INSTANTANEOUS (not averaged) power coefficient from VAWT load measurements and parameters.

% Mean power generated [W]
mPgen = Ft*param.R*param.rotf*2*pi; % mean power generated [W]

% Mean motor power [W]
mPm = abs(Mz.*dpitch);

% Power available in the flow 0.5*rho*U^3*sweptA_blade 
mPw = 0.5*param.rho*param.Uinf^3*param.span*param.R*2;

Cp = (mPgen-mPm) ./ mPw;
end
