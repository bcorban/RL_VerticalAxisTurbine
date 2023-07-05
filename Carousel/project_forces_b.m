function out1 = project_forces_b(forces, pitch, param)
%% Function to compute the loads
% INPUTS:
%  - forces
%  - phase: the phase angle of the carousel
%  - pitch: the pitch angle of the blade
%  - param: struct containing the required forces parameters
%    - param.Finertial
%    - param.Fsp        : the force from the splitter plate

% Forces
res.Fr = forces(:,1).*sind(pitch) + forces(:,2).*cosd(pitch) + param.Finertial;
res.Ft = forces(:,1).*cosd(pitch) - forces(:,2).*sind(pitch) + param.Fsp;
res.Ftot = hypot(res.Fr,res.Ft);
res.Ctot = res.Ftot ./ param.denom;
res.Cr = res.Fr ./ param.denom;
res.Ct = res.Ft ./ param.denom;

% Moments
res.Cm = forces(:, end) ./ (param.denom * param.c);
%     res.Cm_full = Mz_full ./ (param.denom * param.c);
% res.Cmx = forces(:, 3) ./ (param.denom * param.c);
% res.Cmy = forces(:, 4) ./ (param.denom * param.c);

% Function outputs
out1 = res;
% out2(:,1) = res.Ft;
% out2(:,2) = res.Fr;
% out2(:,3) = forces(:,3);
% out2(:,4) = forces(:,4);
% out2(:,5) = forces(:,5);

end