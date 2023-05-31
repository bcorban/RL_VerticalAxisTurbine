%% Physical motion of Motor E (#1) - MAIN ROTATION
YAE = 32;
m(1).ms = 200*YAE/360; % motor step per degree
m(1).es = 20000/360; % encoder step per degree
m(1).n  = 'E';
m(1).ls = true; % does this motor have a limit switch?
m(1).free = true; % can this motor spin freely or does it have angular constraints?
m(1).PVT = false; % does this motor need PVT commands to move?

%% Physical motion of Motor F (#2)
YAF = 16;
m(2).ms = 400*YAF*12/360; % motor step per degree (400 steps per revolution, 16 steps counts per full motor step and 1:12 gearbox reduction)
m(2).es = 20000*12/360; % encoder step per degree
m(2).n  = 'F';
m(2).ls = true; % does this motor have a limit switch?
m(2).free = false; % can this motor spin freely or does it have angular constraints?
m(2).PVT = true; % does this motor need PVT commands to move?

% Add to the parameters
% param.m = m;

%% Obtain experimental parameter - structure
% Input known parameters and compute rest with carousell_param
param.R      = 0.15; % Radius [m]
param.c      = 0.06; % Blade chord [m]
param.span   = 0.15; % m
param.Nb     = 1; % Number of blades [-]
param.lambda = 1.5; % Tip speed ratio 
param.Re     = 60000; % chord based Reynolds number
param.rho = 998.2;           % density in kg/m^3
param.mu =  0.0010005;          % dynamic viscosity in N s/m^2            
param.nu = param.mu/param.rho;       % kinematic viscosity in m^2/s
param = carousel_param(param); % Compute outputs based on given inputs
param.JG = round(param.rotf*m(1).ms*360);
param.rotT = 1 / param.rotf; % Rotation period in second
% Note:
% rotf: en # de rotation/secondes

%% Add random pulse!
% [param.phase_rnd_pulse, param.pitch_random_pulse] = get_random_pulse(param);

%% Motion parameters
% ------------------

% Define initial and total number of rotations
param.n_rot_act = 30; % Number of actuated rotations

% Recover the corresponding times
param.T_act = param.n_rot_act / param.rotf;

% Recover corresponding number of encoder steps
param.n_steps_act = param.n_rot_act * 360 * m(1).ms;

disp("    + Number of rotation (with actuation): "+ param.n_rot_act)

%% NI PARAMETERS
NI.rate= 1000; % [Hz] Scan rate for NI card
NI.channels = [0 1 2 3 4]; % Analog input channels to use
NI.device = 'cDAQ1Mod1Mod1'; %
% NI.calmat = 'carou_calimat4.mat'; % For Carouuuuuuuusel
% param.NI = NI;

%% Additional pre-processicng parameters, s.a. forces
% ------------------------------------

% General denominator for the forces
param.f_denom = 0.5.*param.rho.*(param.Ub).^2.*param.c.*param.span;

% Inertial force
param.Finertial = 0.04684*(param.rotf*2*pi)^2; % inertial force

% Drag offset created from taking zero measurement with flow
param.Csp = 0.09; % thin disk drag force coeffient (measured)
param.spr = 0.06; % splitter plate radius
CDnaca0 = 0.02;
param.F0 = (param.Uinf).^2*0.5*param.rho*(param.Csp*param.spr^2*pi*2+CDnaca0*param.c*param.span);


save('param.mat','param')


% Calibration matrices. 
% Notevariable: I keep only the positive matrix as I'm not interested in Mx and My.
% Also, columns 3 and 4 can be removed from the matrix (also because I
% don't need to compute Mx and My)
% load(NI.calmat,'R4');
% param.R4 = R4(:,[1,2,5]);
