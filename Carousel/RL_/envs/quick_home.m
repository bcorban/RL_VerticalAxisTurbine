
%                    -         CAROUSELL HOME         -                   % 

% PURPOSE : User friendly code to home the Carousel

%  @Author: Sébastien Le Fouest (20/07/21)

piv = false;
pitching = false;
%% Physical motion of Motor E (#1)
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
m(2).es = 20000*12/360; % aencoder step per degree
m(2).n  = 'F';
m(2).ls = true; % does this motor have a limit switch?
m(2).free = false; % can this motor spin freely or does it have angular constraints?
m(2).PVT = true; % does this motor need PVT commands to move?)

%% Physical motion of Motor A (#3) - ROTATING MIRROR SYSTEM
if piv
    gear_ratio = 25/12; 
    counts_per_rev = 4*4096;
    m(3).ms = counts_per_rev*gear_ratio/360; % motor step per degree
    m(3).n  = 'A';
    m(3).ls = true; % does this motor have a limit switch?
    m(3).free = true; % can this motor spin freely or does it have angular constraints?
    m(3).PVT = false; % does this motor need PVT commands to move?
end
%                             -------------                               %

disp('Initiating the motors')
run ('carousell_motor_ini')
g.command(['SH' AllLsNam(m)]);    
pos = [0 0 0]; % Position relative to reference position [deg]
disp('Homing the motors')
CarousellHome(g, m, offset, 'pos', pos, 'reset', true, 'slowJG', 5, 'fastJG', 50, 'brushed', false); % JG speed in degree per seconds
if piv == 1
waitError(g,m(3))  
end