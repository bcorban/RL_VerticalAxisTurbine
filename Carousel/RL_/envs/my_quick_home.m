cd('/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/RL_/envs')
g = ConnectGalil('192.168.255.200');
% HOMING 
load('motor_offset','offset'); 
load('/Users/PIVUSER/Desktop/RL_VerticalAxisTurbine/Carousel/param.mat','param');
m=param.m;
pos = [0 0 0]; % Position relative to reference position [deg]
disp('Homing the motors')
g.command("OEF=0");
CarousellHome(g, m, offset, 'pos', pos, 'reset', true, 'slowJG', 2, 'fastJG', 50, 'brushed', false); % JG speed in degree per seconds
% Set F encoder position to 0 (because it's not done by the homing)
g.command("OEF=2");
g.command("DEF=0");