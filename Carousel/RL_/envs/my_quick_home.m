g = ConnectGalil('192.168.255.200');
% HOMING 
load('motor_offset','offset'); %obtained where?
load('param','param');
m=param.m;
pos = [0 0 0]; % Position relative to reference position [deg]
disp('Homing the motors')
CarousellHome(g, m, offset, 'pos', pos, 'reset', true, 'slowJG', 2, 'fastJG', 50, 'brushed', false); % JG speed in degree per seconds
% Set F encoder position to 0 (because it's not done by the homing)
g.command("DEF=0");