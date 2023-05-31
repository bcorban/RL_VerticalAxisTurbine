NI.rate= 1000; % [Hz] Scan rate for NI card
NI.channels = [0 1 2 3 4]; % Analog input channels to use
NI.device = 'cDAQ1Mod1Mod1'; %
NI.calmat = 'carou_calimat4.mat'; % For Carouuuuuuuusel
param.NI = NI;

lc=loadcellg(NI);
lc.startLC()