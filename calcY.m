function [Y] = calcY(net,X,Xi,Ai,T,EW)
[net,rawData,~,~] = nntraining.setup(net,net.trainFcn,X,{Xi{1}(1),Xi{1}(2)},Ai,T,EW,false);
% Calculation Mode
calcMode = nncalc.defaultMode(net,Ai);
% calcMode.options = nnet.options.calc.defaults;
calcMode.options.showResources = 'yes';
% Hints
tools = nn7;
hints = tools.netHints(net,tools.hints);
hints.outputInd = find(net.outputConnect);

[~,~,calcData,~,net,~] = nncalc.setup1(calcMode,net,rawData);
data = nn7.y_all(net,calcData,hints);
Y = data.Y;
end