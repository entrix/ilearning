function [Y] = calcY(net,X,Xi,Ai,T,EW)
l = length(X{1});
n = length(Xi{1}) / length(X{1});
Xic = cell(1, n);
for i = 0:(n-1)
  Xic{i+1} = Xi{1}((i*l+1):((i+1)*l));
end
[net,rawData,~,~] = nntraining.setup(net,net.trainFcn,X,Xic,Ai,T,EW,false);
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