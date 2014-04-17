function [trainPerf,Je,Jx,Y] = calcJeJx(net,X,Xi,Ai,T,EW)
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
[trainPerf,~,~,Je,Jx,Y,~,~,~]=perfsJEJJ_2(net, calcData, hints);
end