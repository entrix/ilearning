ilearning
=========

Repository for some kinds of matlab programs which use to incremental (adaptive) learning processes.
----------------------------------------------------------------------------------------------------


Some notions about Matlab.

I'm starting use some legacy code in order to use Kalman Filter procedure to learn simple recurrent (time delay) network in the same way as with the Levenberg-Markwardt gradient procedure. And I encountered with problems of non-compatibility code. I want to touch on some of them.

In the past Matlab has functions like calcpd, calca, calce and calcperf for computing delayed network inputs/outputs, layer inputs/outputs and so on, but now all these features have been excluded.

_calcpd_ and _calca_ can be replaced by
```
  [Y,Af] = calcLib.y(calcNet);
```
_calcNet_ is network representaion after some initialization like
```
  [X,Xi,Ai,T,EW,~] = preparets(net,X,Tl);
  [net,rawData,~,~] = nntraining.setup(net,net.trainFcn,X,Xi,Ai,T,EW,false);
  %   end
  
  % Calculation Mode
  calcMode = nncalc.defaultMode(net,Ai);
  calcMode.options = nnet.options.calc.defaults;
  calcMode.options.showResources = 'yes';

  % Setup simulation/training calculation mode, network, data and hints
  [calcMode,calcNet,calcData,calcHints,~,~] = nncalc.setup1(calcMode,net,rawData);
  [calcLib,calcNet] = nncalc.setup2(calcMode,calcNet,calcData,calcHints);
```
*Y* and *Af* similar to the outputs of the sim function (*Y* for _calcpd_, *Af* for _calca_).

_calcperf_ can be replaced by 
```  
  perf2 = calcLib.trainPerf(calcNet);
```
with analougus procedures for initialization as mentioned above.

calce can be replaced by
```
  [~,~,~,Je,Jx,~] = calcLib.perfsJEJJ(calcNet);
```
where  *Je* correspond to each coeff in the network and *Jx* is a Jacobian.
These procedure initialized as well as above.

