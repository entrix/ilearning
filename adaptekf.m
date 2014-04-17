function [out1,out2,out3] = adaptekf(in1,in2,in3,in4,in5,in6,in7)
%ADAPTWB Sequential order incremental adaption w/learning functions.
%
%  [NET,AR,AC] = <a href="matlab:doc adaptwb">adaptwb</a>(NET,PD,T,AI) takes a network, delayed inputs,
%  targets, and initial layer states, and returns the updated network,
%  adaption record, and layer outputs after applying the network's
%  weight and bias learning rules for each timestep in T.
%
%  <a href="matlab:doc adaptwb">adaptwb</a> is not commonly called directly, it is called by ADAPT when
%  the network's adaption function net.<a href="matlab:doc nnproperty.net_adaptFcn">adaptFcn</a> is set to <a href="matlab:doc adaptwb">adaptwb</a>.

% Mark Beale, 11-31-97
% Copyright 1992-2012 The MathWorks, Inc.
% $Revision: 1.1.10.8 $  $Date: 2012/08/21 01:03:31 $

% TODO - Replace PD with Xc, TAPDELAY, return Yc?

%% =======================================================
%  BOILERPLATE_START
%  This code is the same for all Adapt Functions.

  persistent INFO;
  if isempty(INFO), INFO = get_info; end
  if (nargin < 1), error(message('nnet:Args:NotEnough')); end
  if ischar(in1)
    switch (in1)
      case 'info'
        out1 = INFO;
      case 'check_param'
        out1 = check_param(in2);
      otherwise,
        try
          out1 = eval(['INFO.' in1]);
        catch me, nnerr.throw(['Unrecognized first argument: ''' in1 ''''])
        end
    end
  else
    [out1,out2,out3] = adapt_network(in1,in2,in3,in4,in5);
  end
end

%  BOILERPLATE_END
%% =======================================================

function [net,Ac,tr] = adapt_network(net,X,PD,T,Ai)
  Q = numsamples(T);
  TS = numtimesteps(T);
  hints = nn7.netHints(net);
  hints = nn.connections(net,hints);
  hints.simLayerOrder = nn.layer_order(net);
  hints.outputInd = find(net.outputConnect);
  
  % Calculation Mode
  calcMode = nncalc.defaultMode(net,Ai);
%   calcMode.options = nnet.options.calc.defaults;
  calcMode.options.showResources = 'yes';
  
  % Constants
%   numLayers = net.numLayers;
%   numInputs = net.numInputs;
%   performFcn = net.performFcn;
%   performParam = net.performParam;
%   needGradient = nn.needsGradient(net);
%   numLayerDelays = net.numLayerDelays;

  %Signals
  Ac = [Ai cell(net.numLayers,TS)];
    
  % Initialize
  tr.timesteps = 1:TS;
  tr.perf = zeros(1,TS);
  
  WB = getwb(net);
  lengthWB = length(WB);
  
  e = net.userdata.e;
  nu = net.userdata.nu;
  q = net.userdata.q;
  norm_max = net.userdata.norm_max;
  isIncremental = net.userdata.isIncremental;
  toDiagonal = net.userdata.toDiagonal;
  
  P = (1 / e)  * eye( lengthWB );
  R = (1 / nu) * eye( lengthWB );
  Q = q        * eye( lengthWB );
  
  % Adapt
  for ts=1:TS
    % Extended Kalman Filter
    % **********************************************   

    %     while true
    [tr.perf(ts),Ewb,Jx,Y] = calcJeJx(net, X(ts), PD(1,:,ts), Ai, T(ts), {1});
    
    if ~isIncremental
        P = (1 / e)  * eye( lengthWB );
    elseif toDiagonal
        P = toDiag(P);
    end
    
    %             Q = eye( calcHints )*1e-6;
    H = -Jx';
    normP = max(max(P));
    
    if(normP > 5*norm_max),
        P = P.*norm_max/normP;
    end
    
    % dbstop if warning
    I = eye( lengthWB );
    
    P = P + Q;
    S = H*P*H' + R;
    K = P*H'/S;
    P = (I-K*H)*P*(I-K*H)'+K*R*K';
    WB = WB + K*Ewb;
    
    fprintf('\titer = %d\tmse  = %f\n', ts, mse(net, cell2mat(T(ts)), Y));
    net = setwb(net, WB);
    % **********************************************
  end
end

function v = fcnversion
  v = 7;
end

%===============================================================

function info = get_info
  info = nnfcnAdaptive(mfilename,'Weight/Bias Rule Adaption',fcnversion,...
    true,true,[]);
end

function err = check_param(param)
  err = '';
end
